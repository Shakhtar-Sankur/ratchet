"""The trainer's copy of the policy: a Llama/Qwen2 decoder in PyTorch whose parameters
have exactly relay's names and fused layout (wqkv, w_gate_up, ...), so a weight sync is
a copy, tensor by tensor, with no reshaping or renaming. It is built from a relay.Model
(same weights, same rotary frequencies) and computes the same function as relay's
backends: in float32 its log-probabilities match relay's CPU rollouts to rounding."""

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class _Rounded(torch.autograd.Function):
    """The weight rounded to `dtype` and back (round to nearest even, as relay converts
    it); the gradient passes straight through to the float32 master weight."""

    @staticmethod
    def forward(ctx, w, dtype):
        return w.to(dtype).to(w.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, None


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        q, kv = c.heads * c.head_dim, c.kv_heads * c.head_dim
        self.attn_norm = nn.Parameter(torch.empty(c.hidden))
        self.wqkv = nn.Parameter(torch.empty(q + 2 * kv, c.hidden))
        self.bqkv = nn.Parameter(torch.empty(q + 2 * kv)) if c.qkv_bias else None
        self.wo = nn.Parameter(torch.empty(c.hidden, q))
        self.mlp_norm = nn.Parameter(torch.empty(c.hidden))
        self.w_gate_up = nn.Parameter(torch.empty(2 * c.intermediate, c.hidden))
        self.w_down = nn.Parameter(torch.empty(c.hidden, c.intermediate))


class Policy(nn.Module):
    """forward(tokens [B, T]) -> hidden states [B, T, hidden] after the final norm;
    logits() and token_logprobs() apply the LM head. Sequences are right-padded: with a
    causal mask, padding after a sequence never affects its real positions."""

    def __init__(self, config, rope_inv_freq):
        super().__init__()
        c = self.config = config
        self.embed = nn.Parameter(torch.empty(c.vocab, c.hidden))
        self.layers = nn.ModuleList(Layer(c) for _ in range(c.layers))
        self.final_norm = nn.Parameter(torch.empty(c.hidden))
        self.lm_head = None if c.tie_embeddings else nn.Parameter(torch.empty(c.vocab, c.hidden))
        self.register_buffer("inv_freq", torch.as_tensor(rope_inv_freq, dtype=torch.float32), persistent=False)

    @classmethod
    def from_relay(cls, model, device="cpu"):
        """A policy with the relay model's current weights (relay already converted them
        to float32 and fused the projections)."""
        p = cls(model.config, model.rope_inv_freq())
        with torch.no_grad():
            for name, param in p.named_parameters():
                param.copy_(torch.from_numpy(model.tensor(name)).view_as(param))
        return p.to(device)

    def sync_to(self, model):
        """Copies every parameter into the relay model's host weights (then call
        Engine.reload_weights() before the next rollout)."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                model.tensor(name)[:] = param.detach().reshape(-1).to("cpu", torch.float32).numpy()

    def push_to(self, engine):
        """The fast sync: every parameter straight into the engine's backend. With the
        policy on a GPU and relay's CUDA backend, the copy stays on the GPU (fp32 to fp16
        on the device, a peer copy if they are on different GPUs); the relay model's host
        copy is not updated. On the CPU backend it writes the host weights."""
        on_gpu = self.embed.is_cuda
        if on_gpu:  # relay reads on its own CUDA stream: let the optimizer's writes land first
            torch.cuda.synchronize(self.embed.device)
        with torch.no_grad():
            for name, param in self.named_parameters():
                t = param.detach()
                if engine.backend == "cpu" and t.is_cuda:
                    t = t.cpu()
                engine.update_tensor(name, t.to(torch.float32).contiguous())
        engine.finish_update()
        if on_gpu and engine.backend == "cuda":  # ...and finish reading before the next update
            torch.cuda.synchronize(torch.device("cuda", engine.device))

    def _rope(self, x, positions):
        # x [B, H, T, D]: pairs (i, i + D/2) rotated by position * inv_freq[i] (rotate_half).
        angle = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]  # [T, D/2]
        cos, sin = angle.cos(), angle.sin()
        a, b = x.chunk(2, dim=-1)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    checkpoint = False  # recompute each layer in backward instead of storing its activations

    # The precision the inference engine stores the weight matrices in (embedding, LM head
    # and projections; relay keeps norms and biases in float32). relay's CUDA backend uses
    # fp16, so set torch.float16 there: the trainer then computes with exactly the weights
    # the rollouts were sampled with, and the optimizer updates the float32 master copy.
    # Without it the two drift apart: an update of about the learning rate is below half
    # an fp16 step for most weights, so relay rounds it away while the trainer keeps it.
    weight_dtype = None

    def _w(self, w):
        return w if self.weight_dtype is None else _Rounded.apply(w, self.weight_dtype)

    def _layer(self, L, x, pos):
        c = self.config
        B, T = x.shape[:2]
        H, KV, D = c.heads, c.kv_heads, c.head_dim
        h = rmsnorm(x, L.attn_norm, c.rms_eps)
        qkv = F.linear(h, self._w(L.wqkv), L.bqkv)
        q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
        q = self._rope(q.view(B, T, H, D).transpose(1, 2), pos)
        k = self._rope(k.view(B, T, KV, D).transpose(1, 2), pos)
        v = v.view(B, T, KV, D).transpose(1, 2)
        if KV != H:  # grouped-query attention: query head h reads kv head h // (H / KV)
            k = k.repeat_interleave(H // KV, dim=1)
            v = v.repeat_interleave(H // KV, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(D))
        x = x + F.linear(a.transpose(1, 2).reshape(B, T, H * D), self._w(L.wo))
        h = rmsnorm(x, L.mlp_norm, c.rms_eps)
        gate, up = F.linear(h, self._w(L.w_gate_up)).chunk(2, dim=-1)
        x = x + F.linear(F.silu(gate) * up, self._w(L.w_down))
        return x

    def forward(self, tokens):
        x = self._w(F.embedding(tokens, self.embed))  # rounding the rows used = rounding the table
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        for L in self.layers:
            if self.checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self._layer, L, x, pos, use_reentrant=False)
            else:
                x = self._layer(L, x, pos)
        return rmsnorm(x, self.final_norm, self.config.rms_eps)

    def head(self):
        return self._w(self.embed if self.lm_head is None else self.lm_head)

    def logits(self, tokens):
        return F.linear(self(tokens), self.head())

    def token_logprobs(self, prompts, responses, temperature=1.0):
        """log p(response token | everything before it) for each (prompt, response) pair,
        under softmax(logits / temperature): a list of 1-D tensors, with gradients.
        Only the positions that predict response tokens go through the LM head."""
        lengths = [len(p) + len(r) for p, r in zip(prompts, responses)]
        T = max(lengths)
        device = self.embed.device
        ids = torch.zeros(len(prompts), T, dtype=torch.long, device=device)
        for i, (p, r) in enumerate(zip(prompts, responses)):
            ids[i, : lengths[i]] = torch.tensor(list(p) + list(r), device=device)
        hidden = self(ids)
        rows, cols, targets = [], [], []
        for i, (p, r) in enumerate(zip(prompts, responses)):
            n = len(p)
            rows += [i] * len(r)
            cols += range(n - 1, n - 1 + len(r))  # position t predicts token t + 1
            targets += list(r)
        h = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
        lg = F.linear(h, self.head()) / temperature
        lp = lg.gather(1, torch.tensor(targets, device=device)[:, None])[:, 0] - torch.logsumexp(lg, dim=-1)
        return list(lp.split([len(r) for r in responses]))

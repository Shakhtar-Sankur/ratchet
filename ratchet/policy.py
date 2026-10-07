"""The trainer's copy of the policy: a Llama/Qwen2 decoder in PyTorch whose parameters
have exactly relay's names and fused layout (wqkv, w_gate_up, ...), so a weight sync is
a copy, tensor by tensor, with no reshaping or renaming. It is built from a relay.Model
(same weights, same rotary frequencies) and computes the same function as relay's
backends: in float32 its log-probabilities match relay's CPU rollouts to rounding."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


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
        with torch.no_grad():
            for name, param in self.named_parameters():
                t = param.detach()
                if engine.backend == "cpu" and t.is_cuda:
                    t = t.cpu()
                engine.update_tensor(name, t.to(torch.float32).contiguous())
        engine.finish_update()

    def _rope(self, x, positions):
        # x [B, H, T, D]: pairs (i, i + D/2) rotated by position * inv_freq[i] (rotate_half).
        angle = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]  # [T, D/2]
        cos, sin = angle.cos(), angle.sin()
        a, b = x.chunk(2, dim=-1)
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def forward(self, tokens):
        c = self.config
        B, T = tokens.shape
        H, KV, D = c.heads, c.kv_heads, c.head_dim
        x = F.embedding(tokens, self.embed)
        pos = torch.arange(T, device=tokens.device)
        for L in self.layers:
            h = rmsnorm(x, L.attn_norm, c.rms_eps)
            qkv = F.linear(h, L.wqkv, L.bqkv)
            q, k, v = qkv.split([H * D, KV * D, KV * D], dim=-1)
            q = self._rope(q.view(B, T, H, D).transpose(1, 2), pos)
            k = self._rope(k.view(B, T, KV, D).transpose(1, 2), pos)
            v = v.view(B, T, KV, D).transpose(1, 2)
            if KV != H:  # grouped-query attention: query head h reads kv head h // (H / KV)
                k = k.repeat_interleave(H // KV, dim=1)
                v = v.repeat_interleave(H // KV, dim=1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(D))
            x = x + F.linear(a.transpose(1, 2).reshape(B, T, H * D), L.wo)
            h = rmsnorm(x, L.mlp_norm, c.rms_eps)
            gate, up = F.linear(h, L.w_gate_up).chunk(2, dim=-1)
            x = x + F.linear(F.silu(gate) * up, L.w_down)
        return rmsnorm(x, self.final_norm, c.rms_eps)

    def head(self):
        return self.embed if self.lm_head is None else self.lm_head

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

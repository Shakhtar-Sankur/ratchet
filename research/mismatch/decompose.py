"""Where does the training-inference mismatch come from? A source-by-source decomposition.

In RL post-training the inference engine samples a response and reports log pi_engine(token);
the trainer recomputes log pi_trainer(token) for the same tokens, and the importance ratio
pi_trainer / pi_engine is assumed to be 1 at equal weights. It is not. ratchet controls both
sides (relay's CUDA engine stores its weight matrices in fp16; the trainer is plain PyTorch),
so each source can be switched on alone and measured on the same sampled tokens:

  weights   the trainer's float32 master weights vs the same weights rounded as the engine
            stores them (fp16), or as a bf16 engine would (bf16);
  compute   fp32 arithmetic vs fp16 autocast, with identical (fp16-rounded) weights;
  engine    what remains between the engine and the closest trainer variant: the engine's
            own kernels (relay's flash-attention / flash-decoding, fused projections).

Measured at the released weights (step 0: bf16 checkpoints are exact in fp16, so weight
rounding contributes nothing) and after k real GRPO updates, when the float32 master
weights have moved off the low-precision grid. Each GRPO step also records how many weights
an update actually changes once rounded to fp16 / bf16 ("update visibility"), and how many
master weights are off that grid.

Prints JSON lines. Usage (Kaggle 2x T4, relay built as in scripts/kaggle_m5.sh):
  python research/mismatch/decompose.py --model /tmp/qwen --data /tmp/gsm8k --checkpoints 0,1,3,10
"""

import argparse
import contextlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from ratchet import gsm8k, relay  # noqa: E402
from ratchet.grpo import GRPO  # noqa: E402
from ratchet.policy import Policy  # noqa: E402

PROB_BINS = [(0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]


def stats(a, b, engine_lp=None):
    """Differences a - b (1-D tensors of per-token log-probabilities)."""
    g = a - b
    x = g.abs()
    out = {"tokens": g.numel(), "mean_abs": x.mean().item(), "p99_abs": x.quantile(0.99).item(),
           "max_abs": x.max().item(), "mean_signed": g.mean().item(),
           "ratio_outside_0.2": ((g.exp() - 1).abs() > 0.2).float().mean().item()}
    if engine_lp is not None:  # where the gap sits: by the engine's probability of the token
        p = engine_lp.exp()
        out["mean_abs_by_prob"] = {f"{lo:.1f}-{min(hi, 1):.1f}": x[(p >= lo) & (p < hi)].mean().item()
                                   if ((p >= lo) & (p < hi)).any() else None for lo, hi in PROB_BINS}
    return out


def trainer_logprobs(policy, prompts, responses, micro_batch, weight_dtype, autocast):
    kept = policy.weight_dtype
    policy.weight_dtype = weight_dtype
    ctx = (torch.autocast(policy.embed.device.type, dtype=autocast) if autocast is not None
           else contextlib.nullcontext())
    out = []
    with torch.no_grad(), ctx:
        for k in range(0, len(prompts), micro_batch):
            for lp in policy.token_logprobs(prompts[k:k + micro_batch], responses[k:k + micro_batch]):
                out.append(lp.float().cpu())
    policy.weight_dtype = kept
    return torch.cat(out)


def variants(on_gpu):
    v = {"fp32_weights": (None, None),
         "fp16_weights": (torch.float16, None),
         "bf16_weights": (torch.bfloat16, None)}
    if on_gpu:
        v["fp16_weights_fp16_compute"] = (torch.float16, torch.float16)
    return v


def decompose(policy, engine, tok, problems, args, seed, on_gpu):
    """Fresh rollouts (problems x group), then every trainer variant on the same tokens."""
    pids = [tok.prompt_ids(q) for q, _ in problems]
    G = args.group
    for i, p in enumerate(pids):
        for j in range(G):
            engine.add(i * G + j, p, max_new_tokens=args.max_new, temperature=1.0, seed=seed + i * G + j)
    o = engine.run()
    n = len(pids) * G
    resp = [o[k].tokens for k in range(n)]
    eng = torch.cat([torch.tensor(o[k].logprobs, dtype=torch.float32) for k in range(n)])
    prompts = [pids[k // G] for k in range(n)]
    lp = {name: trainer_logprobs(policy, prompts, resp, args.micro_batch, wd, ac)
          for name, (wd, ac) in variants(on_gpu).items()}
    rec = {"engine_vs_trainer": {name: stats(lp[name], eng, eng) for name in lp},
           "sources": {"weights_rounded_fp16": stats(lp["fp16_weights"], lp["fp32_weights"]),
                       "weights_rounded_bf16": stats(lp["bf16_weights"], lp["fp32_weights"])}}
    if "fp16_weights_fp16_compute" in lp:
        rec["sources"]["compute_fp16_vs_fp32"] = stats(lp["fp16_weights_fp16_compute"], lp["fp16_weights"])
    best = min(rec["engine_vs_trainer"], key=lambda k: rec["engine_vs_trainer"][k]["mean_abs"])
    rec["closest_trainer_variant"] = best
    rec["sources"]["engine_kernels_residual"] = rec["engine_vs_trainer"][best]
    return rec


class Visibility:
    """Per optimizer step: of the weight matrices' entries, how many change when rounded to
    fp16 / bf16, and how many master weights are off that grid."""

    def __init__(self, policy):
        self.params = [p for n, p in policy.named_parameters() if p.dim() == 2]
        self.prev = {d: [p.detach().to(d) for p in self.params] for d in (torch.float16, torch.bfloat16)}

    def measure(self):
        total = sum(p.numel() for p in self.params)
        rec = {}
        for d, prev in self.prev.items():
            changed = off = 0
            now = []
            for p, q in zip(self.params, prev):
                r = p.detach().to(d)
                changed += (r != q).sum().item()
                off += (r.float() != p.detach()).sum().item()
                now.append(r)
            self.prev[d] = now
            name = "fp16" if d == torch.float16 else "bf16"
            rec[f"changed_by_step_{name}"] = changed / total
            rec[f"off_grid_{name}"] = off / total
        return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--train-device", default="cuda:1")
    ap.add_argument("--relay-device", type=int, default=0)
    ap.add_argument("--checkpoints", default="0,1,3,10", help="measure after these many GRPO steps")
    ap.add_argument("--problems", type=int, default=8, help="problems per measurement (x group)")
    ap.add_argument("--prompts", type=int, default=8, help="problems per GRPO step")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--old-logprobs", default="trainer", choices=["trainer", "rollout"])
    ap.add_argument("--sync", default="push", choices=["push", "reload"])
    ap.add_argument("--train-weights", default="fp16", choices=["fp16", "fp32"],
                    help="the precision the trainer trains with (fp16: ratchet's fix; fp32: the bug)")
    ap.add_argument("--kv-blocks", type=int, default=3000)
    ap.add_argument("--max-batch-tokens", type=int, default=1024)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.weight_dtype = args.train_weights

    checkpoints = sorted(int(c) for c in args.checkpoints.split(","))
    tok = gsm8k.ChatTokenizer(args.model)
    model = relay.Model(args.model)
    engine = gsm8k.engine_for(model, args, args.relay_device)
    policy = gsm8k.policy_for(model, args, args.train_device)
    on_gpu = policy.embed.is_cuda
    policy.checkpoint = on_gpu
    train = gsm8k.load_problems(args.data, "train", 0)
    held = train[:args.problems]                       # the same problems at every checkpoint
    pool = train[args.problems:]
    g = GRPO(model, engine, policy, gsm8k.make_reward(tok), gsm8k.grpo_config(args, seed=args.seed))
    vis = Visibility(policy)
    header = {"experiment": "mismatch_decomposition", "model": os.path.basename(os.path.normpath(args.model)),
              "backend": args.backend, "lr": args.lr, "train_weights": args.train_weights,
              "checkpoints": checkpoints, "problems": args.problems, "group": args.group}
    gsm8k.log(header, args.out)
    step = 0
    for c in checkpoints:
        while step < c:
            batch = pool[step * args.prompts:(step + 1) * args.prompts]
            t = time.perf_counter()
            m = g.step([tok.prompt_ids(q) for q, _ in batch], [a for _, a in batch])
            step += 1
            rec = {"step": step, "reward": m["reward"], "seconds": time.perf_counter() - t}
            rec.update(vis.measure())
            gsm8k.log(rec, args.out)
        rec = {"checkpoint": c}
        rec.update(decompose(policy, engine, tok, held, args, seed=1000 + 17 * c, on_gpu=on_gpu))
        gsm8k.log(rec, args.out)


if __name__ == "__main__":
    main()

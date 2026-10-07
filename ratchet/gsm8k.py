"""GRPO on GSM8K: the experiment.

  python -m ratchet.gsm8k check      --model M --data D        # one GPU: the checks below
  python -m ratchet.gsm8k eval       --model M --data D        # accuracy, greedy, on relay
  python -m ratchet.gsm8k colocated  --model M --data D        # tandem DDP: every GPU generates, then trains
  python -m ratchet.gsm8k split      --model M --data D        # GPU 1 generates while GPU 0 trains

check measures, on the real model:
  - the gap between relay's fp16 log-probabilities and the fp32 trainer's for the same
    tokens (sampled rollouts), the gap the importance ratio absorbs;
  - weight sync time: host copy + rebuild (reload) against relay updating its weights
    in place from the trainer's GPU tensors (push), and that both give identical
    rollouts;
  - the answer-length distribution (the long tail) and rollout throughput.

colocated and split evaluate on the GSM8K test set before and after training and
print one JSON line per step. D is a directory with gsm8k_train.jsonl and
gsm8k_test.jsonl ({"question", "answer"} per line, as in the dataset); M a Hugging Face
model directory (Qwen2.5-0.5B-Instruct)."""

import argparse
import json
import os
import random
import sys
import time

import torch

from . import relay, tasks
from .grpo import GRPO, GRPOConfig
from .policy import Policy

# ---- data, prompts, reward -------------------------------------------------------


def load_problems(data_dir, split, limit=0):
    out = []
    with open(os.path.join(data_dir, f"gsm8k_{split}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out.append((r["question"], tasks.reference_answer(r["answer"])))
    return out[:limit] if limit else out


class FakeTokenizer:
    """For CPU smoke tests with the tiny test models (RATCHET_FAKE_TOKENIZER=1): text to
    ids by character, ids back to digits. Nothing here is meaningful but the plumbing."""

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.vocab = json.load(f)["vocab_size"]

    def prompt_ids(self, question):
        return [3 + ord(c) % (self.vocab - 3) for c in question[:24]]

    def decode(self, ids):
        return "#### " + "".join(str(i % 10) for i in ids[-2:])


class ChatTokenizer:
    """The model's own tokenizer and chat template (transformers)."""

    def __new__(cls, model_dir):
        if os.environ.get("RATCHET_FAKE_TOKENIZER") == "1":
            return FakeTokenizer(model_dir)
        return super().__new__(cls)

    def __init__(self, model_dir):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)

    def prompt_ids(self, question):
        # Render the chat template to text, then tokenize: apply_chat_template(tokenize=True)
        # returns a list in some transformers versions and a BatchEncoding in others.
        msgs = [{"role": "user", "content": tasks.PROMPT.format(question=question)}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return [int(t) for t in self.tok(text, add_special_tokens=False)["input_ids"]]

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def make_reward(tok):
    def reward(sample, answer):
        return tasks.reward(tok.decode(sample.tokens), answer)

    return reward


def evaluate(engine, tok, problems, max_new, ids=None):
    """Greedy accuracy on relay. Returns counts, so ranks can add them up."""
    ids = ids or [tok.prompt_ids(q) for q, _ in problems]
    for i, p in enumerate(ids):
        engine.add(10_000_000 + i, p, max_new_tokens=max_new, temperature=0.0)
    t0 = time.perf_counter()
    out = engine.run()
    dt = time.perf_counter() - t0
    correct = formatted = truncated = tokens = 0
    for i, (_, ref) in enumerate(problems):
        c = out[10_000_000 + i]
        text = tok.decode(c.tokens)
        correct += tasks.reward(text, ref) > 0
        formatted += "####" in text
        truncated += c.finish == relay.FINISH_LENGTH
        tokens += len(c.tokens)
    return {"n": len(problems), "correct": correct, "formatted": formatted, "truncated": truncated,
            "tokens": tokens, "seconds": dt}


def summarize(ev):
    n = max(ev["n"], 1)
    return {"accuracy": ev["correct"] / n, "format_rate": ev["formatted"] / n, "truncated": ev["truncated"] / n,
            "mean_tokens": ev["tokens"] / n, "n": ev["n"], "tok_per_s": ev["tokens"] / max(ev["seconds"], 1e-9)}


# ---- building the pieces ------------------------------------------------------------


def engine_for(model, args, device):
    return relay.Engine(model, backend=args.backend, device=device, num_blocks=args.kv_blocks, block_size=16,
                        max_batch_tokens=args.max_batch_tokens, max_seqs=args.max_seqs)


def grpo_config(args, seed, **kw):
    return GRPOConfig(group_size=args.group, max_new_tokens=args.max_new, temperature=1.0, lr=args.lr,
                      clip=args.clip, micro_batch=args.micro_batch, old_logprobs=args.old_logprobs,
                      sync=args.sync, seed=seed, **kw)


def log(rec, path=None):
    line = json.dumps(rec)
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def step_record(mode, m, extra=None):
    r = {k: v for k, v in m.items() if k != "samples"}
    r["mode"] = mode
    r["tokens_generated"] = sum(len(s.tokens) for s in m["samples"])
    r.update(extra or {})
    return r


def batches(problems, tok, per_step, steps, seed, rank=0, world=1, want=None):
    """The same shuffled order on every rank; each rank takes its slice of each step."""
    order = list(range(len(problems)))
    random.Random(seed).shuffle(order)
    pos = 0
    for _ in range(steps):
        n = want() if want else per_step
        take = [order[(pos + i) % len(order)] for i in range(n * world)]
        pos += n * world
        mine = take[rank * n:(rank + 1) * n]
        yield [tok.prompt_ids(problems[i][0]) for i in mine], [problems[i][1] for i in mine]


# ---- check ----------------------------------------------------------------------------


def check(args):
    dev = args.train_device
    tok = ChatTokenizer(args.model)
    model = relay.Model(args.model)
    engine = engine_for(model, args, args.relay_device)
    policy = Policy.from_relay(model, device=dev)
    probs = load_problems(args.data, "train", 64)
    rec = {"phase": "check", "backend": args.backend, "model": os.path.basename(os.path.normpath(args.model))}

    # 1. Rollouts: 8 problems x 8 samples at temperature 1.
    ids = [tok.prompt_ids(q) for q, _ in probs[:8]]
    for i, p in enumerate(ids):
        for j in range(8):
            engine.add(i * 8 + j, p, max_new_tokens=args.max_new, temperature=1.0, seed=1 + i * 8 + j)
    t0 = time.perf_counter()
    out = engine.run()
    dt = time.perf_counter() - t0
    resp = [out[k].tokens for k in range(64)]
    lens = sorted(len(r) for r in resp)
    rec["rollout"] = {"samples": 64, "tokens": sum(lens), "seconds": dt, "tok_per_s": sum(lens) / dt,
                      "len_min": lens[0], "len_median": lens[32], "len_p90": lens[57], "len_max": lens[-1],
                      "max_over_median": lens[-1] / max(lens[32], 1)}

    # 2. The fp16 (relay) vs fp32 (trainer) log-probability gap on those tokens.
    gaps = []
    with torch.no_grad():
        for k in range(0, 64, args.micro_batch):
            prompts = [ids[i // 8] for i in range(k, min(k + args.micro_batch, 64))]
            lps = policy.token_logprobs(prompts, resp[k:k + args.micro_batch])
            for i, lp in enumerate(lps):
                gaps.append(lp.cpu() - torch.tensor(out[k + i].logprobs))
    g = torch.cat(gaps)
    a = g.abs()
    rec["logprob_gap"] = {"tokens": g.numel(), "max_abs": a.max().item(), "mean_abs": a.mean().item(),
                          "p99_abs": a.quantile(0.99).item(), "mean_signed": g.mean().item(),
                          "ratio_max_dev": (g.exp() - 1).abs().max().item(),
                          "ratio_outside_clip_0.2": ((g.exp() - 1).abs() > 0.2).float().mean().item()}

    # 3. Weight sync: perturb the policy (as an update would), then reload vs push.
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(0)
        for p in policy.parameters():
            p.add_(torch.randn(p.shape, generator=gen).to(p.device) * 1e-3 * p.detach().abs().mean())

    def greedy():
        for i, p in enumerate(ids):
            engine.add(100 + i, p, max_new_tokens=64, temperature=0.0)
        o = engine.run()
        return [(o[100 + i].tokens, o[100 + i].logprobs) for i in range(len(ids))]

    def timed(fn):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t

    t_reload = timed(lambda: (policy.sync_to(model), engine.reload_weights()))
    a_out = greedy()
    t_push = sorted(timed(lambda: policy.push_to(engine)) for _ in range(3))
    b_out = greedy()
    nbytes = sum(p.numel() * 4 for p in policy.parameters())
    rec["sync"] = {"params": sum(p.numel() for p in policy.parameters()), "fp32_bytes": nbytes,
                   "reload_seconds": t_reload, "push_seconds_median": t_push[1], "push_seconds_all": t_push,
                   "speedup": t_reload / t_push[1], "push_GBps": nbytes / t_push[1] / 1e9,
                   "push_equals_reload": a_out == b_out}
    log(rec, args.out)
    return rec


# ---- eval -------------------------------------------------------------------------------


def eval_only(args):
    tok = ChatTokenizer(args.model)
    model = relay.Model(args.model)
    engine = engine_for(model, args, args.relay_device)
    probs = load_problems(args.data, "test", args.eval_limit)
    rec = {"phase": "eval", **summarize(evaluate(engine, tok, probs, args.eval_max_new))}
    log(rec, args.out)
    return rec


# ---- colocated: tandem DDP, every rank generates then trains ------------------------------


def _colocated_rank(group, args):
    from . import dist  # noqa: F401  (puts tandem on the path)

    rank, world = group.rank, group.size
    dev = f"cuda:{rank}" if args.backend == "cuda" else "cpu"
    if args.backend == "cuda":
        torch.cuda.set_device(rank)
    torch.manual_seed(args.seed)
    tok = ChatTokenizer(args.model)
    model = relay.Model(args.model)
    engine = engine_for(model, args, rank)
    policy = Policy.from_relay(model, device=dev)
    policy.checkpoint = True
    ddp = dist.wrap(policy, group)
    g = GRPO(model, engine, policy, make_reward(tok), grpo_config(args, seed=args.seed * 1000 + rank), ddp=ddp)
    test = load_problems(args.data, "test", args.eval_limit)
    mine = test[rank::world]

    def eval_all(tag):
        ev = evaluate(engine, tok, mine, args.eval_max_new)
        counts = torch.tensor([ev[k] for k in ("n", "correct", "formatted", "truncated", "tokens")],
                              dtype=torch.float64, device=dev)
        group.all_reduce(counts)
        tot = dict(zip(("n", "correct", "formatted", "truncated", "tokens"), counts.tolist()))
        tot["seconds"] = ev["seconds"]
        if rank == 0:
            log({"phase": "eval", "when": tag, "mode": "colocated", **summarize(tot)}, args.out)

    if not args.skip_eval:
        eval_all("before")
    train = load_problems(args.data, "train")
    t0 = time.perf_counter()
    per_rank = args.prompts // world
    for prompts, answers in batches(train, tok, per_rank, args.steps, args.seed, rank, world):
        m = g.step(prompts, answers)
        # Every rank's reward and generated tokens, added up (rank 0 logs; the ranks run in
        # lockstep, since DDP's reduction waits for all of them every step).
        both = torch.tensor([m["reward"], sum(len(s.tokens) for s in m["samples"]), m["max_length"]],
                            dtype=torch.float64, device=dev)
        group.all_reduce(both)
        if rank == 0:
            r = step_record("colocated", m)
            r.update({"reward": both[0].item() / world, "tokens_generated_all": both[1].item(),
                      "max_length_sum_of_ranks": both[2].item(), "elapsed": time.perf_counter() - t0})
            log(r, args.out)
    if not args.skip_eval:
        eval_all("after")
    return None


def colocated(args):
    from . import dist

    dist.launch(_colocated_rank, args.world, args, device="cuda" if args.backend == "cuda" else "cpu")


# ---- split: GPU 1 generates while GPU 0 trains --------------------------------------------


def split(args):
    dev = args.train_device
    tok = ChatTokenizer(args.model)
    model = relay.Model(args.model)
    engine = engine_for(model, args, args.relay_device)
    policy = Policy.from_relay(model, device=dev)
    policy.checkpoint = True
    kw = {}
    if args.partial:
        kw = dict(partial=True, groups_per_step=args.prompts, groups_in_flight=int(args.prompts * args.in_flight),
                  max_staleness=args.max_staleness)
    g = GRPO(model, engine, policy, make_reward(tok), grpo_config(args, seed=args.seed, **kw))
    test = load_problems(args.data, "test", args.eval_limit)
    mode = "split" + ("+async" if args.ahead else "") + ("+partial" if args.partial else "")
    if not args.skip_eval:
        log({"phase": "eval", "when": "before", "mode": mode, **summarize(evaluate(engine, tok, test, args.eval_max_new))},
            args.out)
    train = load_problems(args.data, "train")
    want = g.wanted if args.partial else None
    gen = batches(train, tok, args.prompts, args.steps, args.seed, want=want)
    t0 = time.perf_counter()
    steps = g.run_async(gen) if args.ahead else (g.step(p, a) for p, a in gen)
    for m in steps:
        r = step_record(mode, m)
        r["elapsed"] = time.perf_counter() - t0
        log(r, args.out)
    if not args.skip_eval:
        log({"phase": "eval", "when": "after", "mode": mode, **summarize(evaluate(engine, tok, test, args.eval_max_new))},
            args.out)


# ---- command line ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["check", "eval", "colocated", "split"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None, help="append JSON lines here too")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--train-device", default="cuda:0")
    ap.add_argument("--relay-device", type=int, default=0)
    ap.add_argument("--world", type=int, default=2, help="colocated: ranks (GPUs)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=8, help="problems per step (all ranks together)")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--old-logprobs", default="trainer", choices=["trainer", "rollout"])
    ap.add_argument("--sync", default="push", choices=["push", "reload"])
    ap.add_argument("--ahead", action="store_true", help="split: generate the next batch while training")
    ap.add_argument("--partial", action="store_true", help="split: partial rollouts")
    ap.add_argument("--in-flight", type=float, default=1.5, help="partial: groups in flight / groups per step")
    ap.add_argument("--max-staleness", type=int, default=2)
    ap.add_argument("--eval-limit", type=int, default=0, help="test problems (0: all 1319)")
    ap.add_argument("--eval-max-new", type=int, default=512)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--kv-blocks", type=int, default=3000)
    ap.add_argument("--max-batch-tokens", type=int, default=1024)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"check": check, "eval": eval_only, "colocated": colocated, "split": split}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())

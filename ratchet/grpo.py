"""GRPO (group relative policy optimization) with rollouts on relay.

One step:
  1. rollout: every prompt is sampled group_size times on relay's engine, which also
     reports the log-probability of each sampled token;
  2. reward: reward_fn scores each response;
  3. advantage: within a group, (reward - mean) / std, so a response is pushed up or
     down relative to the other answers to the same prompt (no value network);
  4. update: the PPO clipped objective, averaged over all response tokens, for
     `epochs` passes over the batch in micro-batches;
  5. sync: the new weights go into relay (Policy.push_to, or sync_to + reload).

The "old" log-probabilities in the importance ratio are recomputed by the trainer
before the update (old_logprobs="trainer", as most RL frameworks do), or taken as
relay's own (old_logprobs="rollout"), which also corrects for any gap between the
inference engine and the trainer. Every step records that gap.

The long tail. Generation is memory-bound decoding, and a step waits for its longest
answer. Two remedies, which make some tokens come from an older policy than the one
being trained; for those tokens the ratio always uses relay's sampling-time
log-probabilities, the behaviour policy's, so the importance correction stays right:
  - partial rollouts (partial=True): more groups are kept generating
    (groups_in_flight) than a step trains on (groups_per_step); a step stops as soon as
    groups_per_step groups have finished, and the unfinished answers keep their tokens
    and resume in the next step (relay's add_resume) under the new weights, unless they
    have waited more than max_staleness steps. wanted() says how many new prompts keep
    groups_in_flight groups going;
  - one step ahead (run_async): the next batch is generated, with the current weights,
    while the current one trains."""

import contextlib
import random
import threading
import time
from dataclasses import dataclass, field

import torch

from . import relay


@dataclass
class GRPOConfig:
    group_size: int = 8
    max_new_tokens: int = 64
    temperature: float = 1.0
    lr: float = 1e-6
    betas: tuple = (0.9, 0.99)
    weight_decay: float = 0.0
    clip: float = 0.2
    epochs: int = 1                 # optimizer passes over each rollout batch
    micro_batch: int = 8            # sequences per forward/backward
    scale_by_std: bool = True       # GRPO; False gives Dr. GRPO's unscaled advantages
    max_grad_norm: float = 1.0
    old_logprobs: str = "trainer"   # or "rollout"
    skip_zero_advantage: bool = True  # groups whose answers all scored the same teach nothing
    sync: str = "push"              # "push": update relay's backend in place; "reload": host copy + rebuild
    partial: bool = False           # stop each rollout once enough groups finished; resume the rest
    groups_per_step: int = 0        # partial: finished groups a step trains on (0: as many as were submitted)
    groups_in_flight: int = 0       # partial: groups kept generating (see wanted())
    max_staleness: int = 4          # partial: drop a group started more than this many steps ago
    seed: int = 0


def group_advantages(rewards, group_size, scale_by_std=True, eps=1e-6):
    """rewards: [n_prompts * group_size], grouped consecutively."""
    r = torch.as_tensor(rewards, dtype=torch.float32).view(-1, group_size)
    a = r - r.mean(dim=1, keepdim=True)
    if scale_by_std:
        a = a / (r.std(dim=1, keepdim=True, unbiased=False) + eps)
    return a.view(-1)


def clipped_objective(new_lp, old_lp, adv, clip):
    """The PPO clipped surrogate, summed over tokens (the caller divides by the token
    count of the whole batch, so micro-batches add up to a token-level mean).
    new_lp, old_lp, adv: 1-D, one entry per response token."""
    ratio = torch.exp(new_lp - old_lp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    loss = -torch.minimum(unclipped, clipped).sum()
    with torch.no_grad():
        clipped_frac = ((unclipped > clipped) & (adv != 0)).float().sum()
    return loss, clipped_frac


@dataclass
class Sample:
    prompt_index: int        # index in the batch the group was submitted with
    prompt: list
    tokens: list
    logprobs: list           # relay's, at sampling time (the behaviour policy's)
    finish: int
    reward: float = 0.0
    advantage: float = 0.0
    group: int = 0           # global group id
    answer: object = None
    seed: int = 0
    born: int = 0            # the step its group was submitted in
    done: bool = False
    stale: bool = False      # some tokens were sampled with older weights than the trainer's
    meta: dict = field(default_factory=dict)


class GRPO:
    """ddp: optionally tandem's DDP wrapping `policy`, for data parallelism across ranks
    (each rank its own relay engine and its own share of the prompts). Gradients are then
    averaged once per optimizer step: micro-batches before the last accumulate locally,
    and a rank with nothing to train still joins the reduction with zero gradients."""

    def __init__(self, model, engine, policy, reward_fn, config=GRPOConfig(), ddp=None):
        self.model, self.engine, self.policy = model, engine, policy
        self.ddp = ddp
        self.reward_fn, self.cfg = reward_fn, config
        self.opt = torch.optim.AdamW(policy.parameters(), lr=config.lr, betas=config.betas,
                                     weight_decay=config.weight_decay)
        self.steps = 0
        self.rng = random.Random(config.seed)
        self.pending = []      # groups (lists of Samples) not yet finished
        self.next_group = 0
        self.next_rid = 1
        self.dropped = 0       # groups dropped for staleness

    # ---- rollout ---------------------------------------------------------------

    def wanted(self):
        """How many new prompts keep groups_in_flight groups generating (partial mode)."""
        return max(0, self.cfg.groups_in_flight - len(self.pending))

    def generate(self, prompts, answers=None):
        """Starts a group for every prompt and generates until they have finished, or, with
        partial=True, until groups_per_step groups (new or carried over) have finished.
        Returns the finished groups' samples, group by group."""
        cfg, G = self.cfg, self.cfg.group_size
        for i, p in enumerate(prompts):
            gid = self.next_group
            self.next_group += 1
            self.pending.append([
                Sample(i, list(p), [], [], relay.FINISH_NONE, group=gid,
                       answer=None if answers is None else answers[i],
                       seed=(cfg.seed * 1_000_003 + gid * G + j) & (2**63 - 1), born=self.steps)
                for j in range(G)])
        if cfg.partial:
            keep = [g for g in self.pending if self.steps - g[0].born <= cfg.max_staleness]
            self.dropped += len(self.pending) - len(keep)
            self.pending = keep

        by_rid = {}
        for g in self.pending:
            for s in g:
                if s.done:
                    continue
                rid, self.next_rid = self.next_rid, self.next_rid + 1
                by_rid[rid] = s
                if s.tokens:  # an unfinished answer from an earlier step: same seed, next index
                    s.stale = True
                    self.engine.add_resume(rid, s.prompt, s.tokens, max_new_tokens=cfg.max_new_tokens,
                                           temperature=cfg.temperature, seed=s.seed)
                else:
                    self.engine.add(rid, s.prompt, max_new_tokens=cfg.max_new_tokens,
                                    temperature=cfg.temperature, seed=s.seed)
        target = (cfg.groups_per_step or len(prompts)) if cfg.partial else None
        finished = []
        while self.engine.has_work():
            for e in self.engine.step():
                s = by_rid[e.id]
                s.tokens.append(e.token)
                s.logprobs.append(e.logprob)
                if e.finish != relay.FINISH_NONE:
                    s.done, s.finish = True, e.finish
            for g in self.pending:
                if all(s.done for s in g):
                    finished.append(g)
            if finished:
                self.pending = [g for g in self.pending if not all(s.done for s in g)]
            if target is not None and len(finished) >= target:
                break
        if target is not None and len(finished) > target:  # several finished in the last pass
            self.pending = finished[target:] + self.pending
            finished = finished[:target]
        if self.engine.has_work():
            self.engine.cancel_all()  # the rest resume next step, from the tokens they have
            for g in self.pending:     # ...under newer weights than their tokens so far
                for s in g:
                    if s.tokens and not s.done:
                        s.stale = True
        return [s for g in finished for s in g]

    # ---- learning --------------------------------------------------------------

    def learn(self, samples):
        """Rewards, advantages and one update from finished groups. Touches only the
        policy, not relay, so it can run while relay generates the next batch."""
        cfg = self.cfg
        for s in samples:
            s.reward = float(self.reward_fn(s, s.answer))
        if samples:
            adv = group_advantages([s.reward for s in samples], cfg.group_size, cfg.scale_by_std)
            for s, a in zip(samples, adv.tolist()):
                s.advantage = a
        train = [s for s in samples if s.tokens and (s.advantage != 0 or not cfg.skip_zero_advantage)]
        stats = {"loss": 0.0, "clipped": 0.0, "tokens": 0, "gap_max": 0.0, "gap_mean": 0.0, "gap_tokens": 0,
                 "gap_over": 0}
        if train:
            self._update(train, stats)
        elif self.ddp is not None:
            self._join_with_zero_gradients()
        stats["trained_samples"] = len(train)
        return stats

    def sync(self):
        if self.cfg.sync == "push":
            self.policy.push_to(self.engine)
        else:
            self.policy.sync_to(self.model)
            self.engine.reload_weights()
        self.steps += 1

    def step(self, prompts, answers=None):
        """One GRPO step: generate, learn, sync. Returns metrics."""
        t0 = time.perf_counter()
        samples = self.generate(prompts, answers)
        t1 = time.perf_counter()
        stats = self.learn(samples)
        t2 = time.perf_counter()
        self.sync()
        t3 = time.perf_counter()
        return self._metrics(samples, stats, t1 - t0, t2 - t1, t3 - t2, t3 - t0)

    def run_async(self, batches):
        """One step ahead: while batch k trains, batch k+1 is generated with the weights
        from before that update; then the new weights are synced. Every trained sample
        is therefore one update behind (stale) and uses relay's log-probabilities in the
        ratio. Yields metrics per step. batches: iterable of (prompts, answers)."""
        it = iter(batches)
        first = next(it, None)
        if first is None:
            return
        samples = self.generate(*first)
        for nxt in list(it) + [None]:
            t0 = time.perf_counter()
            box, th, t_gen = {}, None, 0.0
            if nxt is not None:
                def work(b=nxt):
                    g0 = time.perf_counter()
                    box["s"] = self.generate(*b)
                    box["t"] = time.perf_counter() - g0
                th = threading.Thread(target=work)
                th.start()
            t1 = time.perf_counter()
            stats = self.learn(samples)
            t_learn = time.perf_counter() - t1
            if th is not None:
                th.join()
                t_gen = box["t"]
            t2 = time.perf_counter()
            self.sync()
            t3 = time.perf_counter()
            yield self._metrics(samples, stats, t_gen, t_learn, t3 - t2, t3 - t0)
            if nxt is not None:
                samples = box["s"]
                for s in samples:
                    s.stale = True  # generated before the update that was just synced

    def _metrics(self, samples, stats, t_gen, t_learn, t_sync, t_step):
        lengths = [len(s.tokens) for s in samples] or [0]
        n_tok = max(stats["tokens"], 1)
        return {
            "step": self.steps,
            "reward": sum(s.reward for s in samples) / max(len(samples), 1),
            "groups": len(samples) // self.cfg.group_size,
            "trained_samples": stats["trained_samples"],
            "stale_samples": sum(s.stale for s in samples),
            "carried_groups": len(self.pending),
            "dropped_groups": self.dropped,
            "mean_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "loss": stats["loss"],
            "clip_frac": stats["clipped"] / n_tok,
            "logprob_gap_max": stats["gap_max"],  # |trainer - relay| at the same weights
            "logprob_gap_mean": stats["gap_mean"] / max(stats["gap_tokens"], 1),
            # tokens whose importance ratio relay/trainer is outside 1 +- 0.2 at equal weights
            "logprob_gap_frac_over_0.2": stats["gap_over"] / max(stats["gap_tokens"], 1),
            "time_generate": t_gen,
            "time_train": t_learn,
            "time_sync": t_sync,
            "time_step": t_step,
            "samples": samples,
        }

    def _update(self, train, stats):
        cfg, pol = self.cfg, self.policy
        total = sum(len(s.tokens) for s in train)
        stats["tokens"] = total
        dev = next(pol.parameters()).device
        # Old log-probabilities: the trainer's at the current weights, or relay's. Tokens
        # sampled with older weights always use relay's, the behaviour policy's; only
        # fresh samples measure the gap between relay and the trainer at equal weights.
        old = []
        with torch.no_grad():
            for k in range(0, len(train), cfg.micro_batch):
                mb = train[k:k + cfg.micro_batch]
                lps = pol.token_logprobs([s.prompt for s in mb], [s.tokens for s in mb], cfg.temperature)
                for s, lp in zip(mb, lps):
                    rel = torch.tensor(s.logprobs, device=dev)
                    if not s.stale:
                        gap = (lp - rel).abs()
                        stats["gap_max"] = max(stats["gap_max"], gap.max().item())
                        stats["gap_mean"] += gap.sum().item()
                        stats["gap_tokens"] += gap.numel()
                        stats["gap_over"] += ((lp - rel).exp() - 1).abs().gt(0.2).sum().item()
                    old.append(lp if cfg.old_logprobs == "trainer" and not s.stale else rel)
        order = list(range(len(train)))
        for _ in range(cfg.epochs):
            self.rng.shuffle(order)
            self.opt.zero_grad(set_to_none=True)
            chunks = [order[k:k + cfg.micro_batch] for k in range(0, len(order), cfg.micro_batch)]
            for n, idx in enumerate(chunks):
                last = n == len(chunks) - 1
                ctx = self.ddp.no_sync() if self.ddp is not None and not last else contextlib.nullcontext()
                with ctx:
                    mb = [train[i] for i in idx]
                    new = pol.token_logprobs([s.prompt for s in mb], [s.tokens for s in mb], cfg.temperature)
                    new_lp = torch.cat(new)
                    old_lp = torch.cat([old[i] for i in idx])
                    adv = torch.cat([torch.full((len(s.tokens),), s.advantage, device=dev) for s in mb])
                    loss, clipped = clipped_objective(new_lp, old_lp, adv, cfg.clip)
                    (loss / total).backward()
                stats["loss"] += loss.item() / total
                stats["clipped"] += clipped.item()
            self._optimizer_step()
        stats["loss"] /= cfg.epochs
        stats["clipped"] /= cfg.epochs

    def _optimizer_step(self):
        if self.cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

    def _join_with_zero_gradients(self):
        """This rank has nothing to train this step (every group scored the same), but the
        others do: join their gradient reduction with zeros, then take the same optimizer
        step on the averaged gradients, so every rank keeps identical weights. The zeros
        come from a real forward and backward (on a two-token input, loss times 0) so that
        gradients arrive in the same order as on the other ranks: DDP groups them into
        buckets by arrival order, and the ranks' buckets must match."""
        for _ in range(self.cfg.epochs):
            self.opt.zero_grad(set_to_none=True)
            (torch.cat(self.policy.token_logprobs([[1]], [[1]])).sum() * 0.0).backward()
            self._optimizer_step()

"""GRPO (group relative policy optimization) with rollouts on relay.

One step:
  1. rollout: every prompt is sampled group_size times on relay's engine, which also
     reports the log-probability of each sampled token;
  2. reward: reward_fn scores each response;
  3. advantage: within a group, (reward - mean) / std, so a response is pushed up or
     down relative to the other answers to the same prompt (no value network);
  4. update: the PPO clipped objective, averaged over all response tokens, for
     `epochs` passes over the batch in micro-batches;
  5. sync: the new weights are copied into relay (policy.sync_to + reload_weights).

The "old" log-probabilities in the importance ratio are recomputed by the trainer
before the update (old_logprobs="trainer", as most RL frameworks do), or taken as
relay's own (old_logprobs="rollout"), which also corrects for any gap between the
inference engine and the trainer. Either way every step records that gap."""

import random
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
    prompt_index: int
    prompt: list
    tokens: list
    logprobs: list           # relay's, at sampling time
    finish: int
    reward: float = 0.0
    advantage: float = 0.0
    meta: dict = field(default_factory=dict)


class GRPO:
    def __init__(self, model, engine, policy, reward_fn, config=GRPOConfig()):
        self.model, self.engine, self.policy = model, engine, policy
        self.reward_fn, self.cfg = reward_fn, config
        self.opt = torch.optim.AdamW(policy.parameters(), lr=config.lr, betas=config.betas,
                                     weight_decay=config.weight_decay)
        self.steps = 0
        self.rng = random.Random(config.seed)

    def rollout(self, prompts):
        """Samples every prompt group_size times; returns Samples grouped by prompt."""
        G, base = self.cfg.group_size, (self.steps + 1) << 32
        for i, p in enumerate(prompts):
            for j in range(G):
                rid = base + i * G + j
                self.engine.add(rid, p, max_new_tokens=self.cfg.max_new_tokens, temperature=self.cfg.temperature,
                                seed=(self.cfg.seed * 1_000_003 + rid) & (2**64 - 1))
        done = self.engine.run()
        out = []
        for i, p in enumerate(prompts):
            for j in range(G):
                c = done[base + i * G + j]
                out.append(Sample(i, list(p), c.tokens, c.logprobs, c.finish))
        return out

    def step(self, prompts, answers=None):
        """One GRPO step on a batch of prompts (token ids); answers[i] is passed to the
        reward function with prompt i. Returns metrics."""
        cfg, t0 = self.cfg, time.perf_counter()
        samples = self.rollout(prompts)
        t_roll = time.perf_counter() - t0

        for s in samples:
            s.reward = float(self.reward_fn(s, None if answers is None else answers[s.prompt_index]))
        adv = group_advantages([s.reward for s in samples], cfg.group_size, cfg.scale_by_std)
        for s, a in zip(samples, adv.tolist()):
            s.advantage = a
        train = [s for s in samples if s.tokens and (s.advantage != 0 or not cfg.skip_zero_advantage)]

        t1 = time.perf_counter()
        stats = {"loss": 0.0, "clipped": 0.0, "tokens": 0, "gap_max": 0.0, "gap_mean": 0.0}
        if train:
            self._update(train, stats)
        t_train = time.perf_counter() - t1

        t2 = time.perf_counter()
        if cfg.sync == "push":
            self.policy.push_to(self.engine)
        else:
            self.policy.sync_to(self.model)
            self.engine.reload_weights()
        t_sync = time.perf_counter() - t2
        self.steps += 1

        lengths = [len(s.tokens) for s in samples]
        n_tok = max(stats["tokens"], 1)
        return {
            "step": self.steps,
            "reward": sum(s.reward for s in samples) / len(samples),
            "trained_samples": len(train),
            "mean_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "loss": stats["loss"],
            "clip_frac": stats["clipped"] / n_tok,
            "logprob_gap_max": stats["gap_max"],     # |trainer - relay| at the same weights
            "logprob_gap_mean": stats["gap_mean"] / n_tok,
            "time_rollout": t_roll,
            "time_train": t_train,
            "time_sync": t_sync,
            "samples": samples,
        }

    def _update(self, train, stats):
        cfg, pol = self.cfg, self.policy
        total = sum(len(s.tokens) for s in train)
        stats["tokens"] = total
        dev = next(pol.parameters()).device
        # Old log-probabilities at the current (pre-update) weights, and their gap to relay's.
        old = []
        with torch.no_grad():
            for k in range(0, len(train), cfg.micro_batch):
                mb = train[k:k + cfg.micro_batch]
                lps = pol.token_logprobs([s.prompt for s in mb], [s.tokens for s in mb], cfg.temperature)
                for s, lp in zip(mb, lps):
                    rel = torch.tensor(s.logprobs, device=dev)
                    gap = (lp - rel).abs()
                    stats["gap_max"] = max(stats["gap_max"], gap.max().item())
                    stats["gap_mean"] += gap.sum().item()
                    old.append(lp if cfg.old_logprobs == "trainer" else rel)
        order = list(range(len(train)))
        for _ in range(cfg.epochs):
            self.rng.shuffle(order)
            self.opt.zero_grad(set_to_none=True)
            for k in range(0, len(order), cfg.micro_batch):
                idx = order[k:k + cfg.micro_batch]
                mb = [train[i] for i in idx]
                new = pol.token_logprobs([s.prompt for s in mb], [s.tokens for s in mb], cfg.temperature)
                new_lp = torch.cat(new)
                old_lp = torch.cat([old[i] for i in idx])
                adv = torch.cat([torch.full((len(s.tokens),), s.advantage, device=dev) for s in mb])
                loss, clipped = clipped_objective(new_lp, old_lp, adv, cfg.clip)
                (loss / total).backward()
                stats["loss"] += loss.item() / total
                stats["clipped"] += clipped.item()
            if cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(pol.parameters(), cfg.max_grad_norm)
            self.opt.step()
        stats["loss"] /= cfg.epochs
        stats["clipped"] /= cfg.epochs

"""The long tail: partial rollouts and one-step-ahead training.

The test model is chat-tiny with its end-of-sequence row boosted, so answer lengths are
long-tailed (64 samples: median 5 tokens, 90th percentile 23, longest 48)."""

import os

import torch

from ratchet import relay
from ratchet.grpo import GRPO, GRPOConfig
from ratchet.policy import Policy

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "relay", "tests", "fixtures")


def subset_reward(sample, _answer):
    return sum(t % 4 == 0 for t in sample.tokens) / max(len(sample.tokens), 1)


def make(lr=3e-2, **kw):
    torch.manual_seed(0)
    model = relay.Model(os.path.join(FIXTURES, "chat-tiny"))
    H, eos = model.config.hidden, model.config.eos_ids[0]
    head = model.tensor("lm_head") if model.tensor("lm_head") is not None else model.tensor("embed")
    head[eos * H:(eos + 1) * H] *= 40.0
    engine = relay.Engine(model, max_seqs=64)
    policy = Policy.from_relay(model)
    cfg = GRPOConfig(group_size=8, max_new_tokens=48, lr=lr, micro_batch=16, **kw)
    return GRPO(model, engine, policy, subset_reward, cfg)


def prompts(step, n=4):
    return [[1 + (11 * step + 5 * i + t) % 400 for t in range(3 + i)] for i in range(n)]


PARTIAL = dict(partial=True, groups_per_step=4, groups_in_flight=6)


def partial_step(g, s):
    """Top up to groups_in_flight with new prompts, then a step that trains on 4 groups."""
    return g.step(prompts(s, g.wanted()))


def by_group(samples):
    out = {}
    for s in samples:
        out.setdefault(s.group, []).append((s.tokens, s.logprobs))
    return out


def test_partial_rollouts_with_frozen_weights_reproduce_full_ones():
    # With lr = 0 the weights never change, so a group finished across several steps
    # must equal the same group generated in one go: resume is exact.
    part = make(lr=0.0, max_staleness=100, **PARTIAL)
    b, submitted = {}, []
    for s in range(6):
        submitted += prompts(s, part.wanted())
        b.update(by_group(part.step(prompts(s, part.wanted()))["samples"]))
    while part.pending:  # let the carried groups finish
        b.update(by_group(part.step([])["samples"]))
    full = make(lr=0.0)
    a = by_group(full.step(submitted)["samples"])  # the same groups (same ids and seeds) in one go
    assert a.keys() == b.keys()
    for g in a:
        assert a[g] == b[g], g


def test_partial_rollouts_end_steps_early_and_carry_the_rest():
    part, full = make(lr=0.0, max_staleness=100, **PARTIAL), make(lr=0.0)
    for s in range(6):
        m = partial_step(part, s)
        assert m["groups"] == 4                      # a step ends once 4 groups have finished...
        assert m["carried_groups"] > 0               # ...and the others keep going
        full.step(prompts(s, 4))
    # The same number of groups trained (24); far fewer forward passes on the engine.
    assert part.engine.stats()["steps"] < full.engine.stats()["steps"]
    assert any(s.stale for g in part.pending for s in g if s.tokens)


def test_groups_older_than_max_staleness_are_dropped():
    part = make(lr=0.0, max_staleness=0, **PARTIAL)
    for s in range(6):
        m = partial_step(part, s)
        assert all(x.born == part.steps - 1 for x in m["samples"])  # only groups from this step
    assert m["dropped_groups"] > 0


def test_partial_rollouts_still_learn():
    g = make(max_staleness=2, **PARTIAL)
    rewards = [partial_step(g, s)["reward"] for s in range(40)]
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.3, rewards


def test_one_step_ahead_with_frozen_weights_equals_the_synchronous_run():
    sync, ahead = make(lr=0.0), make(lr=0.0)
    batches = [(prompts(s), None) for s in range(5)]
    a = [by_group(sync.step(*b)["samples"]) for b in batches]
    b = [by_group(m["samples"]) for m in ahead.run_async(batches)]
    assert a == b


def test_one_step_ahead_learns_and_marks_its_samples_stale():
    g = make()
    ms = list(g.run_async([(prompts(s), None) for s in range(40)]))
    assert all(x.stale for m in ms[1:] for x in m["samples"])
    rewards = [m["reward"] for m in ms]
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.3, rewards

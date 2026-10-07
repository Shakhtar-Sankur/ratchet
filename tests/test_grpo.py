"""GRPO: the advantage and loss maths, a toy task that a tiny model learns on the CPU
(rollouts on relay, updates in PyTorch, weights synced back every step), and a whole
run being reproducible bit for bit."""

import os

import pytest
import torch

from ratchet import relay
from ratchet.grpo import GRPO, GRPOConfig, clipped_objective, group_advantages
from ratchet.policy import Policy

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "relay", "tests", "fixtures")


def test_group_advantages():
    a = group_advantages([1, 0, 0, 1, 2, 2, 2, 2], group_size=4)
    assert torch.allclose(a[:4], torch.tensor([1.0, -1, -1, 1]), atol=1e-5)
    assert torch.equal(a[4:], torch.zeros(4))  # all answers equal: nothing to learn
    b = group_advantages([1, 0, 0, 1], group_size=4, scale_by_std=False)
    assert torch.allclose(b, torch.tensor([0.5, -0.5, -0.5, 0.5]))


def test_unclipped_gradient_is_the_policy_gradient():
    # At ratio 1 the gradient of the surrogate is -A * grad log p for every token.
    lp = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
    adv = torch.tensor([2.0, -1.0, 0.5])
    loss, clipped = clipped_objective(lp, lp.detach(), adv, clip=0.2)
    loss.backward()
    assert torch.allclose(lp.grad, -adv) and clipped == 0


@pytest.mark.parametrize("delta,adv,flows", [
    (0.5, 1.0, False),    # already much more likely, positive advantage: clipped
    (-0.5, -1.0, False),  # already much less likely, negative advantage: clipped
    (0.5, -1.0, True),    # more likely but bad: the gradient still pushes it down
    (-0.5, 1.0, True),    # less likely but good: still pushed up
])
def test_clipping(delta, adv, flows):
    old = torch.tensor([-1.0])
    new = (old + delta).clone().requires_grad_(True)
    loss, _ = clipped_objective(new, old, torch.tensor([adv]), clip=0.2)
    loss.backward()
    assert (new.grad.abs().item() > 0) == flows


def subset_reward(sample, _answer):
    """Toy task: the fraction of generated tokens whose id is a multiple of 4 (a quarter
    of the vocabulary, so about 0.25 before training)."""
    return sum(t % 4 == 0 for t in sample.tokens) / max(len(sample.tokens), 1)


def make(old_logprobs="trainer", seed=0, lr=3e-2):
    torch.manual_seed(seed)
    model = relay.Model(os.path.join(FIXTURES, "chat-tiny"))
    engine = relay.Engine(model, max_seqs=64)
    policy = Policy.from_relay(model)
    cfg = GRPOConfig(group_size=8, max_new_tokens=8, lr=lr, micro_batch=16, old_logprobs=old_logprobs, seed=seed)
    return GRPO(model, engine, policy, subset_reward, cfg)


def prompts(step, n=4):
    return [[1 + (11 * step + 5 * i + t) % 400 for t in range(3 + i)] for i in range(n)]


@pytest.mark.parametrize("old_logprobs", ["trainer", "rollout"])
def test_a_tiny_model_learns_the_toy_task(old_logprobs):
    g = make(old_logprobs)
    rewards = [g.step(prompts(s))["reward"] for s in range(30)]
    first, last = sum(rewards[:5]) / 5, sum(rewards[-5:]) / 5
    assert first < 0.4, rewards
    assert last > first + 0.3, rewards


def test_relay_and_the_trainer_agree_throughout_a_run():
    g = make()
    for s in range(5):
        m = g.step(prompts(s))
        assert m["trained_samples"] > 0
        assert m["logprob_gap_max"] < 1e-5, m["logprob_gap_max"]


def test_a_run_is_reproducible_bit_for_bit():
    a, b = make(), make()
    for s in range(4):
        ma, mb = a.step(prompts(s)), b.step(prompts(s))
        assert ma["reward"] == mb["reward"] and ma["loss"] == mb["loss"]
        assert [x.tokens for x in ma["samples"]] == [x.tokens for x in mb["samples"]]
    for (n, p), q in zip(a.policy.named_parameters(), b.policy.parameters()):
        assert torch.equal(p, q), n


def test_the_fast_sync_gives_the_same_run_as_a_reload():
    a, b = make(), make()
    a.cfg.sync, b.cfg.sync = "push", "reload"
    for s in range(4):
        ma, mb = a.step(prompts(s)), b.step(prompts(s))
        assert [x.tokens for x in ma["samples"]] == [x.tokens for x in mb["samples"]]
        assert [x.logprobs for x in ma["samples"]] == [x.logprobs for x in mb["samples"]]

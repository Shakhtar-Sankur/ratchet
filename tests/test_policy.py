"""The trainer's policy computes the same function as relay: its logits match Hugging
Face transformers (relay's reference outputs), and the log-probability it assigns to
each rollout token matches the one relay reported while sampling it. After the policy
is changed and synced into relay, the two still agree: the sync is complete."""

import json
import os

import pytest
import torch

from ratchet import relay
from ratchet.policy import Policy

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "relay", "tests", "fixtures")
MODELS = ["chat-tiny", "qwen2-bias", "llama3-rope", "llama-gqa-tied-bf16", "llama-mha"]

torch.manual_seed(0)


def load(name):
    m = relay.Model(os.path.join(FIXTURES, name))
    return m, Policy.from_relay(m)


def rollouts(model, n=6, max_new=12, temperature=1.0, seed=0):
    v = model.config.vocab
    prompts = [[1 + (7 * t + 3 * i) % (v - 1) for t in range(2 + 4 * i)] for i in range(n)]
    e = relay.Engine(model, max_seqs=8)
    for i, p in enumerate(prompts):
        e.add(i, p, max_new_tokens=max_new, temperature=temperature, seed=seed + i, ignore_eos=True)
    out = e.run()
    return prompts, [out[i].tokens for i in range(n)], [out[i].logprobs for i in range(n)]


def worst_gap(policy, prompts, responses, logprobs, temperature=1.0):
    with torch.no_grad():
        mine = policy.token_logprobs(prompts, responses, temperature)
    return max((m - torch.tensor(lp)).abs().max().item() for m, lp in zip(mine, logprobs))


@pytest.mark.parametrize("name", MODELS)
def test_logits_match_transformers(name):
    _, policy = load(name)
    ref = json.load(open(os.path.join(FIXTURES, name, "relay-reference.json")))
    with torch.no_grad():
        got = policy.logits(torch.tensor([ref["prompt"]]))[0]
    want = torch.tensor(ref["prompt_logits"])
    rel = ((got - want).norm(dim=-1) / want.norm(dim=-1)).max().item()
    assert rel < 1e-6, rel  # measured: 0 on four models, 2e-7 on llama3-rope


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_rollout_logprobs_match_the_trainer(name, temperature):
    model, policy = load(name)
    prompts, responses, logprobs = rollouts(model, temperature=temperature)
    gap = worst_gap(policy, prompts, responses, logprobs, temperature)
    assert gap < 1e-5, gap  # measured: at most 1e-6 (relay stores them as float)


@pytest.mark.parametrize("name", ["chat-tiny", "qwen2-bias"])
def test_after_an_update_and_a_sync_they_still_agree(name):
    model, policy = load(name)
    prompts, responses, _ = rollouts(model)
    # One optimizer step that raises the likelihood of the sampled responses.
    opt = torch.optim.SGD(policy.parameters(), lr=0.5)
    loss = -torch.cat(policy.token_logprobs(prompts, responses)).mean()
    loss.backward()
    opt.step()
    policy.sync_to(model)
    p2, r2, lp2 = rollouts(model, seed=100)  # a new engine, built from the synced weights
    assert worst_gap(policy, p2, r2, lp2) < 1e-5
    # And the update did something: the old responses became more likely.
    with torch.no_grad():
        after = -torch.cat(policy.token_logprobs(prompts, responses)).mean()
    assert after < loss.detach()


def test_batching_and_padding_do_not_change_logprobs():
    model, policy = load("qwen2-bias")
    prompts, responses, _ = rollouts(model)
    with torch.no_grad():
        together = policy.token_logprobs(prompts, responses)
        for i in range(len(prompts)):
            alone = policy.token_logprobs([prompts[i]], [responses[i]])[0]
            assert (alone - together[i]).abs().max().item() < 1e-5


def test_gradients_flow_to_every_parameter():
    _, policy = load("qwen2-bias")
    lp = policy.token_logprobs([[1, 2, 3]], [[4, 5, 6]])
    torch.cat(lp).sum().backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name

"""Run in each tandem rank by test_ddp.py (spawned processes import it by name)."""
import os

import torch

from ratchet import dist, relay
from ratchet.grpo import GRPO, GRPOConfig
from ratchet.policy import Policy

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "relay", "tests", "fixtures")


def subset_reward(sample, _answer):
    return sum(t % 4 == 0 for t in sample.tokens) / max(len(sample.tokens), 1)


def flat_reward(sample, _answer):
    return 0.5  # every answer equal: zero advantages, nothing to train


def worker(group, steps, rank1_flat):
    torch.manual_seed(group.rank)  # different on purpose: DDP must broadcast rank 0's weights
    model = relay.Model(os.path.join(FIXTURES, "chat-tiny"))
    engine = relay.Engine(model, max_seqs=64)
    policy = Policy.from_relay(model)
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ddp = dist.wrap(policy, group)
    reward = flat_reward if (rank1_flat and group.rank == 1) else subset_reward
    cfg = GRPOConfig(group_size=8, max_new_tokens=8, lr=3e-2, micro_batch=8, seed=1000 * group.rank)
    g = GRPO(model, engine, policy, reward, cfg, ddp=ddp)
    trained, rewards = [], []
    for s in range(steps):
        prompts = [[1 + (11 * s + 5 * i + 97 * group.rank + t) % 400 for t in range(3 + i)] for i in range(2)]
        m = g.step(prompts)
        trained.append(m["trained_samples"])
        rewards.append(m["reward"])
    weights = torch.cat([p.detach().reshape(-1) for p in policy.parameters()])
    return {"weights": weights, "trained": trained, "rewards": rewards}

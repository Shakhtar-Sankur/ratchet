"""Data-parallel GRPO on two tandem ranks (CPU): the ranks start from rank 0's weights,
average their gradients every step and stay identical, also when one rank has nothing
to train; and two ranks still learn the toy task."""

import os

import pytest
import torch

from ratchet import dist

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def spawned_ranks_find_the_worker(monkeypatch):
    paths = [HERE, os.path.dirname(HERE), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in paths if p))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def run(steps, rank1_flat):
    import ddp_worker

    return dist.launch(ddp_worker.worker, 2, steps, rank1_flat)


def test_ranks_stay_identical():
    a, b = run(4, False)
    assert torch.equal(a["weights"], b["weights"])
    assert all(t > 0 for t in a["trained"] + b["trained"])


def test_a_rank_with_nothing_to_train_still_joins_and_stays_identical():
    a, b = run(4, True)
    assert all(t == 0 for t in b["trained"]) and any(t > 0 for t in a["trained"])
    assert torch.equal(a["weights"], b["weights"])


def test_two_ranks_learn():
    a, b = run(30, False)
    r = [(x + y) / 2 for x, y in zip(a["rewards"], b["rewards"])]
    assert sum(r[-5:]) / 5 > sum(r[:5]) / 5 + 0.3, r

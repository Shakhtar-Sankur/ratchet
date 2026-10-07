"""The GSM8K driver end to end on the CPU with a tiny model and a stand-in tokenizer:
every phase runs and reports what the GPU run will report. Plumbing, not accuracy."""

import json
import os

import pytest

from ratchet import gsm8k

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "relay", "tests", "fixtures", "chat-tiny")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RATCHET_FAKE_TOKENIZER", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([os.path.dirname(HERE), os.environ.get("PYTHONPATH", "")]))
    for split, n in (("train", 40), ("test", 12)):
        with open(tmp_path / f"gsm8k_{split}.jsonl", "w") as f:
            for i in range(n):
                f.write(json.dumps({"question": f"What is {i} plus {i}?", "answer": f"... #### {2 * i}"}) + "\n")
    return tmp_path


def run(env, *args):
    out = env / "out.jsonl"
    gsm8k.main([*args, "--model", MODEL, "--data", str(env), "--out", str(out), "--backend", "cpu",
                "--train-device", "cpu", "--max-new", "12", "--eval-max-new", "8", "--group", "4",
                "--kv-blocks", "256", "--lr", "1e-3"])
    return [json.loads(line) for line in open(out)]


def test_check(env):
    (rec,) = run(env, "check")
    assert rec["rollout"]["samples"] == 64
    assert rec["logprob_gap"]["max_abs"] < 1e-5          # CPU: fp32 on both sides
    assert rec["sync"]["push_equals_reload"] is True


def test_eval(env):
    (rec,) = run(env, "eval")
    assert rec["n"] == 12 and 0 <= rec["accuracy"] <= 1


def test_split_one_step_ahead_with_partial_rollouts(env):
    recs = run(env, "split", "--steps", "4", "--prompts", "2", "--ahead", "--partial")
    steps = [r for r in recs if "step" in r]
    evals = [r for r in recs if r.get("phase") == "eval"]
    assert len(steps) == 4 and [e["when"] for e in evals] == ["before", "after"]
    assert all(s["mode"] == "split+async+partial" and s["groups"] == 2 for s in steps)


def test_colocated_on_two_tandem_ranks(env):
    recs = run(env, "colocated", "--steps", "3", "--prompts", "4", "--world", "2")
    steps = [r for r in recs if "step" in r]
    evals = [r for r in recs if r.get("phase") == "eval"]
    assert len(steps) == 3 and all(s["mode"] == "colocated" for s in steps)
    assert [e["when"] for e in evals] == ["before", "after"] and evals[0]["n"] == 12

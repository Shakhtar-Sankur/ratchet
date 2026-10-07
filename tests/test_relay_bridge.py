"""relay through ctypes: rollouts are deterministic and batch-invariant (on the CPU
backend), log-probabilities come with every token, unfinished rollouts resume
exactly, and weights written from Python reach the engine after a reload."""

import math
import os

import numpy as np
import pytest

from ratchet import relay

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "relay", "tests", "fixtures")
MODELS = ["chat-tiny", "qwen2-bias", "llama3-rope", "llama-gqa-tied-bf16"]


def prompts(vocab, n=6):
    return [[1 + (7 * t + 3 * i) % (vocab - 1) for t in range(2 + 5 * i)] for i in range(n)]


def rollout(engine, ps, max_new=10, seed=0, temperature=1.0):
    for i, p in enumerate(ps):
        engine.add(i, p, max_new_tokens=max_new, temperature=temperature, seed=seed + i, ignore_eos=True)
    return engine.run()


@pytest.fixture(params=MODELS)
def model(request):
    return relay.Model(os.path.join(FIXTURES, request.param))


def test_every_tensor_exists_with_the_configured_size(model):
    c = model.config
    q, kv = c.heads * c.head_dim, c.kv_heads * c.head_dim
    want = {"attn_norm": c.hidden, "wqkv": (q + 2 * kv) * c.hidden, "bqkv": q + 2 * kv, "wo": c.hidden * q,
            "mlp_norm": c.hidden, "w_gate_up": 2 * c.intermediate * c.hidden, "w_down": c.hidden * c.intermediate}
    for name in model.tensor_names():
        t = model.tensor(name)
        assert t is not None and t.dtype == np.float32, name
        if name.startswith("layers."):
            assert t.size == want[name.split(".")[2]], name
    assert model.tensor("embed").size == c.vocab * c.hidden
    assert (model.tensor("lm_head") is None) == c.tie_embeddings
    assert model.tensor("layers.0.bqkv") is not None if c.qkv_bias else model.tensor("layers.0.bqkv") is None
    assert model.rope_inv_freq().shape == (c.head_dim // 2,)


def test_rollouts_are_deterministic_and_carry_logprobs(model):
    ps = prompts(model.config.vocab)
    a = rollout(relay.Engine(model, max_seqs=8), ps)
    b = rollout(relay.Engine(model, max_seqs=8), ps)
    assert a.keys() == b.keys() == set(range(len(ps)))
    for i in a:
        assert a[i].tokens == b[i].tokens and a[i].logprobs == b[i].logprobs
        assert len(a[i].tokens) == 10 and a[i].finish == relay.FINISH_LENGTH
        assert all(math.isfinite(x) and x <= 0 for x in a[i].logprobs)


def test_a_rollout_is_the_same_alone_or_in_a_batch(model):
    # The CPU backend is batch-invariant: the trainer can rely on rollouts not depending
    # on what else was in the batch.
    ps = prompts(model.config.vocab)
    together = rollout(relay.Engine(model, max_seqs=8), ps)
    for i, p in enumerate(ps):
        e = relay.Engine(model, max_seqs=8)
        e.add(i, p, max_new_tokens=10, seed=i, ignore_eos=True)
        alone = e.run()[i]
        assert alone.tokens == together[i].tokens and alone.logprobs == together[i].logprobs


def test_greedy_tokens_are_the_most_likely(model):
    out = rollout(relay.Engine(model), prompts(model.config.vocab, 3), temperature=0.0)
    for c in out.values():
        # The greedy token's probability is at least 1/vocab.
        assert all(x >= -math.log(model.config.vocab) - 1e-4 for x in c.logprobs)


def test_partial_rollouts_resume_exactly(model):
    ps = prompts(model.config.vocab, 4)
    full = rollout(relay.Engine(model), ps, max_new=12)
    e = relay.Engine(model)
    for i, p in enumerate(ps):
        e.add_resume(100 + i, p, full[i].tokens[:5], max_new_tokens=12, seed=i, ignore_eos=True)
    rest = e.run()
    for i in range(len(ps)):
        assert rest[100 + i].tokens == full[i].tokens[5:]
        assert rest[100 + i].logprobs == full[i].logprobs[5:]


def test_weights_written_from_python_reach_the_engine_after_reload(model):
    ps = prompts(model.config.vocab, 3)
    e = relay.Engine(model)
    before = rollout(e, ps)
    w = model.tensor("final_norm")
    saved = w.copy()
    w *= -1.0
    e.reload_weights()
    changed = rollout(e, ps)
    assert any(changed[i].logprobs != before[i].logprobs for i in before)
    w[:] = saved
    e.reload_weights()
    again = rollout(e, ps)
    assert all(again[i].tokens == before[i].tokens and again[i].logprobs == before[i].logprobs for i in before)


def test_reload_is_refused_mid_rollout(model):
    e = relay.Engine(model)
    e.add(1, [1, 2, 3], max_new_tokens=4)
    with pytest.raises(relay.RelayError, match="in flight"):
        e.reload_weights()
    e.cancel_all()
    e.reload_weights()


def test_errors_surface_as_exceptions():
    with pytest.raises(relay.RelayError):
        relay.Model("/nonexistent/model")
    m = relay.Model(os.path.join(FIXTURES, "chat-tiny"))
    with pytest.raises(relay.RelayError, match="backend"):
        relay.Engine(m, backend="tpu")
    e = relay.Engine(m)
    with pytest.raises(relay.RelayError):
        e.add(1, [], max_new_tokens=3)

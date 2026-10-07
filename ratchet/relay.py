"""relay from Python: load a model, read and write its weights, and generate rollouts
with per-token log-probabilities.

A thin ctypes layer over relay's C interface (relay/engine/include/relay/c_api.h),
built as librelay_c.so. It is found through RATCHET_RELAY_LIB, or in the relay
checkout next to this package (relay/build, or relay/build-cuda when it exists).
scripts/build_relay.sh builds it."""

import ctypes
import os
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

FINISH_NONE, FINISH_LENGTH, FINISH_STOP = 0, 1, 2
MAX_EOS = 8


class RelayError(RuntimeError):
    pass


class _Config(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in
                ("hidden", "intermediate", "layers", "heads", "kv_heads", "head_dim", "vocab", "max_position")] + [
        ("rms_eps", ctypes.c_double), ("rope_theta", ctypes.c_double),
        ("tie_embeddings", ctypes.c_int), ("qkv_bias", ctypes.c_int),
        ("num_eos", ctypes.c_int), ("eos_ids", ctypes.c_int * MAX_EOS)]


class _Sampling(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_float), ("top_p", ctypes.c_float), ("top_k", ctypes.c_int),
                ("max_new_tokens", ctypes.c_int), ("seed", ctypes.c_uint64), ("ignore_eos", ctypes.c_int)]


class _Event(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("token", ctypes.c_int), ("index", ctypes.c_int),
                ("finish", ctypes.c_int), ("logprob", ctypes.c_float)]


class _Stats(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in
                ("steps", "forward_tokens", "preemptions", "prompt_tokens", "prefix_hit_tokens")]


def _find_lib():
    env = os.environ.get("RATCHET_RELAY_LIB")
    if env:
        return env
    for d in ("build-cuda", "build"):
        p = os.path.join(_REPO, "relay", d, "librelay_c.so")
        if os.path.exists(p):
            return p
    raise RelayError("librelay_c.so not found: run scripts/build_relay.sh or set RATCHET_RELAY_LIB")


_lib = None


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(_find_lib())
        P, I, U64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        IP = ctypes.POINTER(ctypes.c_int)
        sig = {
            "relay_last_error": (ctypes.c_char_p, []),
            "relay_cuda_available": (I, []),
            "relay_model_load": (P, [ctypes.c_char_p]),
            "relay_model_free": (None, [P]),
            "relay_model_config": (I, [P, ctypes.POINTER(_Config)]),
            "relay_model_rope_inv_freq": (I, [P, ctypes.POINTER(ctypes.c_float)]),
            "relay_model_tensor": (ctypes.POINTER(ctypes.c_float), [P, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64)]),
            "relay_engine_new": (P, [P, ctypes.c_char_p, I, I, I, I, I, I]),
            "relay_engine_free": (None, [P]),
            "relay_engine_add": (I, [P, U64, IP, I, ctypes.POINTER(_Sampling)]),
            "relay_engine_add_resume": (I, [P, U64, IP, I, ctypes.POINTER(_Sampling), IP, I]),
            "relay_engine_step": (I, [P, ctypes.POINTER(_Event), I]),
            "relay_engine_has_work": (I, [P]),
            "relay_engine_cancel": (I, [P, U64]),
            "relay_engine_cancel_all": (None, [P]),
            "relay_engine_reload_weights": (I, [P]),
            "relay_engine_stats": (I, [P, ctypes.POINTER(_Stats)]),
        }
        for name, (res, args) in sig.items():
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _lib = L
    return _lib


def _check(rc):
    if rc == -1 or rc is None:
        raise RelayError(lib().relay_last_error().decode())
    return rc


def cuda_available():
    return bool(lib().relay_cuda_available())


@dataclass(frozen=True)
class Config:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    max_position: int
    rms_eps: float
    rope_theta: float
    tie_embeddings: bool
    qkv_bias: bool
    eos_ids: tuple


class Model:
    """A model's float32 weights on the host, in relay's fused layout. tensor() gives
    writable numpy views of them: copying a trainer's parameters into these, then
    Engine.reload_weights(), is the simplest weight sync."""

    def __init__(self, path):
        self._h = lib().relay_model_load(os.fsencode(path))
        if not self._h:
            _check(-1)
        c = _Config()
        _check(lib().relay_model_config(self._h, ctypes.byref(c)))
        self.config = Config(c.hidden, c.intermediate, c.layers, c.heads, c.kv_heads, c.head_dim, c.vocab,
                             c.max_position, c.rms_eps, c.rope_theta, bool(c.tie_embeddings), bool(c.qkv_bias),
                             tuple(c.eos_ids[i] for i in range(c.num_eos)))
        self.path = path
        self._views = {}

    def rope_inv_freq(self):
        out = np.empty(self.config.head_dim // 2, dtype=np.float32)
        _check(lib().relay_model_rope_inv_freq(self._h, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        return out

    def tensor(self, name):
        """A writable float32 view of one weight, or None if the model has no such tensor."""
        if name not in self._views:
            n = ctypes.c_int64()
            p = lib().relay_model_tensor(self._h, name.encode(), ctypes.byref(n))
            if not p:
                return None
            self._views[name] = np.ctypeslib.as_array(p, shape=(n.value,))
        return self._views[name]  # valid while this Model lives

    def tensor_names(self):
        names = ["embed", "final_norm"]
        if not self.config.tie_embeddings:
            names.append("lm_head")
        per = ["attn_norm", "wqkv"] + (["bqkv"] if self.config.qkv_bias else []) + ["wo", "mlp_norm", "w_gate_up", "w_down"]
        for i in range(self.config.layers):
            names += [f"layers.{i}.{p}" for p in per]
        return names

    def __del__(self):
        if getattr(self, "_h", None):
            lib().relay_model_free(self._h)
            self._h = None


@dataclass
class Event:
    id: int
    token: int
    index: int
    finish: int
    logprob: float


@dataclass
class Completion:
    tokens: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    finish: int = FINISH_NONE
    start: int = 0  # token index of tokens[0]: len(generated) for a resumed rollout


class Engine:
    """relay's continuous-batching engine on one backend ("cpu" or "cuda"). Every
    generated token comes with its log-probability under softmax(logits / temperature)
    over the whole vocabulary: with temperature 1 and no top-k/top-p, exactly the
    policy log-probability an RL trainer needs."""

    def __init__(self, model, backend="cpu", device=0, num_blocks=256, block_size=16, max_batch_tokens=512,
                 max_seqs=64, prefix_caching=True):
        self.model = model
        self.max_seqs = max_seqs
        self._h = lib().relay_engine_new(model._h, backend.encode(), device, num_blocks, block_size,
                                         max_batch_tokens, max_seqs, int(prefix_caching))
        if not self._h:
            _check(-1)
        self._events = (_Event * max(max_seqs, 1))()

    @staticmethod
    def _sampling(temperature, top_p, top_k, max_new_tokens, seed, ignore_eos):
        return _Sampling(temperature, top_p, top_k, max_new_tokens, seed, int(ignore_eos))

    def add(self, id, prompt, max_new_tokens, temperature=1.0, top_p=1.0, top_k=0, seed=0, ignore_eos=False):
        p = (ctypes.c_int * len(prompt))(*prompt)
        sp = self._sampling(temperature, top_p, top_k, max_new_tokens, seed, ignore_eos)
        _check(lib().relay_engine_add(self._h, id, p, len(prompt), ctypes.byref(sp)))

    def add_resume(self, id, prompt, generated, max_new_tokens, temperature=1.0, top_p=1.0, top_k=0, seed=0,
                   ignore_eos=False):
        """Continue a rollout that already produced `generated` (token index len(generated) next)."""
        p = (ctypes.c_int * len(prompt))(*prompt)
        g = (ctypes.c_int * max(len(generated), 1))(*generated)
        sp = self._sampling(temperature, top_p, top_k, max_new_tokens, seed, ignore_eos)
        _check(lib().relay_engine_add_resume(self._h, id, p, len(prompt), ctypes.byref(sp), g, len(generated)))

    def step(self):
        n = _check(lib().relay_engine_step(self._h, self._events, len(self._events)))
        return [Event(e.id, e.token, e.index, e.finish, e.logprob) for e in self._events[:n]]

    def has_work(self):
        return bool(lib().relay_engine_has_work(self._h))

    def cancel(self, id):
        return bool(lib().relay_engine_cancel(self._h, id))

    def cancel_all(self):
        lib().relay_engine_cancel_all(self._h)

    def reload_weights(self):
        """Use the model's current host weights (after a trainer wrote them); drops the prefix cache."""
        _check(lib().relay_engine_reload_weights(self._h))

    def stats(self):
        s = _Stats()
        _check(lib().relay_engine_stats(self._h, ctypes.byref(s)))
        return {k: getattr(s, k) for k, _ in _Stats._fields_}

    def run(self):
        """Steps until every request has finished; returns {id: Completion}."""
        out = {}
        while self.has_work():
            for e in self.step():
                c = out.get(e.id)
                if c is None:
                    c = out[e.id] = Completion(start=e.index)
                assert e.index == c.start + len(c.tokens), "events arrive in token order"
                c.tokens.append(e.token)
                c.logprobs.append(e.logprob)
                c.finish = e.finish
        return out

    def __del__(self):
        if getattr(self, "_h", None):
            lib().relay_engine_free(self._h)
            self._h = None

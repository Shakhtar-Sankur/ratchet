"""The long tail on the CPU test model (chat-tiny with a boosted end-of-sequence row:
answer lengths median 5, 90th percentile 23, max 48). Trains 4 groups of 8 per step
for 20 steps, four ways, and reports the engine's forward passes and the wall time per
step. A tiny model on a CPU: the shape of the effect, not GPU numbers (those are M5's)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from test_longtail import PARTIAL, make, prompts  # noqa: E402

STEPS = 20


def run(name, g, use_async):
    t0 = time.perf_counter()
    if use_async:
        def batches():
            for s in range(STEPS):
                yield prompts(s, g.wanted() if g.cfg.partial else 4), None
        ms = list(g.run_async(batches()))
    else:
        ms = [g.step(prompts(s, g.wanted() if g.cfg.partial else 4)) for s in range(STEPS)]
    wall = time.perf_counter() - t0
    groups = sum(m["groups"] for m in ms)
    passes = g.engine.stats()["steps"]
    gen = sum(m["time_generate"] for m in ms)
    print(f"{name:34s} groups trained {groups:3d}   engine passes/group {passes / groups:6.1f}   "
          f"time/step {1000 * wall / STEPS:6.1f} ms   (generate {1000 * gen / STEPS:5.1f} ms)")


for lr in (0.0, 3e-2):
    print(f"lr = {lr}")
    run("synchronous", make(lr=lr), False)
    run("partial (6 in flight, 4 per step)", make(lr=lr, **PARTIAL), False)
    run("one step ahead", make(lr=lr), True)
    run("partial + one step ahead", make(lr=lr, **PARTIAL), True)

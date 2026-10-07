"""Summarizes the JSON lines in a runs/ directory (check, eval, and per-step records)."""
import glob
import json
import os
import statistics
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "runs"
for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    recs = [json.loads(line) for line in open(path)]
    name = os.path.basename(path)[:-6]
    for r in recs:
        if r.get("phase") == "check":
            ro, gp, sy = r["rollout"], r["logprob_gap"], r["sync"]
            print(f"[{name}] rollout: {ro['samples']} samples, {ro['tokens']} tokens, {ro['tok_per_s']:.0f} tok/s; "
                  f"lengths median {ro['len_median']}, p90 {ro['len_p90']}, max {ro['len_max']}")
            print(f"[{name}] fp16 relay vs fp32 trainer log-prob: max |gap| {gp['max_abs']:.3g}, mean {gp['mean_abs']:.3g}, "
                  f"p99 {gp['p99_abs']:.3g}, mean signed {gp['mean_signed']:.3g}; ratio outside 1±0.2: "
                  f"{100 * gp['ratio_outside_clip_0.2']:.2f}% of tokens")
            print(f"[{name}] weight sync ({sy['params'] / 1e6:.0f}M params): reload {sy['reload_seconds']:.3f} s, "
                  f"push {sy['push_seconds_median']:.3f} s ({sy['speedup']:.1f}x, {sy['push_GBps']:.1f} GB/s); "
                  f"identical rollouts: {sy['push_equals_reload']}")
            au = r.get("gap_after_updates")
            if au:
                for k in ("float32_weights", "fp16_rounded_weights"):
                    g = au[k]
                    print(f"[{name}] after {au['steps']} GRPO steps (lr {au['lr']:g}), trainer with {k.replace('_', ' ')}: "
                          f"max |gap| {g['max_abs']:.3g}, mean {g['mean_abs']:.3g}, p99 {g['p99_abs']:.3g}; "
                          f"ratio outside 1±0.2: {100 * g['ratio_outside_clip_0.2']:.2f}% of tokens")
        elif r.get("phase") == "eval":
            print(f"[{name}] eval {r.get('when', '')}: accuracy {100 * r['accuracy']:.1f}% of {int(r['n'])}, "
                  f"'####' format {100 * r['format_rate']:.0f}%, truncated {100 * r['truncated']:.0f}%, "
                  f"mean {r['mean_tokens']:.0f} tokens, {r['tok_per_s']:.0f} tok/s")
    steps = [r for r in recs if "step" in r]
    if steps:
        def med(k):
            v = [s[k] for s in steps if s.get(k) is not None]
            return statistics.median(v) if v else float("nan")
        first = statistics.mean(s["reward"] for s in steps[:10])
        last = statistics.mean(s["reward"] for s in steps[-10:])
        print(f"[{name}] {len(steps)} steps, {steps[0]['mode']}: step {med('time_step'):.1f} s (generate "
              f"{med('time_generate'):.1f}, train {med('time_train'):.1f}, sync {med('time_sync'):.2f}); "
              f"reward first 10 {first:.3f} -> last 10 {last:.3f}; max length median {med('max_length'):.0f}; "
              f"log-prob gap max (fresh samples) median {med('logprob_gap_max'):.3g}, mean {med('logprob_gap_mean'):.2g}, "
              f"ratio off by >0.2 {100 * med('logprob_gap_frac_over_0.2'):.2f}%; clip {100 * med('clip_frac'):.1f}%; "
              f"total {steps[-1].get('elapsed', 0) / 60:.1f} min")

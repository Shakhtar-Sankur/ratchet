"""One line per run of scripts/kaggle_mlsys.sh: accuracy after training, the relay/trainer
log-probability gap on fresh samples, how many tokens leave the 1 +- 0.2 ratio band, and the
truncated importance sampling weights. Usage: python scripts/summarize_mlsys.py RUNS_DIR"""

import glob
import json
import os
import statistics
import sys


def main(d):
    print(f"{'run':16s} {'acc':>6s} {'n':>5s} {'gap mean':>9s} {'gap max':>8s} {'>0.2':>7s} "
          f"{'tis w':>6s} {'capped':>7s} {'reward 1-10':>11s} {'91-100':>7s} {'step s':>7s}")
    for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        recs = [json.loads(line) for line in open(path) if line.startswith("{")]
        name = os.path.basename(path)[:-6]
        ev = [r for r in recs if r.get("phase") == "eval" or "accuracy" in r]
        acc = f"{100 * ev[-1]['accuracy']:.1f}" if ev else "-"
        n = f"{int(ev[-1].get('n', 0))}" if ev else "-"
        steps = [r for r in recs if "step" in r and "reward" in r]
        if not steps:
            print(f"{name:16s} {acc:>6s} {n:>5s}")
            continue
        fresh = [s for s in steps if s.get("trained_samples")]
        med = lambda k: statistics.median(s.get(k, 0) for s in fresh) if fresh else 0
        mean = lambda k, xs: sum(s.get(k, 0) for s in xs) / max(len(xs), 1)
        print(f"{name:16s} {acc:>6s} {n:>5s} {med('logprob_gap_mean'):9.5f} {med('logprob_gap_max'):8.3f} "
              f"{100 * mean('logprob_gap_frac_over_0.2', fresh):6.2f}% {mean('tis_weight_mean', fresh):6.3f} "
              f"{100 * mean('tis_capped_frac', fresh):6.2f}% {mean('reward', steps[:10]):11.3f} "
              f"{mean('reward', steps[-10:]):7.3f} {mean('time_step', steps):7.1f}")


if __name__ == "__main__":
    main(sys.argv[1])

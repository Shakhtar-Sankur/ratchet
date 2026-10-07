"""Compact view of decompose.py's JSON lines (the full records stay in runs/*.jsonl)."""
import json
import sys

for line in sys.stdin:
    try:
        r = json.loads(line)
    except ValueError:
        continue
    if "experiment" in r:
        print(json.dumps(r), flush=True)
    elif "step" in r:
        if r["step"] in (1, 2, 3) or r["step"] % 5 == 0:
            print(json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    elif "checkpoint" in r:
        out = {"checkpoint": r["checkpoint"], "closest": r["closest_trainer_variant"]}
        for k, v in r["sources"].items():
            out[k] = {x: round(v[x], 5) for x in ("mean_abs", "p99_abs", "max_abs", "ratio_outside_0.2")}
        out["engine_vs"] = {k: {x: round(v[x], 5) for x in ("mean_abs", "max_abs")} for k, v in r["engine_vs_trainer"].items()}
        out["engine_vs_fp32_by_prob"] = {k: (round(v, 5) if v is not None else None)
                                         for k, v in r["engine_vs_trainer"]["fp32_weights"]["mean_abs_by_prob"].items()}
        print(json.dumps(out), flush=True)

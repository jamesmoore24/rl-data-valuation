"""Phase 2b: slice the scored pool into cohorts that VARY on the metric, held
fixed on size. Default metric = pass_rate; bands = LOW/MID/HIGH difficulty."""
from __future__ import annotations
import argparse, json, os, random

BANDS = {  # name: (lo, hi) on pass_rate  (widened so bands fill at k=8)
    "low":  (0.0, 0.25),   # ~always-fail  -> saturated, ~0 GRPO advantage
    "mid":  (0.375, 0.625),# non-saturated -> hypothesis: lifts most
    "high": (0.75, 1.0),   # ~always-pass  -> saturated, ~0 GRPO advantage
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--metric", default="pass_rate")
    ap.add_argument("--per_cohort", type=int, default=300)
    ap.add_argument("--out_dir", default="data/cohorts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.scored)]
    summary = {}
    for name, (lo, hi) in BANDS.items():
        pool = [r for r in rows if lo <= r[args.metric] <= hi]
        random.shuffle(pool)
        cohort = pool[: args.per_cohort]
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for r in cohort:
                f.write(json.dumps(r) + "\n")
        mean_metric = sum(r[args.metric] for r in cohort) / max(len(cohort), 1)
        summary[name] = {"n": len(cohort), f"mean_{args.metric}": round(mean_metric, 3)}
        print(f"{name}: {len(cohort)} tasks, mean {args.metric}={mean_metric:.3f}")
        if len(cohort) < args.per_cohort:
            print(f"  WARN: only {len(cohort)} tasks in band {name} (wanted {args.per_cohort})")
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

if __name__ == "__main__":
    main()

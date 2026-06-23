"""Phase 4: correlate cohort metric vs measured lift. Spearman ρ + scatter."""
from __future__ import annotations
import argparse, json, os
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--cohorts", default="data/cohorts")
    ap.add_argument("--metric", default="pass_rate")
    ap.add_argument("--out", default="metric_vs_lift.png")
    args = ap.parse_args()

    base = json.load(open(os.path.join(args.results_dir, "base.json")))["accuracy"]
    summary = json.load(open(os.path.join(args.cohorts, "summary.json")))

    names, xs, ys = [], [], []
    for name in ["low", "mid", "high"]:
        rp = os.path.join(args.results_dir, f"{name}.json")
        if not os.path.exists(rp):
            print(f"skip {name}: no result"); continue
        acc = json.load(open(rp))["accuracy"]
        metric = summary[name][f"mean_{args.metric}"]
        lift = acc - base
        names.append(name); xs.append(metric); ys.append(lift)
        print(f"{name:5s}  {args.metric}={metric:.3f}  lift={lift:+.3f}")

    if len(xs) >= 3:
        rho, p = spearmanr(xs, ys); r, _ = pearsonr(xs, ys)
        print(f"\nSpearman ρ = {rho:.3f} (p={p:.3f})   Pearson r = {r:.3f}   n={len(xs)}")
    else:
        rho = float("nan"); print("\nneed >=3 cohorts for a correlation")

    plt.figure(figsize=(7,5), dpi=160)
    plt.scatter(xs, ys, s=140, color="#2b4c7e", zorder=3)
    for n,x,y in zip(names,xs,ys):
        plt.annotate(n, (x,y), xytext=(6,6), textcoords="offset points")
    plt.axhline(0, color="#bbb", lw=1)
    plt.xlabel(f"cohort metric ({args.metric})"); plt.ylabel("GSM8K lift (acc − base)")
    plt.title(f"metric vs lift   (Spearman ρ = {rho:.2f}, n={len(xs)})")
    plt.tight_layout(); plt.savefig(args.out, bbox_inches="tight")
    print(f"saved -> {args.out}")

if __name__ == "__main__":
    main()

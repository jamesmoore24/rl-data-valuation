#!/usr/bin/env bash
# End-to-end dry run. Edit MODEL / sizes as needed. Run on the GPU box.
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
STEPS="${STEPS:-250}"
mkdir -p results data runs

echo "== Phase 1: baseline (smoke test) =="
python eval_gsm8k.py --model "$MODEL" --n 300 --out results/base.json

echo "== Phase 2: score pool + build cohorts =="
python score_metric.py   --model "$MODEL" --k 8 --n_tasks 1500 --out data/scored.jsonl
python cohort_builder.py --scored data/scored.jsonl --per_cohort 300 --out_dir data/cohorts

echo "== Phase 3: train + eval each cohort =="
for c in low mid high; do
  echo "-- cohort $c --"
  python train_grpo.py --cohort data/cohorts/$c.jsonl --model "$MODEL" --steps "$STEPS" --out_dir runs/$c
  python eval_gsm8k.py --model runs/$c/final --n 300 --out results/$c.json
done

echo "== Phase 4: correlate =="
python analyze.py --results_dir results --cohorts data/cohorts --out metric_vs_lift.png
echo "done -> metric_vs_lift.png"

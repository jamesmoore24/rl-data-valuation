# Does This Data Teach? — reference dry-run

Validates the hackathon problem end-to-end: **does a cheap data-quality metric predict downstream RL lift** on a small model, in a few hours, on 1 node?

Mirrors exactly what contestants do:
1. score a task pool with a cheap metric (base-model rollouts, no training)
2. slice it into cohorts that vary on the metric
3. GRPO-train a small model on each cohort
4. eval each on a held-out benchmark → lift
5. correlate metric vs lift (Spearman ρ)

Default substrate (fastest clean signal — see research): **Qwen2.5-1.5B-Instruct + GSM8K**.
Hypothesis under test: MID-difficulty (non-saturated) cohorts lift most → "non-saturation predicts lift" (DAPO mechanism).

## Run order (≈ half a day on 2–6× H100)
```bash
pip install -r requirements.txt

# Phase 1 — baseline (the contestant smoke test): does GRPO lift at all?
python eval_gsm8k.py  --model Qwen/Qwen2.5-1.5B-Instruct --n 300 --out results/base.json

# Phase 2 — score the pool + build cohorts (cheap, no training)
python score_metric.py    --model Qwen/Qwen2.5-1.5B-Instruct --k 8 --n_tasks 1500 --out data/scored.jsonl
python cohort_builder.py  --scored data/scored.jsonl --per_cohort 300 --out_dir data/cohorts

# Phase 3 — train one GRPO run per cohort, eval each
for c in low mid high; do
  python train_grpo.py  --cohort data/cohorts/$c.jsonl --model Qwen/Qwen2.5-1.5B-Instruct \
                        --steps 250 --out_dir runs/$c
  python eval_gsm8k.py  --model runs/$c/final --n 300 --out results/$c.json
done

# Phase 4 — correlate
python analyze.py --results_dir results --cohorts data/cohorts --out metric_vs_lift.png
```
`run_all.sh` chains all of it.

## Stacks
- **This repo = TRL `GRPOTrainer`** (fastest to stand up; what we use to *validate*).
- **`prime_rl_notes.md`** = how to run the same on the contestants' actual stack (prime-rl + verifiers), which is what the hackathon ships. Validate on both if time.

## What "it works" means
A monotone metric→lift relationship over the 3 cohorts (ρ > 0, MID ≥ LOW/HIGH). If flat/noisy, the problem needs a bigger model or more cohorts — better to learn that now.

> Reference / starter code. Pin versions on the GPU box; TRL/vLLM APIs move fast.

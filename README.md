# rl-data-valuation

**Which cheap, training-free metrics predict a model's post-RL performance gain — and where do they sit on the cost↔quality frontier?**

Worked solution to the **Inference-Time Compute Hackathon (Applied AI Track)**.
Full problem statement: [`docs/problem-statement.pdf`](docs/problem-statement.pdf) · author: James Moore.

## The problem (what we're actually solving)

Frontier labs buy post-training data from domain experts. Before spending GPU-hours training on it, can we *cheaply* predict whether a dataset will actually improve the model?

Concretely: define **task-level** and **dataset-level** metrics — computed with **no training** — that predict the **lift** a cohort produces after GRPO, and characterize each metric's place on the **cost–quality Pareto frontier**.

```
quality
(R²/ρ of metric→lift)
  ^
  |                  ★ the prize: cheap metric, high predictive power
  |            . - ‾ ‾
  |        . ‾          ← frontier
  |    . ‾
  |  ‾              ● expensive metric (e.g. a full mini-RL probe)
  +-------------------------------------> cost ($, wall-clock to compute the metric)
```

The deliverable is **not one metric** — it's a *portfolio* of metrics placed on that frontier, each with a **first-principles reason** it should track lift. Per the brief: *"we are more interested in your rationale for selecting your metrics."* Bonus is finding a metric that **breaks** the frontier (cheap *and* predictive).

## How we get ground-truth lift (the measurement harness)

To test whether a metric predicts lift, we first need real lift to predict. This pipeline produces it — cohort is the only variable:

1. **score** a task pool with a cheap metric — base-model rollouts, no training
2. **slice** into cohorts varying one property at a time (size held fixed; real *and* synthetic, to ablate data quality)
3. **GRPO-train** the model on each cohort, identical config across all
4. **eval** each on a held-out benchmark → `lift = acc_after − acc_before`
5. **fit** metric → lift; quantify on **held-out** cohorts (R² / RMSE / Spearman ρ)

Substrate (fastest clean signal — see research): **Qwen2.5-1.5B-Instruct + GSM8K**.

## Metric portfolio (the judged part — WIP)

Every candidate carries `(cost, rationale)`. Task-level metrics reduce to dataset-level.

| metric | level | cost | first-principles why it should predict lift |
|---|---|---|---|
| pass-rate / difficulty | task→dataset | cheap (k base rollouts) | mid (non-saturated) tasks carry non-zero advantage variance → largest GRPO gradient (DAPO mechanism) |
| reward variance | task→dataset | cheap | zero variance ⇒ zero GRPO signal; the group-normalized advantage *is* the gradient |
| _token length / reasoning steps_ | task | cheap | longer-horizon tasks have more credit-assignment surface to improve |
| _task diversity (Vendi / cosine)_ | dataset | medium | redundant cohorts waste gradient steps; coverage → broader generalization |
| _learnability / reducible-holdout loss_ | dataset | medium–high | RHO-loss intuition: train on what's learnable-but-not-learned |
| _mini-RL probe (few steps)_ | dataset | expensive | the "ground truth" upper bound — used to anchor the cost axis |

(Italic rows = planned. The rationale column is the actual submission.)

## Run order (≈ half a day on 2–6× H100, or single-GPU on Modal)
```bash
pip install -r requirements.txt

# Phase 1 — baseline smoke test: does GRPO lift at all?
python eval_gsm8k.py  --model Qwen/Qwen2.5-1.5B-Instruct --n 300 --out results/base.json

# Phase 2 — score the pool + build cohorts (cheap, no training)
python score_metric.py    --model Qwen/Qwen2.5-1.5B-Instruct --k 8 --n_tasks 1500 --out data/scored.jsonl
python cohort_builder.py  --scored data/scored.jsonl --per_cohort 300 --out_dir data/cohorts

# Phase 3 — one GRPO run per cohort, eval each
for c in low mid high; do
  python train_grpo.py  --cohort data/cohorts/$c.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --steps 250 --out_dir runs/$c
  python eval_gsm8k.py  --model runs/$c/final --n 300 --out results/$c.json
done

# Phase 4 — correlate metric vs lift
python analyze.py --results_dir results --cohorts data/cohorts --out metric_vs_lift.png
```
`run_all.sh` chains all of it. `modal_run.py` runs the whole thing single-GPU on Modal (no infra ticket).

## Stacks
- **This repo = TRL `GRPOTrainer`** — fastest to stand up; what we use to *validate*.
- **`prime_rl_notes.md`** = the same on the contestants' actual stack (prime-rl + verifiers), which the hackathon ships. Validate on both if time.

## Status
- **v0 (done):** harness validated on ONE metric (difficulty / non-saturation), 3 cohorts → proves the loop and gives the first frontier point.
- **next:** expand the metric portfolio (task + dataset level), holdout fit, and build the cost–quality frontier plot — that frontier + the per-metric rationale is the submission.

> Reference / research code. Pin versions on the GPU box; TRL/vLLM APIs move fast.

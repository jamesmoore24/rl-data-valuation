# rl-data-valuation

**Which cheap, training-free metrics predict a model's post-RL performance gain — and where do they sit on the cost↔quality frontier?**

Worked solution to the **Inference-Time Compute Hackathon (Applied AI Track)**.
Full problem statement: [`docs/problem-statement.pdf`](docs/problem-statement.pdf) · author: James Moore.
Open questions I'm working through: [`QUESTIONS.md`](QUESTIONS.md) · relevant literature: [`PAPERS.md`](PAPERS.md).

## The problem

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
  |  ‾              ● expensive metric (e.g. a few-step mini-RL probe)
  +-------------------------------------> cost ($, wall-clock to compute the metric)
```

The deliverable is **not one metric** — it's a *portfolio* of metrics placed on that frontier, each with a **first-principles reason** it should track lift. Per the brief: *"we are more interested in your rationale for selecting your metrics."* Bonus: a metric that **breaks** the frontier (cheap *and* predictive).

## Approach

1. **Measurement harness** — produce ground-truth lift: score a task pool → slice into cohorts that vary one property at a time (real + synthetic) → GRPO-train each on an identical config (cohort = the only variable) → eval → `lift = acc_after − acc_before`.
2. **Candidate metrics** — compute cheap task/dataset metrics with no training; each carries `(value, $, wall_clock)`.
3. **Fit & place** — fit metric → lift, score on **held-out** cohorts (R²/RMSE/ρ), and plot each metric on the cost–quality frontier.

Substrate (fastest clean signal): **Qwen2.5-1.5B-Instruct + GSM8K**.
Stack: validate fast on **TRL `GRPOTrainer`**, run the real experiment on the hackathon's **`verifiers` + prime-rl** (single-turn env). "Different verifiers" = varying the reward function `V`, a cheap, controllable axis on the frontier — not an agentic harness like Harbor (overkill for math; revisit only if we extend to code/agentic tasks).

## Metric portfolio (the judged part — WIP)

| metric | level | cost | first-principles why it should predict lift |
|---|---|---|---|
| pass-rate / difficulty | task→dataset | cheap (k base rollouts) | mid (non-saturated) tasks carry non-zero advantage variance → largest GRPO gradient (DAPO) |
| reward variance | task→dataset | cheap | zero variance ⇒ zero GRPO signal; group-normalized advantage *is* the gradient |
| token length / reasoning steps | task | cheap | longer-horizon tasks have more credit-assignment surface to improve |
| task diversity (Vendi / cosine) | dataset | medium | redundant cohorts waste gradient steps; coverage → broader generalization |
| learnability / reducible-holdout loss | dataset | medium–high | RHO-loss: train on what's learnable-but-not-learned |
| mini-RL probe (few steps) | dataset | expensive | the upper-bound anchor for the cost axis |

(Rationale column is the actual submission; rows get validated against measured lift.)

## Status

Clean slate — problem framing + plan committed. Building the harness next on TRL → verifiers/prime-rl. Tracking unknowns in [`QUESTIONS.md`](QUESTIONS.md).

# Open Questions

A running log. Each question gets a working answer as I learn — `A:` filled in along the way, with a source/experiment when I have one. The *rationale* this builds is the actual hackathon deliverable.

Status key: 🔴 open · 🟡 partial · 🟢 answered

---

## 1. Problem framing
- 🔴 **What exactly is "lift" here, and at what altitude?** Per-task Δaccuracy, or dataset-level Δbenchmark? `acc_after − acc_before` on a held-out benchmark — but held-out from what?
  - A:
- 🔴 **What's the unit of value — a task, a trajectory, or a cohort?** (Aksh's argument: trajectory is the right atom.)
  - A:
- 🔴 **What would actually convince a frontier lab to buy/skip a dataset?** i.e. what decision does this metric serve?
  - A:

## 2. Metric selection — the judged rationale
### Task-level
- 🟡 **Why should non-saturation / mid-difficulty predict lift?** Hypothesis: mid pass-rate → non-zero advantage variance → largest GRPO gradient (DAPO). Need to confirm it holds, not just assert.
  - A:
- 🔴 **Reward variance vs difficulty — are they the same signal or distinct?** Zero variance ⇒ zero GRPO gradient; but is variance more direct than pass-rate?
  - A:
- 🔴 **Do token length / reasoning steps / branching factor add predictive power beyond difficulty?**
  - A:

### Dataset-level
- 🔴 **Does diversity (Vendi / cosine coverage) predict lift independent of difficulty?** Redundant cohorts should waste gradient steps.
  - A:
- 🔴 **Is learnability (reducible-holdout loss / RHO) worth its cost?** "Train on what's learnable-but-not-yet-learned."
  - A:
- 🔴 **How do task-level metrics best reduce to dataset-level — mean, variance, tail?**
  - A:

## 3. The cost–quality frontier (the deliverable)
- 🔴 **How do I make the cost axis real?** Every metric must report `(value, $, wall_clock)`. What's the honest accounting (base rollouts? judge tokens?).
  - A:
- 🔴 **Which cheap metric is the candidate to *break* the frontier** (cheap AND predictive)?
  - A:
- 🔴 **What's the expensive anchor** — a few-step mini-RL probe as the "ground-truth-ish" upper bound?
  - A:

## 4. Verifier as a variable
- 🔴 **Does reward shaping change lift on a fixed cohort?** exact-match vs format-reward vs partial-credit vs LLM-judge.
  - A:
- 🔴 **Where do verifiers sit on the cost axis?** exact-match ≈ $0; LLM-judge = real tokens. Is a cheap verifier good enough?
  - A:

## 5. Measurement validity (does my lift signal even mean anything?)
- 🟡 **Is run-to-run variance just temperature?** No — 4 sources: sampling (temp), numerical nondeterminism (even temp=0), env, grader. Temp=0 ≠ deterministic without batch-invariant kernels.
  - A: confirmed conceptually; need to quantify on GSM8K — how much of my lift Δ is noise?
- 🔴 **How many seeds / eval-n do I need so lift Δ between cohorts is distinguishable from noise?** (the floor below which a result is jitter.)
  - A:
- 🔴 **Does a variance floor measured on GSM8K transfer to other task types?** Likely heteroscedastic (trajectory branching is task-dependent) — so probably no without stratified sampling.
  - A:

## 6. Experimental design
- 🔴 **How many cohorts for a credible metric→lift fit?** Holdout honesty matters more than the statistic with few cohorts.
  - A:
- 🔴 **Real vs synthetic cohorts — how do I generate synthetic ones that cleanly ablate data quality** (vary ONE property, size fixed)?
  - A:
- 🔴 **What's the right fit + goodness-of-fit?** R² / RMSE / Spearman ρ on held-out cohorts.
  - A:

## 7. Substrate & scale
- 🟡 **Is Qwen2.5-1.5B + GSM8K enough signal, or will the effect be too small/noisy?** README hypothesis says monotone over 3 cohorts; if flat, need bigger model / more cohorts.
  - A:
- 🔴 **TRL (validate) vs verifiers+prime-rl (real) — do results agree?** Run both if time.
  - A:
- 🔴 **Single-GPU Modal vs the 8×H100 node — what fits where, and what's the $ per cohort run?**
  - A:

# Running the same on the contestants' stack (prime-rl + verifiers)

This repo uses TRL to *validate fast*. The hackathon ships **prime-rl + verifiers** — run it there too so we de-risk the actual contestant path.

## Install
```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash
```

## Smoke test (confirm the stack)
```bash
# their quickstart: Qwen3-0.6B reverse_text, ~2 GPUs, minutes
uv run rl @ examples/reverse_text/rl.toml
```

## Swap to our experiment
1. Install the GSM8K env:  `prime env install primeintellect/gsm8k@latest`
2. Copy `examples/reverse_text/rl.toml`, set:
   - `model.name = "Qwen/Qwen2.5-1.5B-Instruct"`
   - env → `gsm8k`
3. Launch (single command spins inference + orchestrator + trainer):
   ```bash
   uv run rl @ configs/<your>.toml --wandb.project dtdt --wandb.name mid
   ```
4. Eval:  `uv run vf-eval gsm8k -m <model> -b http://localhost:8000/v1 -n 300`

## Cohorts
Reuse `cohort_builder.py` here — point the env's dataset at `data/cohorts/{low,mid,high}.jsonl`
(verifiers envs take a HF dataset / jsonl path). One run per cohort, same config = the control.

## GPU layout (per run)
1 inference (vLLM) + 1 trainer GPU is the documented minimum; on an 8×H100 node run
multiple cohorts in parallel (≈4 runs at 2 GPUs each).

## Code track (optional)
Swap model → `Qwen/Qwen2.5-Coder-1.5B-Instruct`, env → fork `livecodebench` or wire an
EvalPlus (MBPP+) reward. Keep tasks MBPP-easy so rollout groups stay mixed pass/fail.

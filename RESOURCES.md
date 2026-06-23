# Resources

Services and interfaces available to run experiments right now.

## Compute
- **Modal** (`mercor-rl` workspace) — serverless GPU (L4 / A10G / A100 / H100), pay-per-second, scales to 0. **Ready now** — authenticated locally, used for prior dtdt runs. Default for single-GPU SFT/GRPO, sweeps, evals. No infra ticket.
- **Hackathon node** — 8× H100 (80GB), single node. Provided for the event.
- **Jupiter v2** (internal) — SageMaker HyperPod + Slurm, A100/H100. Self-serve via `sbatch -p gpu` once you have an LDAP account (one-time request in #infra-requests). Built for many-small-jobs, not big training.
- **Lambda** (external) — on-demand H100/H200, billed to personal Brex. Fallback only.

## RL / training stack
- **prime-rl + verifiers + Environments Hub** (Prime Intellect) — the hackathon's open RL stack; `verifiers` is the env/reward layer, prime-rl is the trainer. $100 Prime Intellect compute credits.
- **TRL `GRPOTrainer`** — fastest local/Modal path to validate the loop.
- **Tinker** (Thinking Machines, managed SFT/post-train API) — available at Mercor; needs `TINKER_API_KEY`. Zero-infra option for post-training.

## Experiment tracking
- **Weights & Biases** — shared `WANDB_API_KEY` in Notion `apex-rl-secrets`, or a free personal account. prime-rl and TRL both log to it.

## Model APIs
- **Anthropic API** — $100/participant hackathon credit, plus Mercor Anthropic access. For LLM-as-judge verifiers, data gen, baselines.
- **Cognition / Devin API** — hackathon access (credit amount TBC).

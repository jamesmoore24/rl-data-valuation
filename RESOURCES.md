# Resources

Services and interfaces I actually have access to right now.

## Compute
- **Modal** (`mercor-rl` workspace) — serverless GPU (L4 / A10G / A100 / H100), pay-per-second, scales to 0. **Ready now** — authenticated locally, used for prior dtdt runs. Default for single-GPU SFT/GRPO, sweeps, evals. No infra ticket. This is the go-to.
- **Lambda** (external) — on-demand H100/H200, billed to personal Brex. Fallback only.

## RL / training stack
- **prime-rl + verifiers + Environments Hub** (Prime Intellect) — open-source RL stack; `verifiers` = env/reward layer, prime-rl = trainer. Free to install (no PI credits).
- **TRL `GRPOTrainer`** — fastest local/Modal path to validate the loop.
- **Tinker** (Thinking Machines, managed post-train API) — exists at Mercor but needs `TINKER_API_KEY` (not confirmed I have one). Zero-infra post-training if available.

## Tooling / agents
- **Claude Code** — agentic coding + research in the terminal (have it). Use for building the harness, generating/curating synthetic cohorts, analysis, and writing up. This vault is driven from it.

## Experiment tracking
- **Weights & Biases** — shared `WANDB_API_KEY` in Notion `apex-rl-secrets`, or a free personal account.

## Model APIs
- **Anthropic models** — via Claude Code (have it) and Mercor's Anthropic API access, for LLM-as-judge verifiers / data gen / baselines.

## NOT available to me (don't plan around these)
- **Hackathon-provided resources** — I do NOT have these: the 8× H100 event node, $100/participant Anthropic credit, $100 Prime Intellect credits, Cognition/Devin event access.
- **Jupiter v2** (internal cluster) — gated behind Nordlayer VPN + private login host + an LDAP account (one-time #infra-requests provisioning). Verified not reachable now (`jupiter.mercor.com` doesn't resolve; never connected). Not worth it vs. Modal for this project.

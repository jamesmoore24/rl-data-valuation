# Papers

Literature relevant to the project (the ones I linked in the problem statement). Notes go under each.

## [Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt](https://arxiv.org/abs/2206.07137)
Mindermann et al., 2022 — RHO-Loss; select points that most reduce holdout loss (learnable, not-yet-learned).



## [LIMR: Less is More for RL Scaling](https://arxiv.org/abs/2502.11886)
Li et al., 2025 — Learning Impact Measurement; 1,389 chosen samples beat the full 8,523.



## [Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning](https://arxiv.org/abs/2504.03380)
Bae et al., 2025 — select intermediate-difficulty samples; directly relevant to the mid-cohort hypothesis.



## [Train at Moving Edge: Online-Verified Prompt Selection for Efficient RL Training of Large Reasoning Models](https://arxiv.org/abs/2603.25184)
Wu et al., 2026 — HIVE; pick high-utility prompts at the "learning edge" using historical reward + entropy.



## [ExGRPO: Learning to Reason from Experience](https://arxiv.org/abs/2510.02245)
Zhan et al., 2025 — identify valuable rollouts via correctness + entropy, reuse with a mixed-policy objective.



## [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617)
Cui et al., 2025 — policy-entropy collapse as the barrier; Clip-Cov / KL-Cov to sustain exploration.



## [The Vendi Score: A Diversity Evaluation Metric for Machine Learning](https://arxiv.org/abs/2210.02410)
Friedman & Dieng, 2022 — domain-agnostic diversity metric (entropy of a similarity matrix's eigenvalues).



## [SemDeDup: Data-efficient Learning at Web-scale through Semantic Deduplication](https://arxiv.org/abs/2303.09540)
Abbas et al., 2023 — remove semantic duplicates via pre-trained embeddings; redundancy reduction.



## Background
- [A Primer on LLM Post-Training](https://pytorch.org/blog/a-primer-on-llm-post-training/) — PyTorch blog.
- [`pass_rate` example implementation](https://pastebin.com/1Np5UuQG) — the F() reference from the problem statement.

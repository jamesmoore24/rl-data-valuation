"""Phase 3: GRPO-train the model on one cohort. TRL GRPOTrainer + vLLM.
Reward = answer correctness (verifiable). Same config across cohorts = the control."""
from __future__ import annotations
import argparse, json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from utils import build_prompt, is_correct

def load_cohort(path):
    rows = [json.loads(l) for l in open(path)]
    return Dataset.from_list([
        {"prompt": build_prompt(r["question"]), "answer": r["answer"]} for r in rows
    ])

def reward_correct(completions, answer, **kwargs):
    return [1.0 if is_correct(c, a) else 0.0 for c, a in zip(completions, answer)]

def reward_format(completions, **kwargs):
    return [0.1 if "answer is" in c.lower() else 0.0 for c in completions]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ds = load_cohort(args.cohort)
    # TRL's GRPOConfig signature drifts across versions — keep only supported kwargs.
    import dataclasses
    desired = dict(
        output_dir=args.out_dir,
        max_steps=args.steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        max_prompt_length=512,
        max_completion_length=512,
        temperature=0.8,
        logging_steps=10,
        save_strategy="no",
        use_vllm=False,             # robust: HF generate (avoids vLLM 0.23/TRL incompat).
        bf16=True,
        report_to="none",
    )
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = [k for k in desired if k not in valid]
    if dropped:
        print(f"[train_grpo] dropping unsupported GRPOConfig kwargs: {dropped}")
    cfg = GRPOConfig(**{k: v for k, v in desired.items() if k in valid})
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_correct, reward_format],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(f"{args.out_dir}/final")
    print(f"saved -> {args.out_dir}/final")

if __name__ == "__main__":
    main()

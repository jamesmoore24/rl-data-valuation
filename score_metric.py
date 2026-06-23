"""Phase 2a: score each train task with a cheap per-task metric from base-model
rollouts (NO training). Emits pass_rate and reward_variance per task."""
from __future__ import annotations
import argparse, json
import numpy as np
from vllm import LLM
from utils import load_gsm8k, gold_answer, build_prompt, is_correct, vllm_generate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8, help="rollouts per task (the N in F(T,M))")
    ap.add_argument("--n_tasks", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="data/scored.jsonl")
    args = ap.parse_args()

    ds = load_gsm8k("train").select(range(args.n_tasks))
    prompts = [build_prompt(q) for q in ds["question"]]
    golds = [gold_answer(e) for e in ds]

    llm = LLM(model=args.model, gpu_memory_utilization=0.9)
    gens = vllm_generate(llm, prompts, n=args.k, temperature=args.temperature)

    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for q, g, comps in zip(ds["question"], golds, gens):
            rewards = [1.0 if is_correct(c, g) else 0.0 for c in comps]
            pass_rate = float(np.mean(rewards))
            variance = float(np.var(rewards))
            f.write(json.dumps({
                "question": q, "answer": g,
                "pass_rate": pass_rate, "reward_variance": variance,
            }) + "\n")
    print(f"scored {len(golds)} tasks -> {args.out}")

if __name__ == "__main__":
    main()

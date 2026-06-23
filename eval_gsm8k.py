"""Phases 1 & 3: eval a model's GSM8K test pass@1 (greedy). Writes {accuracy,n}."""
from __future__ import annotations
import argparse, json, os
from vllm import LLM
from utils import load_gsm8k, gold_answer, build_prompt, is_correct, vllm_generate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_gsm8k("test").select(range(args.n))
    prompts = [build_prompt(q) for q in ds["question"]]
    golds = [gold_answer(e) for e in ds]

    llm = LLM(model=args.model, gpu_memory_utilization=0.9)
    gens = vllm_generate(llm, prompts, n=1, temperature=0.0, max_tokens=512)
    correct = sum(is_correct(g[0], gold) for g, gold in zip(gens, golds))
    acc = correct / len(golds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"model": args.model, "accuracy": acc, "n": len(golds)}, open(args.out, "w"), indent=2)
    print(f"{args.model}: GSM8K pass@1 = {acc:.3f}  (n={len(golds)})")

if __name__ == "__main__":
    main()

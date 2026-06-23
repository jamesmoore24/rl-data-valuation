"""Shared helpers: GSM8K loading, answer extraction, correctness, vLLM gen."""
from __future__ import annotations
import re
from datasets import load_dataset

SYSTEM = "Solve the problem. Show brief reasoning, then end with 'The answer is <number>'."

def load_gsm8k(split: str):
    return load_dataset("openai/gsm8k", "main", split=split)

def gold_answer(example) -> int:
    # gold is after the last '####'
    a = example["answer"].split("####")[-1].strip().replace(",", "")
    return int(re.findall(r"-?\d+", a)[0])

_NUM = re.compile(r"-?\d[\d,]*")
def extract_pred(text: str):
    """Last integer in the completion (after 'answer is' if present)."""
    tail = text.split("answer is")[-1] if "answer is" in text.lower() else text
    nums = _NUM.findall(tail) or _NUM.findall(text)
    if not nums:
        return None
    try:
        return int(nums[-1].replace(",", ""))
    except ValueError:
        return None

def is_correct(completion: str, gold: int) -> bool:
    p = extract_pred(completion)
    return p is not None and p == gold

def build_prompt(question: str) -> str:
    # Qwen2.5-Instruct chat format → concise, terminating answers (else it rambles
    # to max_tokens and clipped_ratio=1, which mutes the RL signal).
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\nProblem: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def vllm_generate(llm, prompts, n=1, temperature=0.7, max_tokens=512):
    """Return list[list[str]] — n completions per prompt."""
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens, top_p=0.95)
    outs = llm.generate(prompts, sp)
    return [[o.text for o in out.outputs] for out in outs]

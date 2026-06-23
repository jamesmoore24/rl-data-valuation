"""Drive the dry-run on Modal. No SSH / no infra ticket needed.

  modal run --detach modal_run.py::smoke   # ~10 min: validates box+deps+vLLM+GSM8K
  modal run --detach modal_run.py::full    # full pipeline; writes outputs to a Volume
  modal volume get dtdt-out metric_vs_lift.png .   # fetch the result
"""
import modal

REPO = "/Users/jamesmoore/code/does-this-data-teach"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trl>=0.15", "transformers>=4.48", "vllm>=0.7", "datasets>=2.19",
        "accelerate>=1.0", "scipy>=1.11", "matplotlib>=3.8", "numpy>=1.26",
    )
    # vLLM 0.23 FlashInfer sampler crashes during KV-cache profiling on this image;
    # route sampling to the native path. Also disable usage stats. (env before add_local_*)
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_DO_NOT_TRACK": "1"})
    .add_local_dir(REPO, remote_path="/root/dtdt", ignore=["__pycache__", "runs", "results", "data", "out"])
)

app = modal.App("does-this-data-teach", image=image)
vol = modal.Volume.from_name("dtdt-out", create_if_missing=True)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _sh(cmd, env=None):
    import subprocess, os
    e = os.environ.copy(); e.update(env or {})
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd="/root/dtdt", check=True, env=e)


@app.function(gpu="H100", timeout=45 * 60, volumes={"/data_out": vol})
def smoke():
    """Just the baseline eval — proves the whole stack short of training."""
    _sh(["python", "eval_gsm8k.py", "--model", MODEL, "--n", "100", "--out", "/data_out/base.json"])
    vol.commit()
    print(open("/data_out/base.json").read())


@app.function(gpu="H100", timeout=45 * 60, volumes={"/data_out": vol})
def grpo_smoke():
    """Validate the TRL GRPO path cheaply: score a little, train MID for a few steps."""
    _sh(["python", "score_metric.py", "--model", MODEL, "--k", "4", "--n_tasks", "200", "--out", "data/scored.jsonl"])
    _sh(["python", "cohort_builder.py", "--scored", "data/scored.jsonl", "--per_cohort", "32", "--out_dir", "data/cohorts"])
    _sh(["python", "train_grpo.py", "--cohort", "data/cohorts/mid.jsonl", "--model", MODEL, "--steps", "8", "--out_dir", "runs/smoke"])
    print("GRPO smoke OK")


@app.function(gpu="H100", timeout=8 * 60 * 60, volumes={"/data_out": vol})
def full(steps: int = 150, n_tasks: int = 1500, per_cohort: int = 100, eval_n: int = 250):
    """End-to-end: baseline → score → cohorts → train×3 → eval → correlate.
    Smaller defaults than run_all.sh for a fast feasibility signal on 1 GPU."""
    import shutil, glob, os
    os.makedirs("/data_out", exist_ok=True)

    _sh(["python", "eval_gsm8k.py", "--model", MODEL, "--n", str(eval_n), "--out", "results/base.json"])
    _sh(["python", "score_metric.py", "--model", MODEL, "--k", "8", "--n_tasks", str(n_tasks), "--out", "data/scored.jsonl"])
    _sh(["python", "cohort_builder.py", "--scored", "data/scored.jsonl", "--per_cohort", str(per_cohort), "--out_dir", "data/cohorts"])
    for c in ["low", "mid", "high"]:
        _sh(["python", "train_grpo.py", "--cohort", f"data/cohorts/{c}.jsonl", "--model", MODEL, "--steps", str(steps), "--out_dir", f"runs/{c}"])
        _sh(["python", "eval_gsm8k.py", "--model", f"runs/{c}/final", "--n", str(eval_n), "--out", f"results/{c}.json"])
    _sh(["python", "analyze.py", "--results_dir", "results", "--cohorts", "data/cohorts", "--out", "metric_vs_lift.png"])

    # persist outputs to the Volume so we can fetch them locally
    for p in glob.glob("results/*.json") + ["data/cohorts/summary.json", "metric_vs_lift.png"]:
        if os.path.exists(p):
            shutil.copy(p, f"/data_out/{os.path.basename(p)}")
    vol.commit()
    print("\n== DONE — outputs in Volume dtdt-out ==")

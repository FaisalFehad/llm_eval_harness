#!/usr/bin/env python3
"""Sweep batch sizes and report speed/accuracy on a fixed N-job subset."""

import subprocess, json, re, os
from pathlib import Path

MODEL = os.path.expanduser("~/qwen3_4B_v15_mlx6bit")
PROMPT = "versions/v16/prompts/student.txt"
TEST_FILE = "versions/v12/data/v12_original/test_labeled_audited.jsonl"
OUTPUT_DIR = "eval_results/v16_batch_sweep"
N_JOBS = 48          # enough to stress-test but fast
BATCH_SIZES = [1, 2, 4, 6, 8, 12, 16, 24, 32]

def run(bs: int) -> dict:
    out_dir = f"{OUTPUT_DIR}/bs{bs}"
    cmd = [
        "python3", "versions/v16/scripts/eval_optimized.py",
        "--model", MODEL,
        "--prompt", PROMPT,
        "--test-file", TEST_FILE,
        "--batch-size", str(bs),
        "--n-jobs", str(N_JOBS),
        "--no-think",
        "--output-dir", out_dir,
        "--save-predictions",
    ]
    print(f"\n{'='*60}\n  batch_size={bs}\n{'='*60}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    stdout = proc.stdout + proc.stderr

    # Parse Duration line: "Duration: 22.3s (0.93s/job, 24 jobs)"
    dur_match = re.search(r"Duration:\s+([\d.]+)s\s+\(([\d.]+)s/job", stdout)
    duration = float(dur_match.group(1)) if dur_match else None
    per_job = float(dur_match.group(2)) if dur_match else None

    # Parse summary JSON if written
    # Find the .summary.json path in stdout
    sum_match = re.search(r"Summary:\s+(\S+)", stdout)
    summary = {}
    if sum_match:
        try:
            with open(sum_match.group(1)) as f:
                summary = json.load(f)
        except Exception:
            pass

    print(f"  duration={duration}s  per_job={per_job}s  accuracy={summary.get('label_accuracy')}%")
    return {
        "batch_size": bs,
        "duration": duration,
        "per_job": per_job,
        **{k: summary.get(k) for k in (
            "label_accuracy","score_accuracy","parse_failures",
            "invalid_tokens","n_valid","n_total",
        )},
    }

if __name__ == "__main__":
    results = []
    for bs in BATCH_SIZES:
        # Skip batch sizes larger than N_JOBS
        if bs > N_JOBS:
            continue
        try:
            results.append(run(bs))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"batch_size": bs, "error": str(e)})

    # Print table
    print(f"\n{'='*80}\n  BATCH SWEEP RESULTS  (n={N_JOBS} jobs)\n{'='*80}")
    print(f"{'BS':>4}  {'sec/job':>8}  {'total':>8}  {'label%':>8}  {'parse':>6}  {'invalid':>8}")
    print("-"*60)
    for r in results:
        print(f"{r.get('batch_size', '?'):>4}  "
              f"{r.get('per_job') or 0:>8.3f}  "
              f"{r.get('duration') or 0:>8.1f}  "
              f"{r.get('label_accuracy') or 0:>8.1f}  "
              f"{r.get('parse_failures') or 0:>6}  "
              f"{r.get('invalid_tokens') or 0:>8}")

    # Find best (lowest per_job time with <2 parse failures)
    valid = [r for r in results if not r.get("error") and (r.get("parse_failures") or 0) <= 2]
    if valid:
        best = min(valid, key=lambda r: r["per_job"])
        print(f"\n  BEST batch_size={best['batch_size']} at {best['per_job']:.3f}s/job")
        print(f"  Label accuracy: {best.get('label_accuracy')}%  (parse={best.get('parse_failures')}, invalid={best.get('invalid_tokens')})")

    # Save JSON
    out_json = Path(OUTPUT_DIR) / "sweep_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_json}")

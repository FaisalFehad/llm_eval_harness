#!/usr/bin/env python3
"""
V15 checkpoint sweep — evaluates all MLX LoRA checkpoints.

For each checkpoint:
  1. Run MLX model inference (eval_student_v7.py)
  2. Run V13.1 hybrid scoring (compute_hybrid_v13_1.py)
  3. Collect model-only + hybrid accuracy

Prints a ranked summary table at the end.
Checkpoints are MLX format: finetune/adapters_v15_4B/NNNNNNN_adapters.safetensors

Usage:
    # Sweep all checkpoints:
    .venv/bin/python3 finetune/sweep_v15.py

    # Sweep specific iters:
    .venv/bin/python3 finetune/sweep_v15.py --iters 400 500 600

    # Skip already-scored:
    .venv/bin/python3 finetune/sweep_v15.py --skip-existing

    # Dry-run:
    .venv/bin/python3 finetune/sweep_v15.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = ".venv/bin/python3"
MODEL = "mlx-community/Qwen3-4B-bf16"
ADAPTER_DIR = Path("finetune/adapters_v15_4B")
TEST_FILE = "versions/v12/data/v12_original/test_labeled_audited.jsonl"
PROMPT = "prompts/student_v15.txt"
RESULTS_DIR = "eval_results/v15_sweep_4B"


def find_checkpoints(iters=None):
    """Return sorted list of checkpoint adapter files."""
    if not ADAPTER_DIR.exists():
        print(f"ERROR: {ADAPTER_DIR} does not exist.")
        sys.exit(1)

    all_ckpts = sorted(
        p for p in ADAPTER_DIR.glob("*_adapters.safetensors")
        if p.name[0].isdigit()
    )

    if iters:
        iter_set = set(iters)
        all_ckpts = [p for p in all_ckpts if int(p.name[:7]) in iter_set]

    return all_ckpts


def run_model_inference(adapter_path, output_dir):
    """Run eval_student_v7.py for this checkpoint."""
    cmd = [
        PYTHON, "finetune/eval_student_v7.py",
        "--model", MODEL,
        "--adapter", str(adapter_path),
        "--test-file", TEST_FILE,
        "--prompt", PROMPT,
        "--output-dir", output_dir,
        "--save-predictions",
        "--no-think",
        "--max-tokens", "300",
    ]
    print(f"    Running: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, text=True, timeout=7200)
    if result.returncode != 0:
        print(f"    INFERENCE FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[-500:]}")
        return False
    return True


def run_hybrid_scoring(predictions_file, output_file):
    """Run compute_hybrid_v13_1.py on predictions."""
    cmd = [
        PYTHON, "finetune/compute_hybrid_v13_1.py",
        "--test-file", TEST_FILE,
        "--predictions", predictions_file,
        "--v12",
        "--output", output_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"    HYBRID SCORING FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"    stderr: {result.stderr[-500:]}")
        return False
    return True


def main():
    global MODEL, ADAPTER_DIR, RESULTS_DIR, PROMPT

    parser = argparse.ArgumentParser(description="V15 checkpoint sweep")
    parser.add_argument("--iters", type=int, nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter-dir", default=str(ADAPTER_DIR))
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    MODEL = args.model
    ADAPTER_DIR = Path(args.adapter_dir)
    RESULTS_DIR = args.results_dir
    PROMPT = args.prompt

    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints(args.iters)
    print(f"Found {len(checkpoints)} checkpoints in {ADAPTER_DIR}")
    for ckpt in checkpoints:
        print(f"  {ckpt.name}")

    if args.dry_run:
        print("\n[DRY RUN — no scoring performed]")
        return

    results = []

    for ckpt in checkpoints:
        iter_num = int(ckpt.name[:7])
        print(f"\n{'='*60}")
        print(f"Checkpoint: iter {iter_num}")
        print(f"{'='*60}")

        # Check if already scored
        hybrid_file = results_path / f"hybrid_iter{iter_num:07d}.json"
        if args.skip_existing and hybrid_file.exists():
            print(f"  Skipping (already scored)")
            with open(hybrid_file) as f:
                results.append((iter_num, json.load(f)))
            continue

        # Step 1: Model inference
        iter_dir = str(results_path / f"iter{iter_num:07d}")
        start = time.time()
        if not run_model_inference(ckpt, iter_dir):
            continue
        elapsed = time.time() - start
        print(f"    Inference done ({elapsed:.0f}s)")

        # Find predictions file
        pred_files = list(Path(iter_dir).glob("*.predictions.jsonl"))
        if not pred_files:
            print(f"    ERROR: No predictions file found in {iter_dir}")
            continue
        pred_file = str(pred_files[0])

        # Step 2: Hybrid scoring
        if not run_hybrid_scoring(pred_file, str(hybrid_file)):
            continue

        # Step 3: Read results
        with open(hybrid_file) as f:
            hybrid_result = json.load(f)
        results.append((iter_num, hybrid_result))

        # Print quick summary
        mo = hybrid_result.get("model_only", {})
        v12 = hybrid_result.get("v12_hybrid", {})
        print(f"    Model-only: {mo.get('accuracy_pct', '?')}% ({mo.get('correct', '?')}/{mo.get('total', '?')})")
        print(f"    Hybrid:     {v12.get('accuracy_pct', '?')}% ({v12.get('correct', '?')}/{v12.get('total', '?')})")
        print(f"    Parse fail: {mo.get('parse_fail', '?')}")

    # ── Summary table ─────────────────────────────────────────────────────
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*80}")
    print("SWEEP SUMMARY — V15 Qwen3-4B bf16")
    print(f"{'='*80}")
    header = f"{'Iter':>6s}  {'Hybrid':>8s}  {'Model-only':>10s}  {'Parse':>6s}  {'Loc':>6s}  {'Arr':>6s}  {'Sen':>6s}  {'Tech':>6s}  {'Comp':>6s}"
    print(header)
    print("-" * 80)

    for iter_num, r in sorted(results):
        mo = r.get("model_only", {})
        v12 = r.get("v12_hybrid", {})
        fa = mo.get("field_accuracy", {})
        row = (f"{iter_num:>6d}  {v12.get('accuracy_pct', 0):>7.1f}%  "
               f"{mo.get('accuracy_pct', 0):>9.1f}%  {mo.get('parse_fail', 0):>6d}  "
               f"{fa.get('loc', 0):>5.1f}%  {fa.get('arr', 0):>5.1f}%  "
               f"{fa.get('sen', 0):>5.1f}%  {fa.get('tech', 0):>5.1f}%  "
               f"{fa.get('comp', 0):>5.1f}%")
        print(row)

    # Best by model-only
    best_mo = max(results, key=lambda x: x[1].get("model_only", {}).get("accuracy_pct", 0))
    best_hy = max(results, key=lambda x: x[1].get("v12_hybrid", {}).get("accuracy_pct", 0))
    mo_acc = best_mo[1]["model_only"]["accuracy_pct"]
    hy_acc = best_hy[1]["v12_hybrid"]["accuracy_pct"]
    print(f"\nBest model-only: iter {best_mo[0]} ({mo_acc}%)")
    print(f"Best hybrid:     iter {best_hy[0]} ({hy_acc}%)")

    # Save summary
    summary = {
        "model": MODEL,
        "prompt": PROMPT,
        "test_file": TEST_FILE,
        "results": {str(iter_num): r for iter_num, r in results},
        "best_model_only_iter": best_mo[0],
        "best_hybrid_iter": best_hy[0],
    }
    summary_path = results_path / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

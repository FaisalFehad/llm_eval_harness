"""
V13.1 1.5B checkpoint sweep — runs model inference + hybrid scoring for each saved adapter.

For each checkpoint:
  1. Run model inference (eval_student_v7.py)
  2. Run V13.1 hybrid scoring (compute_hybrid_v13_1.py)
  3. Collect per-field accuracy + hybrid accuracy

Prints a ranked summary table at the end.

Usage:
    # Score ALL saved checkpoints (slow — ~18 min each on M1):
    .venv/bin/python3 finetune/sweep_v13_1_1.5B.py

    # Score specific checkpoints shortlisted from val loss:
    .venv/bin/python3 finetune/sweep_v13_1_1.5B.py --iters 1400 1600 1800 2000

    # Resume an interrupted sweep (skip already-scored checkpoints):
    .venv/bin/python3 finetune/sweep_v13_1_1.5B.py --iters 1400 1800 2000 --skip-existing

    # Dry-run: show which checkpoints would run without executing:
    .venv/bin/python3 finetune/sweep_v13_1_1.5B.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
ADAPTER_DIR = Path("finetune/adapters_v13_1_1.5B")
TEST_FILE = "data/v12/test_labeled_audited.jsonl"
PROMPT = "prompts/student_v13_1.txt"
RESULTS_DIR = "eval_results/v13_1_1.5B_sweep"

PYTHON = ".venv/bin/python3"


# ── Checkpoint discovery ───────────────────────────────────────────────────

def find_checkpoints(iters: list[int] | None) -> list[Path]:
    """Return sorted list of adapter .safetensors files to score."""
    if not ADAPTER_DIR.exists():
        print(f"ERROR: {ADAPTER_DIR} does not exist. Is training complete?")
        sys.exit(1)

    all_ckpts = sorted(ADAPTER_DIR.glob("*_adapters.safetensors"))
    if not all_ckpts:
        print(f"ERROR: No checkpoints found in {ADAPTER_DIR}")
        sys.exit(1)

    if iters:
        iter_set = set(iters)
        selected = [p for p in all_ckpts if int(p.stem.split("_")[0]) in iter_set]
        missing = iter_set - {int(p.stem.split("_")[0]) for p in selected}
        if missing:
            print(f"WARNING: Checkpoints not found for iters: {sorted(missing)}")
        return selected

    return all_ckpts


def iter_from_path(p: Path) -> int:
    return int(p.stem.split("_")[0])


# ── Model inference step ───────────────────────────────────────────────────

def run_inference(ckpt: Path, skip_existing: bool) -> Path | None:
    """
    Run eval_student_v7.py for this checkpoint.
    Returns path to the .predictions.jsonl file, or None on failure.
    """
    iter_n = iter_from_path(ckpt)
    out_dir = Path(RESULTS_DIR) / ADAPTER_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for an existing predictions file for this checkpoint
    ckpt_stem = f"{iter_n:07d}"
    existing = list(out_dir.glob(f"*_{ckpt_stem}.predictions.jsonl"))
    if skip_existing and existing:
        print(f"  [skip] iter {iter_n}: using existing predictions {existing[0].name}")
        return existing[0]

    cmd = [
        PYTHON, "finetune/eval_student_v7.py",
        "--model", MODEL,
        "--adapter", str(ckpt),
        "--test-file", TEST_FILE,
        "--prompt", PROMPT,
        "--output-dir", RESULTS_DIR,
    ]

    print(f"\n{'='*60}")
    print(f"  Inference — iter {iter_n}")
    print(f"  Checkpoint: {ckpt.name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: inference failed (exit {result.returncode})")
        return None

    # Find the freshest predictions file for this checkpoint stem
    preds = sorted(out_dir.glob(f"*_{ckpt_stem}.predictions.jsonl"), key=lambda p: p.stat().st_mtime)
    if not preds:
        # Fallback: most recently modified predictions file
        preds = sorted(out_dir.glob("*.predictions.jsonl"), key=lambda p: p.stat().st_mtime)

    if not preds:
        print(f"  ERROR: no predictions file found after inference")
        return None

    pred_file = preds[-1]
    print(f"  Done in {elapsed:.0f}s → {pred_file.name}")
    return pred_file


# ── Hybrid scoring step ────────────────────────────────────────────────────

def run_hybrid(pred_file: Path, iter_n: int) -> dict | None:
    """
    Run compute_hybrid_v13_1.py on the predictions file.
    Returns the hybrid summary dict, or None on failure.
    """
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_out = out_dir / f"hybrid_iter{iter_n:07d}.json"

    cmd = [
        PYTHON, "finetune/compute_hybrid_v13_1.py",
        "--test-file", TEST_FILE,
        "--predictions", str(pred_file),
        "--v12",
        "--output", str(hybrid_out),
    ]

    print(f"\n  Hybrid scoring — iter {iter_n}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: hybrid scoring failed:\n{result.stderr[-500:]}")
        return None

    if not hybrid_out.exists():
        print(f"  ERROR: hybrid output not written")
        return None

    with open(hybrid_out) as f:
        return json.load(f)


# ── Summary table ─────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print ranked comparison table of all checkpoint results."""
    if not results:
        print("\nNo results to summarize.")
        return

    results = sorted(results, key=lambda r: r.get("hybrid_acc", 0), reverse=True)
    best_acc = results[0]["hybrid_acc"]

    print("\n" + "="*90)
    print("V13.1 1.5B SWEEP RESULTS — ranked by hybrid accuracy")
    print("="*90)
    hdr = f"{'Iter':>7}  {'Hybrid':>7}  {'Sen':>6}  {'Arr':>6}  {'Loc':>6}  {'Tech':>6}  {'Comp':>6}  {'Parse':>5}  {'good':>5}  {'maybe':>5}  {'bad':>5}"
    print(hdr)
    print("-"*90)

    for r in results:
        acc = r.get("hybrid_acc", 0)
        fa = r.get("field_accuracy", {})
        pl = r.get("per_label", {})
        parse = r.get("parse_fail", "?")

        flag = " ★" if acc == best_acc else ""
        print(
            f"{r['iter']:>7}  {acc:>6.1f}%  "
            f"{fa.get('sen', 0):>5.1f}%  "
            f"{fa.get('arr', 0):>5.1f}%  "
            f"{fa.get('loc', 0):>5.1f}%  "
            f"{fa.get('tech', 0):>5.1f}%  "
            f"{fa.get('comp', 0):>5.1f}%  "
            f"{str(parse):>5}  "
            f"{_pct(pl, 'good_fit'):>5.0f}%  "
            f"{_pct(pl, 'maybe'):>5.0f}%  "
            f"{_pct(pl, 'bad_fit'):>5.0f}%"
            f"{flag}"
        )

    print("="*90)
    best = results[0]
    print(f"\nBest checkpoint: iter {best['iter']:,} — {best['hybrid_acc']:.1f}% hybrid")
    print(f"Adapter: {ADAPTER_DIR}/{best['iter']:07d}_adapters.safetensors")

    summary_path = Path(RESULTS_DIR) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results: {summary_path}")


def _pct(per_label: dict, label: str) -> float:
    d = per_label.get(label, {})
    total = d.get("total", 0)
    return (d.get("correct", 0) / total * 100) if total > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V13.1 1.5B checkpoint sweep")
    parser.add_argument("--iters", type=int, nargs="+",
                        help="Specific iter numbers to score (default: all checkpoints)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip checkpoints that already have predictions files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which checkpoints would run without executing")
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.iters)

    print(f"V13.1 1.5B sweep — {len(checkpoints)} checkpoint(s) to score")
    print(f"Model:    {MODEL}")
    print(f"Adapters: {ADAPTER_DIR}")
    print(f"Test:     {TEST_FILE}")
    print(f"Prompt:   {PROMPT}")

    if args.dry_run:
        print("\nCheckpoints (dry run):")
        for ckpt in checkpoints:
            print(f"  iter {iter_from_path(ckpt):,}: {ckpt.name}")
        return

    results = []

    for ckpt in checkpoints:
        iter_n = iter_from_path(ckpt)

        pred_file = run_inference(ckpt, args.skip_existing)
        if pred_file is None:
            print(f"  Skipping hybrid scoring for iter {iter_n} (inference failed)")
            continue

        hybrid = run_hybrid(pred_file, iter_n)
        if hybrid is None:
            print(f"  Skipping iter {iter_n} from summary (hybrid scoring failed)")
            continue

        # Extract metrics
        v12 = hybrid.get("v12_hybrid", {})
        total = v12.get("total", 1)
        correct = v12.get("correct", 0)
        fa = hybrid.get("v12_hybrid", {}).get("field_accuracy", {})

        results.append({
            "iter": iter_n,
            "hybrid_acc": correct / total * 100 if total > 0 else 0,
            "field_accuracy": fa,
            "parse_fail": v12.get("parse_fail", "?"),
            "per_label": hybrid.get("v12_per_label", {}),
        })

        print(f"\n  ✓ iter {iter_n}: {correct}/{total} ({correct/total*100:.1f}%)"
              f"  sen={fa.get('sen', 0):.1f}%"
              f"  parse={v12.get('parse_fail', '?')}")

    print_summary(results)


if __name__ == "__main__":
    main()

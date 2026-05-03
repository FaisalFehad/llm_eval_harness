"""
V14 checkpoint sweep — scores all HF PEFT checkpoint directories.

For each checkpoint-{step} directory:
  1. Run model inference (eval_student_v14.py)
  2. Run V13.1 hybrid scoring (compute_hybrid_v13_1.py)
  3. Collect per-field accuracy + hybrid accuracy

Prints a ranked summary table at the end.
Checkpoints are saved by HF Trainer as: finetune/adapters_v14/checkpoint-{step}/

Usage:
    # 1.5B sweep (default):
    python3 finetune/sweep_v14.py

    # 0.6B sweep:
    python3 finetune/sweep_v14.py \
        --model Qwen/Qwen3-0.6B \
        --adapter-dir finetune/adapters_v14_0.6B \
        --results-dir eval_results/v14_sweep_0.6B

    # Score specific steps:
    python3 finetune/sweep_v14.py --steps 1400 1600 1800 2000

    # Resume interrupted sweep (skip already-scored):
    python3 finetune/sweep_v14.py --skip-existing

    # Dry-run (show what would run):
    python3 finetune/sweep_v14.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "python3"

# Module-level defaults (overridden by CLI in main())
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NO_THINK = False
ADAPTER_DIR = Path("finetune/adapters_v14")
TEST_FILE = "versions/v12/data/v12_original/test_labeled_audited.jsonl"
PROMPT = "prompts/student_v14.txt"
RESULTS_DIR = "eval_results/v14_sweep"


# ── Checkpoint discovery ───────────────────────────────────────────────────

def find_checkpoints(steps: list[int] | None) -> list[Path]:
    """Return sorted list of HF checkpoint directories to score."""
    if not ADAPTER_DIR.exists():
        print(f"ERROR: {ADAPTER_DIR} does not exist. Has training started?")
        sys.exit(1)

    # HF Trainer saves checkpoints as checkpoint-{step}/
    all_ckpts = sorted(
        [p for p in ADAPTER_DIR.iterdir()
         if p.is_dir() and p.name.startswith("checkpoint-")
         and (p / "adapter_config.json").exists()],
        key=step_from_path,
    )

    if not all_ckpts:
        print(f"ERROR: No checkpoint directories found in {ADAPTER_DIR}")
        print(f"  Expected directories named: checkpoint-200, checkpoint-400, ...")
        sys.exit(1)

    if steps:
        step_set = set(steps)
        selected = [p for p in all_ckpts if step_from_path(p) in step_set]
        missing = step_set - {step_from_path(p) for p in selected}
        if missing:
            print(f"WARNING: Checkpoints not found for steps: {sorted(missing)}")
        return selected

    return all_ckpts


def step_from_path(p: Path) -> int:
    """Extract step number from checkpoint-{step} directory name."""
    return int(p.name.replace("checkpoint-", ""))


# ── Inference step ─────────────────────────────────────────────────────────

def run_inference(ckpt: Path, skip_existing: bool) -> Path | None:
    """Run eval_student_v14.py for this checkpoint."""
    step_n = step_from_path(ckpt)
    out_dir = Path(RESULTS_DIR) / ADAPTER_DIR.name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob(f"*_{step_n}.predictions.jsonl"))
    if skip_existing and existing:
        print(f"  [skip] step {step_n}: using existing {existing[0].name}")
        return existing[0]

    cmd = [
        PYTHON, "finetune/eval_student_v14.py",
        "--model", MODEL,
        "--adapter", str(ckpt),
        "--test-file", TEST_FILE,
        "--prompt", PROMPT,
        "--output-dir", RESULTS_DIR,
        "--save-predictions",
        *( ["--no-think"] if NO_THINK else []),
    ]

    print(f"\n{'='*60}")
    print(f"  Inference — step {step_n}")
    print(f"  Checkpoint: {ckpt.name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: inference failed (exit {result.returncode})")
        return None

    # Find freshest predictions file for this step
    out_subdir = Path(RESULTS_DIR) / ADAPTER_DIR.name
    preds = sorted(
        out_subdir.glob(f"*_{step_n}.predictions.jsonl"),
        key=lambda p: p.stat().st_mtime
    )
    if not preds:
        preds = sorted(out_subdir.glob("*.predictions.jsonl"), key=lambda p: p.stat().st_mtime)

    if not preds:
        print(f"  ERROR: no predictions file found after inference")
        return None

    pred_file = preds[-1]
    print(f"  Done in {elapsed:.0f}s → {pred_file.name}")
    return pred_file


# ── Hybrid scoring step ────────────────────────────────────────────────────

def run_hybrid(pred_file: Path, step_n: int) -> dict | None:
    """Run compute_hybrid_v13_1.py on predictions file."""
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_out = out_dir / f"hybrid_step{step_n:07d}.json"

    cmd = [
        PYTHON, "finetune/compute_hybrid_v13_1.py",
        "--test-file", TEST_FILE,
        "--predictions", str(pred_file),
        "--v12",
        "--output", str(hybrid_out),
    ]

    print(f"\n  Hybrid scoring — step {step_n}")
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
    if not results:
        print("\nNo results to summarize.")
        return

    results = sorted(results, key=lambda r: r.get("hybrid_acc", 0), reverse=True)
    best_acc = results[0]["hybrid_acc"]

    print("\n" + "="*90)
    print("V14 SWEEP RESULTS — ranked by hybrid accuracy")
    print("="*90)
    hdr = (f"{'Step':>7}  {'Hybrid':>7}  {'Sen':>6}  {'Arr':>6}  {'Loc':>6}  "
           f"{'Tech':>6}  {'Comp':>6}  {'Parse':>5}  {'good':>5}  {'maybe':>5}  {'bad':>5}")
    print(hdr)
    print("-"*90)

    for r in results:
        acc = r.get("hybrid_acc", 0)
        fa = r.get("field_accuracy", {})
        pl = r.get("per_label", {})
        parse = r.get("parse_fail", "?")
        flag = " ★" if acc == best_acc else ""
        print(
            f"{r['step']:>7}  {acc:>6.1f}%  "
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
    print(f"\nBest checkpoint: step {best['step']:,} — {best['hybrid_acc']:.1f}% hybrid")
    print(f"Adapter: {ADAPTER_DIR}/checkpoint-{best['step']}")

    summary_path = Path(RESULTS_DIR) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results: {summary_path}")


def _pct(per_label: dict, label: str) -> float:
    d = per_label.get(label, {})
    if isinstance(d, dict):
        total = d.get("total", 0)
        return (d.get("correct", 0) / total * 100) if total > 0 else 0.0
    return float(d) if isinstance(d, (int, float)) else 0.0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V14 checkpoint sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF model ID (default: 1.5B; use Qwen/Qwen3-0.6B for 0.6B sweep)")
    parser.add_argument("--adapter-dir", default="finetune/adapters_v14",
                        help="Directory containing checkpoint-{step}/ subdirs")
    parser.add_argument("--results-dir", default="eval_results/v14_sweep",
                        help="Where to write eval results and sweep_summary.json")
    parser.add_argument("--prompt", default="prompts/student_v14.txt")
    parser.add_argument("--test-file", default="versions/v12/data/v12_original/test_labeled_audited.jsonl")
    parser.add_argument("--steps", type=int, nargs="+",
                        help="Specific step numbers to score (default: all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip checkpoints that already have predictions files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which checkpoints would run without executing")
    parser.add_argument("--no-think", action="store_true",
                        help="Pass --no-think to eval (disables Qwen3 thinking mode)")
    args = parser.parse_args()

    # Override module-level constants with CLI args
    global ADAPTER_DIR, RESULTS_DIR, MODEL, TEST_FILE, PROMPT, NO_THINK
    MODEL = args.model
    NO_THINK = args.no_think
    ADAPTER_DIR = Path(args.adapter_dir)
    RESULTS_DIR = args.results_dir
    TEST_FILE = args.test_file
    PROMPT = args.prompt

    checkpoints = find_checkpoints(args.steps)

    print(f"V14 sweep — {len(checkpoints)} checkpoint(s) to score")
    print(f"Model:    {MODEL}")
    print(f"Adapters: {ADAPTER_DIR}")
    print(f"Test:     {TEST_FILE}")
    print(f"Prompt:   {PROMPT}")

    if args.dry_run:
        print("\nCheckpoints (dry run):")
        for ckpt in checkpoints:
            print(f"  step {step_from_path(ckpt):,}: {ckpt.name}")
        return

    results = []

    for ckpt in checkpoints:
        step_n = step_from_path(ckpt)

        pred_file = run_inference(ckpt, args.skip_existing)
        if pred_file is None:
            print(f"  Skipping hybrid scoring for step {step_n} (inference failed)")
            continue

        hybrid = run_hybrid(pred_file, step_n)
        if hybrid is None:
            print(f"  Skipping step {step_n} from summary (hybrid scoring failed)")
            continue

        v12 = hybrid.get("v12_hybrid", {})
        total = v12.get("total", 1)
        correct = v12.get("correct", 0)
        fa = v12.get("field_accuracy", {})

        results.append({
            "step": step_n,
            "hybrid_acc": correct / total * 100 if total > 0 else 0,
            "field_accuracy": fa,
            "parse_fail": v12.get("parse_fail", "?"),
            "per_label": hybrid.get("v12_per_label", {}),
        })

        print(f"\n  ✓ step {step_n}: {correct}/{total} ({correct/total*100:.1f}%)"
              f"  sen={fa.get('sen', 0):.1f}%"
              f"  parse={v12.get('parse_fail', '?')}")

    print_summary(results)


if __name__ == "__main__":
    main()

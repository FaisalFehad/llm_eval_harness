#!/usr/bin/env python3
"""
Eval MLX models against V5 semantic token eval set.

The V5 student model outputs semantic tokens (LONDON_OR_REMOTE, SENIOR_PLUS, etc.)
instead of numeric scores. This script:
  1. Runs inference with the student prompt
  2. Validates output tokens (with fuzzy matching for typos)
  3. Computes scores/labels via the code layer
  4. Compares tokens AND computed labels against golden set

Usage:
  source .venv/bin/activate

  # Student model (V5 semantic tokens):
  python3 finetune/eval_student.py \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --adapter finetune/adapters_v5 \
      --test-file data/v5/eval_150_golden.jsonl \
      --verbose

  # Baseline (no adapter):
  python3 finetune/eval_student.py \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --test-file data/v5/eval_150_golden.jsonl
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

# Import semantic token definitions
sys.path.insert(0, str(Path(__file__).parent))
from semantic_tokens import (
    FIELD_TOKENS, FIELD_MAPS,
    validate_prediction, compute_from_tokens, fuzzy_match,
)


class Tee:
    """Write to multiple streams simultaneously (terminal + file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


import shutil

import mlx.core as mx
from mlx_lm import load, generate

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_TEST_FILE = "data/v5/eval_150_golden.jsonl"
DEFAULT_PROMPT = "prompts/student_v5.txt"
MAX_TOKENS = 500

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output, handling extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def score_v5_result(predicted: dict, golden: dict) -> dict:
    """Compare predicted vs golden semantic tokens and computed labels."""

    # Validate and fuzzy-correct predicted tokens
    validation = validate_prediction(predicted)
    if not validation["valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
            "label_match": False,
            "golden_label": golden.get("label", "?"),
        }

    pred = validation["corrected"]
    fuzzy_fixes = validation["fuzzy_corrections"]

    # Compute scores from predicted tokens
    pred_computed = compute_from_tokens(pred)

    # Compute scores from golden tokens
    golden_computed = compute_from_tokens(golden)

    # Per-field token comparison
    field_matches = {}
    for field in ("loc", "role", "tech", "comp"):
        field_matches[f"{field}_match"] = pred[field] == golden[field]
        field_matches[f"pred_{field}"] = pred[field]
        field_matches[f"golden_{field}"] = golden[field]

    return {
        "valid": True,
        "fuzzy_fixes": fuzzy_fixes,
        # Token-level accuracy
        **field_matches,
        # Computed label accuracy
        "label_match": pred_computed["label"] == golden_computed["label"],
        "score_match": pred_computed["score"] == golden_computed["score"],
        "pred_label": pred_computed["label"],
        "golden_label": golden_computed["label"],
        "pred_score": pred_computed["score"],
        "golden_score": golden_computed["score"],
        # Reasoning (for analysis)
        "pred_reasoning": pred.get("reasoning", ""),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--job", type=str, default=None,
                        help="Run specific job(s) by line number (1-indexed)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    # Output file setup
    # When an adapter is given, organise under eval_results/{adapter_folder}/
    # and include the checkpoint name so multiple runs never overwrite each other.
    date_str = datetime.date.today().isoformat()
    test_stem = Path(args.test_file).stem
    prompt_stem = Path(args.prompt).stem
    model_stem = args.model.split("/")[-1]

    if args.adapter:
        p = Path(args.adapter)
        adapter_folder = p.parent.name if (p.is_file() or p.suffix == ".safetensors") else p.name
        checkpoint_stem = p.stem if p.is_file() and p.suffix == ".safetensors" else "final"
        # Strip trailing _adapters suffix for readability: 0000300_adapters → 0000300
        checkpoint_stem = checkpoint_stem.replace("_adapters", "")
        output_dir = Path(args.output_dir) / adapter_folder
        output_file = output_dir / f"{date_str}_{test_stem}_{prompt_stem}_{checkpoint_stem}.txt"
    else:
        adapter_folder = "baseline"
        checkpoint_stem = model_stem
        output_dir = Path(args.output_dir) / "baseline"
        output_file = output_dir / f"{date_str}_{test_stem}_{prompt_stem}_{model_stem}.txt"

    output_dir.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_suffix(".txt.partial")
    log_fh = None
    if not args.job:
        log_fh = open(partial_file, "w")
        sys.stdout = Tee(sys.__stdout__, log_fh)
    print(f"Output: {output_file}")

    # Load prompt
    with open(args.prompt) as f:
        prompt_template = f.read()
    print(f"Prompt: {args.prompt}")

    # Load model
    adapter_path = None
    if args.adapter:
        p = Path(args.adapter)
        if p.is_file() and p.name != "adapters.safetensors":
            # Specific checkpoint file — copy to adapters.safetensors in the same
            # directory, because mlx_lm.load() always loads that fixed filename.
            shutil.copy2(p, p.parent / "adapters.safetensors")
            adapter_path = str(p.parent)
            print(f"Adapter: {args.adapter} → {adapter_path}/adapters.safetensors")
        else:
            adapter_path = str(p.parent) if p.is_file() else str(p)
            print(f"Adapter: {adapter_path}")
    else:
        print("No adapter (baseline)")
    print(f"Model: {args.model}")
    print("─" * 60)

    model, tokenizer = load(args.model, adapter_path=adapter_path)
    system_msg = "Respond with JSON only."
    print("─" * 60)

    # Load test examples
    all_examples = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

    # Filter by --job
    job_indices = None
    if args.job:
        job_indices = set()
        for part in args.job.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                job_indices.update(range(int(start), int(end) + 1))
            else:
                job_indices.add(int(part))
        args.verbose = True

    if job_indices:
        test_examples = [(i, all_examples[i - 1]) for i in sorted(job_indices)]
    else:
        test_examples = [(i + 1, job) for i, job in enumerate(all_examples)]

    print(f"Running V5 eval on {len(test_examples)} jobs...\n")

    results = []
    predictions = []
    parse_failures = 0
    invalid_tokens = 0
    fuzzy_total = 0

    for seq, (orig_idx, job) in enumerate(test_examples, 1):
        prompt_text = (prompt_template
            .replace("{{job_title}}", job["title"])
            .replace("{{job_location}}", job.get("location", ""))
            .replace("{{jd_text}}", job["jd_text"]))

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        t0 = time.time()
        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=MAX_TOKENS,
            verbose=False,
            prefill_step_size=4096,
        )
        elapsed = time.time() - t0

        parsed = parse_json_output(response)

        if parsed is None:
            parse_failures += 1
            print(f"[{orig_idx:3d}/{len(all_examples)}] !  {elapsed:4.1f}s  "
                  f"{job['title'][:42]:<42}  PARSE FAIL")
            if args.verbose:
                print(f"           Raw: {response[:300]}")
            results.append({"label_match": False, "parse_fail": True,
                           "golden_label": job.get("label", "?")})
            continue

        # Score with V5 semantic token system
        scored = score_v5_result(parsed, job)

        if not scored["valid"]:
            invalid_tokens += 1
            scored["parse_fail"] = False
            scored["invalid_token"] = True
            results.append(scored)
            print(f"[{orig_idx:3d}/{len(all_examples)}] !  {elapsed:4.1f}s  "
                  f"{job['title'][:42]:<42}  INVALID TOKEN: {scored['errors']}")
            continue

        if scored.get("fuzzy_fixes"):
            fuzzy_total += 1

        scored["parse_fail"] = False
        scored["invalid_token"] = False
        results.append(scored)

        if args.save_predictions:
            predictions.append({
                "job_index": orig_idx,
                "title": job["title"],
                "golden_tokens": {f: job.get(f) for f in ("loc", "role", "tech", "comp")},
                "pred_tokens": {f: scored.get(f"pred_{f}") for f in ("loc", "role", "tech", "comp")},
                "golden_label": scored["golden_label"],
                "pred_label": scored["pred_label"],
                "golden_score": scored["golden_score"],
                "pred_score": scored["pred_score"],
                "reasoning": scored.get("pred_reasoning", ""),
            })

        status = "✓" if scored["label_match"] else "✗"
        correct_so_far = sum(r["label_match"] for r in results)
        running_pct = correct_so_far / seq * 100

        n_total = len(all_examples)
        label_col = (f"{scored['golden_label']}→{scored['pred_label']}"
                     if not scored["label_match"] else scored["golden_label"])

        prefix = f"[{orig_idx:3d}/{n_total}] {status}  {elapsed:4.1f}s  "
        print(f"{prefix}{job['title'][:42]:<42}  {label_col:<20}  {running_pct:3.0f}%")

        if not scored["label_match"] or args.verbose:
            indent = " " * len(prefix)
            parts = []
            for field in ("loc", "role", "tech", "comp"):
                g = scored.get(f"golden_{field}", "?")
                p = scored.get(f"pred_{field}", "?")
                icon = "✓" if scored.get(f"{field}_match") else "✗"
                parts.append(f"{field} {g}→{p} {icon}")
            print(f"{indent}{' │ '.join(parts)}")
            reasoning = scored.get("pred_reasoning", "")
            if reasoning:
                print(f"{indent}{reasoning[:120]}")
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    valid = [r for r in results if not r.get("parse_fail") and not r.get("invalid_token")]

    label_correct = sum(r["label_match"] for r in valid)
    score_correct = sum(r.get("score_match", False) for r in valid)

    W = 68
    run_id = f"{adapter_folder}  iter {checkpoint_stem}"
    print("\n" + "═" * W)
    print(f"  EVAL  {run_id}")
    print(f"  eval set: {test_stem}  │  prompt: {prompt_stem}  │  n={n}")
    print("═" * W)

    # Quality
    issues = parse_failures + invalid_tokens
    quality = "clean" if issues == 0 else f"{issues} issue(s)"
    print(f"  Parse failures: {parse_failures}   Invalid tokens: {invalid_tokens}   "
          f"Fuzzy corrections: {fuzzy_total}   Valid: {len(valid)}/{n}")
    print()

    if valid:
        pct = label_correct / len(valid) * 100
        score_pct = score_correct / len(valid) * 100
        target_met = pct >= 90.0

        # Per-label accuracy (computed early — used in headline and quick-copy)
        lp_by_label = {}
        for lbl in ("good_fit", "maybe", "bad_fit"):
            lbl_r = [r for r in valid if r["golden_label"] == lbl]
            lp_by_label[lbl] = (sum(r["label_match"] for r in lbl_r) / len(lbl_r) * 100) if lbl_r else 0.0
        macro_fit = sum(lp_by_label.values()) / len(lp_by_label)
        macro_met = macro_fit >= 90.0

        # ── Accuracy headline ────────────────────────────────────────────────
        status = "✅ MET" if target_met else "❌ NOT MET"
        print(f"  Label accuracy:  {label_correct}/{len(valid)} = {pct:.1f}%   {status}  (target ≥90%)")
        print(f"  Score accuracy:  {score_correct}/{len(valid)} = {score_pct:.1f}%")
        print()
        macro_status = "✅" if macro_met else "❌"
        print(f"  Fit accuracy (macro):  {macro_fit:.1f}%  {macro_status}   "
              f"gf={lp_by_label['good_fit']:.0f}%  "
              f"maybe={lp_by_label['maybe']:.0f}%  "
              f"bad={lp_by_label['bad_fit']:.0f}%")
        print()

        # ── Field token accuracy ─────────────────────────────────────────────
        field_pcts = {}
        print("  Field token accuracy:")
        for field in ("loc", "role", "tech", "comp"):
            key = f"{field}_match"
            matched = [r[key] for r in valid if key in r]
            correct = sum(matched)
            fp = correct / len(matched) * 100 if matched else 0.0
            field_pcts[field] = fp
            bar_filled = int(fp / 5)  # 20-char bar at 100%
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"    {field:<5} {bar}  {correct:>3}/{len(matched)}  {fp:.1f}%")
        print()

        # ── Per-label breakdown ──────────────────────────────────────────────
        print("  Per-label breakdown:")
        print(f"    {'Label':<10}  {'Correct':>8}  {'Total':>6}  {'Acc':>6}")
        print(f"    {'─'*10}  {'─'*8}  {'─'*6}  {'─'*6}")
        for lbl in ("good_fit", "maybe", "bad_fit"):
            lbl_results = [r for r in valid if r["golden_label"] == lbl]
            if not lbl_results:
                continue
            correct = sum(r["label_match"] for r in lbl_results)
            lp = correct / len(lbl_results) * 100
            flag = "  ⚠" if lp < 60 else ""
            print(f"    {lbl:<10}  {correct:>8}  {len(lbl_results):>6}  {lp:>5.0f}%{flag}")
        print()

        # ── Confusion matrix ─────────────────────────────────────────────────
        labels = ["good_fit", "maybe", "bad_fit"]
        matrix = {g: {p: 0 for p in labels} for g in labels}
        for r in valid:
            g = r["golden_label"]
            p = r["pred_label"]
            if g in matrix and p in matrix[g]:
                matrix[g][p] += 1

        print("  Confusion Matrix (row=golden, col=predicted):")
        col_w = 10
        header = f"    {'':12}" + "".join(f"{l:>{col_w}}" for l in labels)
        print(header)
        for g in labels:
            row = f"    {g:<12}" + "".join(
                f"{'[' + str(matrix[g][p]) + ']':>{col_w}}" if g == labels[labels.index(p)]
                else f"{matrix[g][p]:>{col_w}}"
                for p in labels
            )
            print(row)
        print()

        # ── Field error transitions ──────────────────────────────────────────
        any_transitions = False
        transition_lines = []
        for field in ("loc", "role", "tech", "comp"):
            transitions = {}
            for r in valid:
                g = r.get(f"golden_{field}")
                p = r.get(f"pred_{field}")
                if g and p and g != p:
                    transitions[f"{g}→{p}"] = transitions.get(f"{g}→{p}", 0) + 1
            if transitions:
                any_transitions = True
                top = sorted(transitions.items(), key=lambda x: -x[1])[:4]
                transition_lines.append(f"    {field}: " + "  ".join(f"{k}:{v}" for k, v in top))
        if any_transitions:
            print("  Field error transitions (top per field):")
            for line in transition_lines:
                print(line)
            print()

        # ── Error direction ──────────────────────────────────────────────────
        over_scores = sum(1 for r in valid if not r["label_match"] and
                         r["pred_label"] in ("good_fit", "maybe") and
                         r["golden_label"] == "bad_fit")
        under_scores = sum(1 for r in valid if not r["label_match"] and
                          r["pred_label"] == "bad_fit" and
                          r["golden_label"] in ("good_fit", "maybe"))
        print(f"  Error direction:  over-scoring {over_scores}  │  under-scoring {under_scores}")

        # ── Quick-copy summary line ──────────────────────────────────────────
        print()
        print("─" * W)
        print(
            f"  {adapter_folder}/{checkpoint_stem}"
            f"  label={pct:.1f}%"
            f"  loc={field_pcts['loc']:.1f}%"
            f"  role={field_pcts['role']:.1f}%"
            f"  tech={field_pcts['tech']:.1f}%"
            f"  comp={field_pcts['comp']:.1f}%"
            f"  gf={lp_by_label['good_fit']:.0f}%"
            f"  maybe={lp_by_label['maybe']:.0f}%"
            f"  bad={lp_by_label['bad_fit']:.0f}%"
        )
    print("═" * W)

    # Write predictions
    if args.save_predictions and predictions and not args.job:
        pred_file = output_file.with_suffix(".predictions.jsonl")
        with open(pred_file, "w") as pf:
            for rec in predictions:
                pf.write(json.dumps(rec) + "\n")
        print(f"\nPredictions: {pred_file}")

    # Finalize output
    if log_fh:
        log_fh.close()
        sys.stdout = sys.__stdout__
        if partial_file.exists():
            partial_file.rename(output_file)
        else:
            # Partial file went missing (e.g. moved mid-run) — write output directly
            output_file.write_text(sys.__stdout__.getvalue() if hasattr(sys.__stdout__, 'getvalue') else "")
        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout = sys.__stdout__
        print("\n\nInterrupted.")
        for p in Path("eval_results").rglob("*.partial"):
            p.unlink(missing_ok=True)
        sys.exit(1)

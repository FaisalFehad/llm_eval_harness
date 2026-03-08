#!/usr/bin/env python3
"""
Eval MLX models against held-out test jobs.

Usage:
  source .venv/bin/activate
  python finetune/eval_finetuned.py --adapter finetune/adapters/0000200_adapters.safetensors
  python finetune/eval_finetuned.py  # no adapter = baseline model

  # Different model (e.g. student bake-off):
  python finetune/eval_finetuned.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \\
      --test-file data/finetune/test.jsonl --prompt prompts/scorer_v10.txt --verbose

  # Custom test file (fields: title, location, jd_text, label; loc/role/tech/comp optional):
  python finetune/eval_finetuned.py --adapter finetune/adapters/0000200_adapters.safetensors \\
      --test-file data/sample_10.jsonl

  # Single job by line number (1-indexed):
  python finetune/eval_finetuned.py --adapter finetune/adapters_v2b \\
      --test-file data/new_uk_jobs_golden.jsonl --job 42 --prompt prompts/scorer_v9.txt
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

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

import mlx.core as mx
from mlx_lm import load, generate

# Marker used in *_cached.txt prompts to split static rules from variable job data

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID        = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
DEFAULT_TEST_FILE = "data/finetune/test.jsonl"
MAX_TOKENS      = 300

PROMPT_TEMPLATE = """\
You are a mechanical job-fit scorer. Follow every rule exactly. Score ONLY what is explicitly written. Do NOT infer or assume.

Inputs:
- Job title: {job_title}
- Job location: {job_location}
- Job description: {jd_text}

===== STEP 1: LOCATION (score this FIRST) =====

Use ONLY the job_location field. Ignore any locations mentioned inside jd_text.

  -50 if job_location is outside the UK (e.g. United States, US, USA, India, Germany, California, Bangalore, Sydney, Remote (US), Seattle, Canada)
  +25 if Remote (UK/Global/Worldwide) OR hybrid/on-site in London
  +10 if hybrid/on-site in UK city that is NOT London (e.g. Manchester, Bristol, Edinburgh)
  0   if location is missing or unclear

===== STEP 2: ROLE & SENIORITY (max 25) =====

Search job_title for these EXACT keywords (case-insensitive):
  +25 if title contains ANY of: Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, VP, Snr, Founding
  +15 if title contains ANY of: Full Stack, Full-Stack, Fullstack, Mid-Level, Mid Level, Midlevel, Software Engineer II, Engineer II, SWE II
  0   if NONE of the above keywords appear

Do NOT give points for: Engineer, Developer, Backend, Frontend, Architect, Manager, Product, or any word not listed above.

===== STEP 3: TECH STACK (max 25, points stack, cap 25) =====

Search jd_text requirements/core stack sections ONLY:
  +10 if Node.js or NodeJS is listed as required/core
  +5  if JavaScript or TypeScript is listed as required/core
  +10 if AI/ML/LLM experience is explicitly REQUIRED (e.g. "experience building ML models", "LLM integrations required")

Do NOT award AI points if AI is only in company description, mission, or "nice to have."
Do NOT award points for unlisted tech: React, PostgreSQL, Python, Go, Rust, Java = 0 points each.

===== STEP 4: COMPENSATION (max 25) =====

From jd_text ONLY. Count ONLY base salary explicitly in GBP (£).
Ignore: bonuses, equity, benefits, USD ($), EUR, and all non-GBP amounts.

Converting "k" notation: £100k = £100,000. £80k-£120k = £80,000-£120,000.
- If a range is given (e.g. £80,000-£120,000), use the midpoint.
- "Up to £X" with no lower bound = salary not stated, score 0.

  +25 if midpoint >= £100,000
  +15 if midpoint £75,000-£99,999
  +5  if midpoint £55,000-£74,999
  -30 if midpoint < £45,000
  0   if no GBP salary stated or "up to" with no lower bound

===== WORKED EXAMPLES =====

Example A (good_fit):
Title: "Senior Software Engineer" | Location: "London Area, United Kingdom (Hybrid)" | JD requires Node.js, TypeScript, salary £90,000-£110,000.
→ loc: 25 (London, UK)
→ role: 25 (Senior)
→ tech: 15 (Node.js +10, TypeScript +5)
→ comp: 25 (midpoint £100,000)
→ total: 25+25+15+25 = 90 → good_fit
{{"loc":25,"role":25,"tech":15,"comp":25,"score":90,"label":"good_fit","reasoning":"Senior role in London with Node.js/TS and £100k midpoint salary."}}

Example B (bad_fit):
Title: "Full Stack Engineer" | Location: "United States" | JD mentions React, Node.js, TypeScript, competitive salary.
→ loc: -50 (United States = outside UK)
→ role: 15 (Full Stack)
→ tech: 15 (Node.js +10, TypeScript +5; React = 0)
→ comp: 0 (no GBP salary)
→ total: max(0, -50+15+15+0) = 0 → bad_fit
{{"loc":-50,"role":15,"tech":15,"comp":0,"score":0,"label":"bad_fit","reasoning":"US location (-50) makes total negative, floored to 0."}}

===== OUTPUT FORMAT =====

Score each category independently, then total = max(0, min(100, loc+role+tech+comp)).

Assign label from total:
  IF total >= 70 THEN label = "good_fit"
  IF total >= 50 AND total <= 69 THEN label = "maybe"
  IF total <= 49 THEN label = "bad_fit"

Return ONLY this JSON:
{{"loc":0,"role":0,"tech":0,"comp":0,"score":0,"label":"bad_fit","reasoning":"brief"}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output, handling extra text."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in output
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def score_result(predicted: dict, golden: dict) -> dict:
    """Compare predicted vs golden, return per-field match info.

    loc/role/tech/comp in golden are optional — if absent, field match is None (skipped).
    """
    def field_match(key):
        g = golden.get(key)
        if g is None:
            return None  # field not present in golden — skip comparison
        return predicted.get(key) == g

    return {
        "label_match":  predicted.get("label") == golden["label"],
        "loc_match":    field_match("loc"),
        "role_match":   field_match("role"),
        "tech_match":   field_match("tech"),
        "comp_match":   field_match("comp"),
        "pred_label":   predicted.get("label", "MISSING"),
        "golden_label": golden["label"],
        "pred_loc":     predicted.get("loc"),
        "pred_role":    predicted.get("role"),
        "pred_tech":    predicted.get("tech"),
        "pred_comp":    predicted.get("comp"),
        "golden_loc":   golden.get("loc"),
        "golden_role":  golden.get("role"),
        "golden_tech":  golden.get("tech"),
        "golden_comp":  golden.get("comp"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to adapter checkpoint (omit for baseline)")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE,
                        help=f"JSONL file with test jobs (default: {DEFAULT_TEST_FILE})")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Path to prompt template .txt file (omit to use built-in default)")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help=f"Model ID or local path (default: {MODEL_ID})")
    parser.add_argument("--job", type=str, default=None,
                        help="Run only specific job(s) by line number (1-indexed). "
                             "Comma-separated or range: --job 42 or --job 3,10,11 or --job 1-5. "
                             "Implies --verbose.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each prediction")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory to save output (default: eval_results/)")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Write per-job predictions to .predictions.jsonl alongside the .txt report")
    args = parser.parse_args()

    # Set up output file — auto-named from test file + prompt + date
    # Writes to a .partial temp file first; renamed to final name only on completion.
    # Single-job debug runs (--job) skip file output entirely.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    test_stem = Path(args.test_file).stem
    prompt_stem = Path(args.prompt).stem if args.prompt else "builtin"
    model_stem = args.model.split("/")[-1]
    output_file = output_dir / f"{date_str}_{test_stem}_{prompt_stem}_{model_stem}.txt"
    partial_file = output_file.with_suffix(".txt.partial")
    log_fh = None
    if not args.job:
        log_fh = open(partial_file, "w")
        sys.stdout = Tee(sys.__stdout__, log_fh)
    print(f"Output: {output_file}")

    # Load prompt template (file overrides built-in default)
    prompt_template = PROMPT_TEMPLATE
    if args.prompt:
        with open(args.prompt) as f:
            prompt_template = f.read()
        print(f"Prompt: {args.prompt}")
    else:
        print("Prompt: built-in default")

    # Resolve adapter path to directory (mlx_lm expects the adapters dir)
    adapter_path = None
    if args.adapter:
        p = Path(args.adapter)
        if p.is_file():
            adapter_path = str(p.parent)
        else:
            adapter_path = str(p)
        print(f"Loading model with adapter: {adapter_path}")
    else:
        print("Loading baseline model (no adapter)")

    print(f"Model: {args.model}")
    print("─" * 60)

    # Load model
    model, tokenizer = load(args.model, adapter_path=adapter_path)

    # Detect Qwen3 by model name or by reading config.json for local paths
    def _is_qwen3(model_id: str) -> bool:
        if "qwen3" in model_id.lower():
            return True
        cfg = Path(model_id) / "config.json"
        if cfg.exists():
            import json as _json
            return "qwen3" in _json.load(open(cfg)).get("model_type", "").lower()
        return False

    system_msg = "/no_think" if _is_qwen3(args.model) else "Respond with JSON only."

    print("─" * 60)

    # Load test examples
    all_examples = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                all_examples.append(json.loads(line))

    # Filter by --job if specified (1-indexed line numbers)
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
        # Validate
        for idx in job_indices:
            if idx < 1 or idx > len(all_examples):
                print(f"Error: --job {idx} out of range (file has {len(all_examples)} jobs)")
                sys.exit(1)
        args.verbose = True  # always verbose for single-job debugging

    # Build (original_index, job) pairs for eval
    if job_indices:
        test_examples = [(i, all_examples[i - 1]) for i in sorted(job_indices)]
        print(f"Running eval on {len(test_examples)} selected job(s) (of {len(all_examples)} total)...\n")
    else:
        test_examples = [(i + 1, job) for i, job in enumerate(all_examples)]
        print(f"Running eval on {len(test_examples)} test examples...\n")

    results = []
    predictions = []  # per-job records for --save-predictions
    parse_failures = 0
    is_debug = job_indices is not None

    for seq, (orig_idx, job) in enumerate(test_examples, 1):
        # File-based prompts use {{variable}} (promptfoo style) — use replace() to
        # avoid conflicts with JSON curly braces in worked examples.
        # Built-in PROMPT_TEMPLATE uses {variable} (Python format style).
        if args.prompt:
            prompt_text = (prompt_template
                .replace("{{job_title}}", job["title"])
                .replace("{{job_location}}", job.get("location", ""))
                .replace("{{jd_text}}", job["jd_text"]))
        else:
            prompt_text = prompt_template.format(
                job_title=job["title"],
                job_location=job.get("location", ""),
                jd_text=job["jd_text"],
            )

        messages = [
            {"role": "system",  "content": system_msg},
            {"role": "user",    "content": prompt_text},
        ]

        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

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
            print(f"           Raw ({len(response)} chars): {response[:500]}")
            results.append({"label_match": False, "parse_fail": True,
                             "golden_label": job["label"]})
            if args.save_predictions:
                predictions.append({
                    "job_index": orig_idx, "title": job["title"],
                    "company": job.get("company", ""), "location": job.get("location", ""),
                    "parse_fail": True,
                    "golden": {"label": job["label"], "loc": job.get("loc"), "role": job.get("role"),
                               "tech": job.get("tech"), "comp": job.get("comp")},
                    "pred": None,
                })
            continue

        # Recompute score and label from the four fields — never trust model arithmetic.
        raw = (parsed.get("loc", 0) + parsed.get("role", 0) +
               parsed.get("tech", 0) + parsed.get("comp", 0))
        parsed["score"] = max(0, min(100, raw))
        parsed["label"] = (
            "good_fit" if parsed["score"] >= 70 else
            "maybe"    if parsed["score"] >= 50 else
            "bad_fit"
        )

        scored = score_result(parsed, job)
        scored["parse_fail"] = False
        results.append(scored)

        if args.save_predictions:
            predictions.append({
                "job_index": orig_idx, "title": job["title"],
                "company": job.get("company", ""), "location": job.get("location", ""),
                "parse_fail": False,
                "golden": {"label": job["label"], "loc": job.get("loc"), "role": job.get("role"),
                           "tech": job.get("tech"), "comp": job.get("comp"),
                           "score": job.get("score")},
                "pred": {"label": parsed.get("label"), "loc": parsed.get("loc"),
                         "role": parsed.get("role"), "tech": parsed.get("tech"),
                         "comp": parsed.get("comp"), "score": parsed.get("score"),
                         "reasoning": parsed.get("reasoning", "")},
            })

        status = "✓" if scored["label_match"] else "✗"
        correct_so_far = sum(r["label_match"] for r in results if not r.get("parse_fail"))
        running_pct = correct_so_far / seq * 100

        # Debug mode: show full detail for each job
        if is_debug:
            verdict = "✓ CORRECT" if scored["label_match"] else "✗ WRONG"
            print(f"{'═' * 70}")
            print(f"  Job #{orig_idx}: {job['title']}")
            print(f"  Company:  {job.get('company', 'N/A')}")
            print(f"  Location: {job.get('location', 'N/A')}")
            print(f"  Verdict:  {verdict}")
            print(f"{'─' * 70}")
            print()
            # Comparison table
            print(f"  {'Field':<8} │ {'Golden':>8} │ {'Model':>8} │ {'Diff':>6} │ {'':>5}")
            print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*5}")
            for field in ["loc", "role", "tech", "comp"]:
                g = job.get(field)
                p = parsed.get(field)
                if g is not None and p is not None:
                    diff = p - g
                    diff_str = f"{diff:+d}" if diff != 0 else "—"
                    icon = "✓" if diff == 0 else "✗"
                elif g is None:
                    diff_str = "?"
                    icon = "?"
                else:
                    diff_str = "?"
                    icon = "✗"
                g_str = str(g) if g is not None else "N/A"
                p_str = str(p) if p is not None else "N/A"
                print(f"  {field:<8} │ {g_str:>8} │ {p_str:>8} │ {diff_str:>6} │ {icon:>5}")
            # Score and label rows
            g_score = job.get("score", "N/A")
            p_score = parsed.get("score", "N/A")
            print(f"  {'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*5}")
            print(f"  {'score':<8} │ {str(g_score):>8} │ {str(p_score):>8} │ {'':>6} │ {'':>5}")
            label_icon = "✓" if scored["label_match"] else "✗"
            print(f"  {'label':<8} │ {job['label']:>8} │ {parsed.get('label', 'N/A'):>8} │ {'':>6} │ {label_icon:>5}")
            print()
            # Model reasoning
            print(f"  Model reasoning: {parsed.get('reasoning', 'N/A')}")
            print()
            # Raw model output
            print(f"  Raw model output:")
            print(f"  {response.strip()}")
            print()
            continue

        # ── Per-job output ────────────────────────────────────────────────────
        n_total = len(all_examples)
        label_col = (f"{job['label']}→{scored['pred_label']}"
                     if not scored["label_match"]
                     else job["label"])

        prefix = f"[{orig_idx:3d}/{n_total}] {status}  {elapsed:4.1f}s  "
        print(f"{prefix}{job['title'][:42]:<42}  {label_col:<20}  {running_pct:3.0f}%")

        # On errors (or verbose): show field diff + reasoning, indented under title
        if not scored["label_match"] or args.verbose:
            indent = " " * len(prefix)
            field_parts = []
            for field in ["loc", "role", "tech", "comp"]:
                g = job.get(field)
                p = parsed.get(field)
                g_str = f"{g:>3}" if g is not None else " ??"
                p_str = f"{p:>3}" if p is not None else " ??"
                icon = ("✓" if g == p else "✗") if g is not None else "?"
                field_parts.append(f"{field} {g_str}→{p_str} {icon}")
            print(f"{indent}{' │ '.join(field_parts)}")
            reasoning = parsed.get("reasoning", "")
            if reasoning:
                print(f"{indent}{reasoning[:100]}")
            print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    valid = [r for r in results if not r.get("parse_fail")]

    label_correct = sum(r["label_match"] for r in valid)

    def field_accuracy(key):
        """Return (correct, total) for a field, skipping None (missing golden)."""
        scored = [r[key] for r in valid if r[key] is not None]
        return sum(scored), len(scored)

    print("\n" + "═" * 60)
    print("RESULTS")
    print("═" * 60)
    print(f"Total examples:  {n}")
    print(f"Parse failures:  {parse_failures}")
    print(f"Valid outputs:   {len(valid)}")
    print()
    pct = label_correct / len(valid) * 100 if valid else 0.0
    print(f"Label accuracy:  {label_correct}/{len(valid)} = {pct:.1f}%  "
          f"(baseline: 39.8%)")
    print()

    loc_c, loc_n   = field_accuracy("loc_match")
    role_c, role_n = field_accuracy("role_match")
    tech_c, tech_n = field_accuracy("tech_match")
    comp_c, comp_n = field_accuracy("comp_match")

    if loc_n > 0:
        print("Field accuracy:")
        print(f"  loc:   {loc_c}/{loc_n} = {loc_c/loc_n*100:.1f}%")
        print(f"  role:  {role_c}/{role_n} = {role_c/role_n*100:.1f}%")
        print(f"  tech:  {tech_c}/{tech_n} = {tech_c/tech_n*100:.1f}%")
        print(f"  comp:  {comp_c}/{comp_n} = {comp_c/comp_n*100:.1f}%")
    else:
        print("Field accuracy: (no component scores in golden — label-only eval)")
    print()

    # Per-label breakdown
    labels = ["good_fit", "maybe", "bad_fit"]
    print("Per-label breakdown:")
    for lbl in labels:
        lbl_results = [r for r in valid if r["golden_label"] == lbl]
        if not lbl_results:
            continue
        correct = sum(r["label_match"] for r in lbl_results)
        print(f"  {lbl:<10}: {correct}/{len(lbl_results)} = {correct/len(lbl_results)*100:.0f}%")

    # Error pattern analysis — systematic vs random errors
    over_scores = sum(1 for r in valid if not r["label_match"] and
                      r["pred_label"] in ("good_fit", "maybe") and
                      r["golden_label"] == "bad_fit")
    under_scores = sum(1 for r in valid if not r["label_match"] and
                       r["pred_label"] == "bad_fit" and
                       r["golden_label"] in ("good_fit", "maybe"))
    other_errors = sum(1 for r in valid if not r["label_match"]) - over_scores - under_scores

    print(f"\nError patterns:")
    print(f"  Over-scoring:   {over_scores}  (predicted higher than golden)")
    print(f"  Under-scoring:  {under_scores}  (predicted lower than golden)")
    print(f"  Other:          {other_errors}  (e.g. good_fit↔maybe)")
    print(f"  Parse failures: {parse_failures}")

    # Write predictions JSONL if requested
    if args.save_predictions and predictions and not args.job:
        pred_file = output_file.with_suffix(".predictions.jsonl")
        with open(pred_file, "w") as pf:
            for rec in predictions:
                pf.write(json.dumps(rec) + "\n")
        print(f"\nPredictions: {pred_file} ({len(predictions)} records)")

    # Finalize: rename .partial → .txt (only for full runs, not --job debug)
    if log_fh:
        log_fh.close()
        sys.stdout = sys.__stdout__
        partial_file.rename(output_file)
        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Clean up partial file on Ctrl+C
        sys.stdout = sys.__stdout__
        print("\n\nInterrupted — deleting partial output file.")
        for p in Path("eval_results").glob("*.partial"):
            p.unlink(missing_ok=True)
        sys.exit(1)

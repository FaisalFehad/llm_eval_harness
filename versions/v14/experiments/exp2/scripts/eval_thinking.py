#!/usr/bin/env python3
"""
V14 Experiment 2 — Thinking ON with all parse repairs (copy of eval_student_v7.py).

Changes from base (see docs/V14_EXPERIMENTS_PLAN.md for full rationale):
  A  Fix {{ pre-fill bug: never pre-fill '{' for Qwen3 (prompt already says "Begin with {")
  B  System msg: add double-quote directive
  C  System msg: add _raw conciseness hint (10 words max)
  D  Parser: single-quoted key/token repair ('key': → "key":, 'TOKEN' → "TOKEN")
  E  Parser: trailing comma removal (,} → }, ,] → ])
  F  Parser: Python literals → JSON (None→null, True→true, False→false)
  G  Parser: multi-close truncation (try }, "}, "]}', ]}, not just })
  H  Logic: detect unclosed <think> tag → retry with 3x tokens
  I  Logic: retry any parse fail once at 2x tokens
  J  Logic: dynamic max_tokens = max(600, min(2000, 400 + len(jd_text)//6))
  K  Metrics: true_accuracy, avg_speed_s, total_time_s, retry stats in summary

Run WITHOUT --no-think (thinking ON). No --max-tokens (dynamic per job).
DO NOT MODIFY eval_student_v7.py — this is the experiment copy.

Original docstring:
Eval MLX models against V7 semantic token eval set.

The V7 student model outputs 5 semantic token fields:
  loc, arr, sen, tech, comp
This script:
  1. Runs inference with the V7 student prompt
  2. Validates output tokens (with fuzzy matching for typos)
  3. Computes scores/labels via the code layer (tech=["OOS"] gates seniority)
  4. Compares tokens AND computed labels against golden set

Tech is an array of individual tokens: ["NODE", "REACT", "JS_TS", "AI_ML"] or ["OOS"].
Scores are backward-compatible with V6:
  loc_score, role_score, tech_score, comp_score

Usage:
  source .venv/bin/activate

  # Student model (V7 semantic tokens):
  python3 finetune/eval_student_v7.py \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --adapter finetune/adapters_v7 \
      --test-file data/v7/eval_golden.jsonl \
      --verbose

  # Baseline (no adapter):
  python3 finetune/eval_student_v7.py \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --test-file data/v7/eval_golden.jsonl
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

# Import V7 semantic token definitions
sys.path.insert(0, str(Path(__file__).parent))
from semantic_tokens_v7 import (
    SCALAR_FIELD_TOKENS,
    validate_prediction, compute_from_tokens, fuzzy_match,
    V7_TOKEN_FIELDS, V7_RAW_FIELDS,
)


class Tee:
    """Write to multiple streams simultaneously (terminal + file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


import shutil

import mlx.core as mx
from mlx_lm import load, generate


def greedy_sampler(logits: mx.array) -> mx.array:
    """Greedy decoding — always pick the highest probability token."""
    return mx.argmax(logits, axis=-1)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_TEST_FILE = "data/v7/eval_golden.jsonl"
DEFAULT_PROMPT = "prompts/student_v7.txt"
MAX_TOKENS = 1000

# V7 token fields — 5 fields
TOKEN_FIELDS = ("loc", "arr", "sen", "tech", "comp")
# The 4 fields that contribute to scoring
SCORE_FIELDS = ("loc", "sen", "tech", "comp")
# V7 field to V6-compatible display name
V7_FIELD_TO_V6_NAME = {
    "loc": "loc",
    "sen": "role",
    "tech": "tech",
    "comp": "comp",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output — EXP2 extended repair stack (A-G).

    Repair layers applied in order before each parse attempt:
      F: Python literals (None/True/False → null/true/false)
      E: Trailing comma removal (,} → }, ,] → ])
      D: Single-quoted keys/tokens → double-quoted
      G: Multi-close truncation recovery
    """
    text = text.strip()

    # Strip markdown fences
    text = text.replace("```json", "").replace("```", "").strip()

    # [F] Python literals → JSON (only when appearing as values after ':')
    text = re.sub(r':\s*None([,}\]\s])', lambda m: ': null' + m.group(1), text)
    text = re.sub(r':\s*True([,}\]\s])', lambda m: ': true' + m.group(1), text)
    text = re.sub(r':\s*False([,}\]\s])', lambda m: ': false' + m.group(1), text)

    # [E] Trailing comma removal
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Fix missing opening quote on first key (original repairs)
    # Pattern 1: {loc_raw": -> {"loc_raw":  (missing opening quote only)
    text = re.sub(r'\{(\s*)(\w+)":', r'{"\2":', text)
    # Pattern 2: {loc_raw: -> {"loc_raw":  (fully unquoted key)
    text = re.sub(r'\{(\s*)(\w+):', r'{"\2":', text)
    # Pattern 3: ,loc_raw: -> ,"loc_raw":  (unquoted keys after comma)
    text = re.sub(r',(\s*)(\w+):', r',"\2":', text)

    # [D] Single-quoted keys → double-quoted ('loc_raw': → "loc_raw":)
    # Safe: JSON keys in this schema are always simple word identifiers
    text = re.sub(r"'(\w+)'(?=\s*:)", r'"\1"', text)
    # [D] Single-quoted uppercase tokens → double-quoted ('REMOTE' → "REMOTE")
    # Safe: valid tokens are always uppercase identifiers — never contain apostrophes
    text = re.sub(r"'([A-Z_0-9]+)'", r'"\1"', text)

    # Attempt 1: Direct load
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Regex extraction (complete JSON object)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # [G] Multi-close truncation recovery
    match = re.search(r'\{[\s\S]*', text)
    if match:
        truncated = match.group()
        # Close open strings first
        if truncated.count('"') % 2 != 0:
            truncated += '"'
        # Try closings in order: most to least specific
        for closing in ['"]', '"]}', ']}', '"}', '}']:
            try:
                return json.loads(truncated + closing)
            except json.JSONDecodeError:
                pass
        # Original tech-aware closure as final fallback
        if '"tech"' in truncated and '[' in truncated.split('"tech"')[-1]:
            after_tech = truncated.split('"tech"')[-1]
            if after_tech.count('[') > after_tech.count(']'):
                truncated += ']'
        try:
            return json.loads(truncated + '}')
        except json.JSONDecodeError:
            pass

    return None


def compare_tech_arrays(pred_tech, golden_tech) -> bool:
    """Compare tech arrays (order-independent)."""
    if isinstance(pred_tech, str):
        pred_tech = [pred_tech]
    if isinstance(golden_tech, str):
        golden_tech = [golden_tech]
    return sorted(pred_tech) == sorted(golden_tech)


def score_v7_result(predicted: dict, golden: dict) -> dict:
    """Compare predicted vs golden V7 semantic tokens and computed labels."""

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

    # Per-field token comparison (all 5 V7 fields)
    field_matches = {}
    for field in TOKEN_FIELDS:
        if field == "tech":
            # Tech is an array — compare order-independently
            field_matches[f"{field}_match"] = compare_tech_arrays(
                pred.get(field, []), golden.get(field, []))
        else:
            field_matches[f"{field}_match"] = pred.get(field) == golden.get(field)
        field_matches[f"pred_{field}"] = pred.get(field)
        field_matches[f"golden_{field}"] = golden.get(field)

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
        # Per-field computed scores (V6-compatible names)
        "pred_loc_score": pred_computed["loc_score"],
        "pred_role_score": pred_computed["role_score"],
        "pred_tech_score": pred_computed["tech_score"],
        "pred_comp_score": pred_computed["comp_score"],
        # Raw fields (for analysis)
        **{f"pred_{f}": pred.get(f, "") for f in V7_RAW_FIELDS},
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
    parser.add_argument("--system-msg", type=str, default=None,
                        help="Custom system message (default: 'Respond with JSON only.')")
    parser.add_argument("--preprocess", action="store_true",
                        help="Apply JD preprocessing (V12: must match training)")
    parser.add_argument("--no-think", action="store_true",
                        help="Qwen3: disable thinking mode (enable_thinking=False). "
                             "Faster, eliminates thinking-overflow parse failures.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_TOKENS}). "
                             "Increase to 3000+ to give thinking mode more budget.")
    args = parser.parse_args()

    # Output file setup — timestamped for history (never overwrites)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    test_stem = Path(args.test_file).stem
    prompt_stem = Path(args.prompt).stem
    model_stem = args.model.split("/")[-1]

    if args.adapter:
        p = Path(args.adapter)
        adapter_folder = p.parent.name if (p.is_file() or p.suffix == ".safetensors") else p.name
        checkpoint_stem = p.stem if p.is_file() and p.suffix == ".safetensors" else "final"
        checkpoint_stem = checkpoint_stem.replace("_adapters", "")
        output_dir = Path(args.output_dir) / adapter_folder
        output_file = output_dir / f"{timestamp}_{test_stem}_{prompt_stem}_{checkpoint_stem}.txt"
    else:
        adapter_folder = "baseline"
        checkpoint_stem = model_stem
        output_dir = Path(args.output_dir) / "baseline"
        output_file = output_dir / f"{timestamp}_{test_stem}_{prompt_stem}_{model_stem}.txt"

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
            shutil.copy2(p, p.parent / "adapters.safetensors")
            adapter_path = str(p.parent)
            print(f"Adapter: {args.adapter} -> {adapter_path}/adapters.safetensors")
        else:
            adapter_path = str(p.parent) if p.is_file() else str(p)
            print(f"Adapter: {adapter_path}")
    else:
        print("No adapter (baseline)")
    print(f"Model: {args.model}")
    print("-" * 60)

    model, tokenizer = load(args.model, adapter_path=adapter_path)
    # [B, C] Extended system message: double-quote directive + _raw conciseness hint
    # These are system-level formatting instructions — safe to add without affecting
    # the fine-tuned user/assistant turn distribution.
    system_msg = (args.system_msg or
        "Respond with JSON only. Use double-quoted strings throughout your JSON. "
        "Keep _raw fields concise — copy only the most relevant phrase (10 words max).")
    print("-" * 60)

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

    print(f"Running V7 eval on {len(test_examples)} jobs...\n")

    results = []
    predictions = []
    parse_failures = 0
    invalid_tokens = 0
    fuzzy_total = 0
    total_gen_time = 0.0    # EXP2 [K]: cumulative generation time
    retries_attempted = 0   # EXP2 [I]: how many jobs triggered a retry
    retries_succeeded = 0   # EXP2 [I]: how many retries rescued a parse fail
    unclosed_think_count = 0  # EXP2 [H]: jobs where <think> was unclosed

    # Optional JD preprocessing (V12 train/inference consistency)
    preprocess_fn = None
    if args.preprocess:
        from preprocess_jd import preprocess_jd
        preprocess_fn = preprocess_jd
        print("JD preprocessing: ENABLED")

    for seq, (orig_idx, job) in enumerate(test_examples, 1):
        # V7 golden data stores raw location as job_location
        raw_location = job.get("job_location", job.get("location", ""))
        jd_text = job["jd_text"]
        if preprocess_fn:
            jd_text = preprocess_fn(jd_text)
        prompt_text = (prompt_template
            .replace("{{job_title}}", job["title"])
            .replace("{{job_location}}", raw_location)
            .replace("{{jd_text}}", jd_text))

        is_qwen3 = "qwen3" in args.model.lower()

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]

        # [J] Dynamic max_tokens: scale with JD length
        # Base 400 (thinking budget) + 1 token per 6 chars of JD text
        # Floor 600 (short JDs still need thinking room), ceiling 2000
        dynamic_max = max(600, min(2000, 400 + len(jd_text) // 6))

        no_think = args.no_think if is_qwen3 else False
        chat_kwargs = {"enable_thinking": False} if no_think else {}
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)

        # [A] Never pre-fill '{' for Qwen3 — prompt already says "Begin with {"
        # Pre-filling causes '{{' double-brace → guaranteed parse fail.
        if not is_qwen3:
            formatted += "{"

        t0 = time.time()
        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=dynamic_max,
            verbose=False,
            prefill_step_size=4096,
            sampler=greedy_sampler,
        )

        # [H] Detect unclosed <think>: model ran out of tokens mid-thinking
        # No </think> means no JSON was produced — retry with 3x budget
        if is_qwen3 and "<think>" in response and "</think>" not in response:
            unclosed_think_count += 1
            retry_max = min(dynamic_max * 3, 4000)
            print(f"  [UNCLOSED THINK → retry at {retry_max} tokens]")
            response = generate(
                model, tokenizer,
                prompt=formatted,
                max_tokens=retry_max,
                verbose=False,
                prefill_step_size=4096,
                sampler=greedy_sampler,
            )

        elapsed = time.time() - t0
        total_gen_time += elapsed

        if not is_qwen3:
            response = "{" + response
        else:
            # Strip <think>...</think> tags from Qwen3 output
            response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)

        parsed = parse_json_output(response)

        # [I] Retry on parse fail — double the token budget, one attempt
        if parsed is None:
            retries_attempted += 1
            retry_max = min(dynamic_max * 2, 3000)
            print(f"  [PARSE FAIL → retry at {retry_max} tokens]")
            t_retry = time.time()
            response2 = generate(
                model, tokenizer,
                prompt=formatted,
                max_tokens=retry_max,
                verbose=False,
                prefill_step_size=4096,
                sampler=greedy_sampler,
            )
            if is_qwen3:
                response2 = re.sub(r"<think>.*?</think>\s*", "", response2, flags=re.DOTALL)
            parsed = parse_json_output(response2)
            total_gen_time += time.time() - t_retry
            if parsed is not None:
                retries_succeeded += 1
                print(f"  [RETRY SUCCESS]")
                response = response2  # use retry response for raw output if needed

        if parsed is None:
            parse_failures += 1
            print(f"[{orig_idx:3d}/{len(all_examples)}] !  {elapsed:4.1f}s  "
                  f"{job['title'][:42]:<42}  PARSE FAIL")
            print(f"           Raw: {response[:300]}")
            results.append({"label_match": False, "parse_fail": True,
                           "golden_label": job.get("label", "?")})
            predictions.append({
                "job_index": orig_idx, "title": job["title"],
                "parse_fail": True, "raw_output": response[:500],
            })
            continue

        # Score with V7 semantic token system
        scored = score_v7_result(parsed, job)

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

        predictions.append({
            "job_index": orig_idx,
            "title": job["title"],
            "company": job.get("company", ""),
            "location": job.get("job_location", job.get("location", "")),
            "golden_tokens": {f: job.get(f) for f in TOKEN_FIELDS},
            "pred_tokens": {f: scored.get(f"pred_{f}") for f in TOKEN_FIELDS},
            "golden_label": scored["golden_label"],
            "pred_label": scored["pred_label"],
            "golden_score": scored["golden_score"],
            "pred_score": scored["pred_score"],
            "label_match": scored["label_match"],
            **{f"pred_{f}": scored.get(f"pred_{f}", "") for f in V7_RAW_FIELDS},
        })

        status = "\u2713" if scored["label_match"] else "\u2717"
        correct_so_far = sum(r["label_match"] for r in results)
        running_pct = correct_so_far / seq * 100

        n_total = len(all_examples)
        label_col = (f"{scored['golden_label']}->{scored['pred_label']}"
                     if not scored["label_match"] else scored["golden_label"])

        prefix = f"[{orig_idx:3d}/{n_total}] {status}  {elapsed:4.1f}s  "
        print(f"{prefix}{job['title'][:42]:<42}  {label_col:<20}  {running_pct:3.0f}%")

        if not scored["label_match"] or args.verbose:
            indent = " " * len(prefix)
            parts = []
            for field in TOKEN_FIELDS:
                g = scored.get(f"golden_{field}", "?")
                p = scored.get(f"pred_{field}", "?")
                icon = "\u2713" if scored.get(f"{field}_match") else "\u2717"
                # Format tech arrays for display
                if field == "tech":
                    g_str = ",".join(g) if isinstance(g, list) else str(g)
                    p_str = ",".join(p) if isinstance(p, list) else str(p)
                    parts.append(f"{field} {g_str}->{p_str} {icon}")
                else:
                    parts.append(f"{field} {g}->{p} {icon}")
            print(f"{indent}{' | '.join(parts)}")
            # Show raw fields
            raws = [scored.get(f"pred_{f}", "") for f in V7_RAW_FIELDS]
            raw_str = " | ".join(r for r in raws if r)
            if raw_str:
                print(f"{indent}{raw_str[:180]}")
            print()

        # [K] Free GPU memory between jobs to reduce inter-job latency spikes
        mx.clear_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    valid = [r for r in results if not r.get("parse_fail") and not r.get("invalid_token")]

    label_correct = sum(r["label_match"] for r in valid)
    score_correct = sum(r.get("score_match", False) for r in valid)

    W = 72
    run_id = f"{adapter_folder}  iter {checkpoint_stem}"
    print("\n" + "=" * W)
    print(f"  V7 EVAL  {run_id}")
    print(f"  eval set: {test_stem}  |  prompt: {prompt_stem}  |  n={n}")
    print("=" * W)

    # Quality
    issues = parse_failures + invalid_tokens
    print(f"  Parse failures: {parse_failures}   Invalid tokens: {invalid_tokens}   "
          f"Fuzzy corrections: {fuzzy_total}   Valid: {len(valid)}/{n}")
    print()

    # [K] Speed + retry summary
    avg_speed = total_gen_time / n if n > 0 else 0.0
    print(f"  Speed:  total={total_gen_time/60:.1f}min  avg={avg_speed:.1f}s/job  hardware=Mac M1 MLX 6-bit")
    print(f"  Retries: attempted={retries_attempted}  succeeded={retries_succeeded}  unclosed_think={unclosed_think_count}\n")

    if valid:
        pct = label_correct / len(valid) * 100
        score_pct = score_correct / len(valid) * 100
        target_met = pct >= 90.0
        # [K] true_accuracy: parse fails counted as wrong (fair cross-experiment metric)
        true_acc = label_correct / n * 100

        # Per-label accuracy
        lp_by_label = {}
        for lbl in ("good_fit", "maybe", "bad_fit"):
            lbl_r = [r for r in valid if r["golden_label"] == lbl]
            lp_by_label[lbl] = (sum(r["label_match"] for r in lbl_r) / len(lbl_r) * 100) if lbl_r else 0.0
        macro_fit = sum(lp_by_label.values()) / len(lp_by_label)

        # ── Accuracy headline ────────────────────────────────────────────────
        status_icon = "MET" if target_met else "NOT MET"
        print(f"  Label accuracy:  {label_correct}/{len(valid)} = {pct:.1f}%   {status_icon}  (target >=90%)")
        print(f"  True accuracy:   {label_correct}/{n} = {true_acc:.1f}%  (parse fails as wrong — use for cross-experiment comparison)")
        print(f"  Score accuracy:  {score_correct}/{len(valid)} = {score_pct:.1f}%")
        print()
        macro_icon = "MET" if macro_fit >= 90 else "NOT MET"
        print(f"  Fit accuracy (macro):  {macro_fit:.1f}%  {macro_icon}   "
              f"gf={lp_by_label['good_fit']:.0f}%  "
              f"maybe={lp_by_label['maybe']:.0f}%  "
              f"bad={lp_by_label['bad_fit']:.0f}%")
        print()

        # ── Field token accuracy (all 5 V7 fields) ────────────────────────
        field_pcts = {}
        print("  Field token accuracy:")
        for field in TOKEN_FIELDS:
            key = f"{field}_match"
            matched = [r[key] for r in valid if key in r]
            correct = sum(matched)
            fp = correct / len(matched) * 100 if matched else 0.0
            field_pcts[field] = fp
            bar_filled = int(fp / 5)  # 20-char bar at 100%
            bar = "#" * bar_filled + "." * (20 - bar_filled)
            print(f"    {field:<14} {bar}  {correct:>3}/{len(matched)}  {fp:.1f}%")
        print()

        # ── Per-label breakdown ──────────────────────────────────────────────
        print("  Per-label breakdown:")
        print(f"    {'Label':<10}  {'Correct':>8}  {'Total':>6}  {'Acc':>6}")
        print(f"    {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}")
        for lbl in ("good_fit", "maybe", "bad_fit"):
            lbl_results = [r for r in valid if r["golden_label"] == lbl]
            if not lbl_results:
                continue
            correct = sum(r["label_match"] for r in lbl_results)
            lp = correct / len(lbl_results) * 100
            flag = "  !" if lp < 60 else ""
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
                f"{'[' + str(matrix[g][p]) + ']':>{col_w}}" if g == p
                else f"{matrix[g][p]:>{col_w}}"
                for p in labels
            )
            print(row)
        print()

        # ── Field error transitions (all 5 fields) ──────────────────────────
        any_transitions = False
        transition_lines = []
        for field in TOKEN_FIELDS:
            transitions = {}
            for r in valid:
                g = r.get(f"golden_{field}")
                p = r.get(f"pred_{field}")
                if field == "tech":
                    # Format tech arrays for comparison
                    g_str = ",".join(sorted(g)) if isinstance(g, list) else str(g)
                    p_str = ",".join(sorted(p)) if isinstance(p, list) else str(p)
                    if g_str != p_str:
                        key = f"{g_str}->{p_str}"
                        transitions[key] = transitions.get(key, 0) + 1
                elif g and p and g != p:
                    transitions[f"{g}->{p}"] = transitions.get(f"{g}->{p}", 0) + 1
            if transitions:
                any_transitions = True
                top = sorted(transitions.items(), key=lambda x: -x[1])[:4]
                transition_lines.append(f"    {field:<14}: " + "  ".join(f"{k}:{v}" for k, v in top))
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
        print(f"  Error direction:  over-scoring {over_scores}  |  under-scoring {under_scores}")

        # ── Quick-copy summary line ──────────────────────────────────────────
        print()
        print("-" * W)
        print(
            f"  {adapter_folder}/{checkpoint_stem}"
            f"  label={pct:.1f}%"
            f"  loc={field_pcts['loc']:.1f}%"
            f"  role={field_pcts['sen']:.1f}%"
            f"  tech={field_pcts['tech']:.1f}%"
            f"  comp={field_pcts['comp']:.1f}%"
            f"  wa={field_pcts['arr']:.1f}%"
            f"  gf={lp_by_label['good_fit']:.0f}%"
            f"  maybe={lp_by_label['maybe']:.0f}%"
            f"  bad={lp_by_label['bad_fit']:.0f}%"
        )
    print("=" * W)

    # Always write predictions alongside the eval report
    if predictions and not args.job:
        pred_file = output_file.with_suffix(".predictions.jsonl")
        with open(pred_file, "w") as pf:
            for rec in predictions:
                pf.write(json.dumps(rec) + "\n")
        print(f"\nPredictions: {pred_file}")

        # Also write a summary JSON for programmatic comparison
        summary_file = output_file.with_suffix(".summary.json")
        summary = {
            "timestamp": timestamp,
            "adapter": args.adapter or "baseline",
            "checkpoint": checkpoint_stem,
            "model": args.model,
            "prompt": args.prompt,
            "test_file": args.test_file,
            "n_total": n,
            "n_valid": len(valid),
            "parse_failures": parse_failures,
            "invalid_tokens": invalid_tokens,
            "fuzzy_corrections": fuzzy_total,
            "label_accuracy": round(label_correct / len(valid) * 100, 1) if valid else 0,
            "true_accuracy": round(label_correct / n * 100, 1) if n > 0 else 0,
            "score_accuracy": round(score_correct / len(valid) * 100, 1) if valid else 0,
            "field_accuracy": {f: round(field_pcts.get(f, 0), 1) for f in TOKEN_FIELDS} if valid else {},
            "per_label": {lbl: round(lp_by_label.get(lbl, 0), 1) for lbl in ("good_fit", "maybe", "bad_fit")} if valid else {},
            "avg_speed_s": round(avg_speed, 1),
            "total_time_s": round(total_gen_time, 1),
            "hardware": "Mac M1 16GB MLX 6-bit",
            "experiment": "exp2_thinking",
            "retries_attempted": retries_attempted,
            "retries_succeeded": retries_succeeded,
            "unclosed_think_count": unclosed_think_count,
        }
        with open(summary_file, "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Summary: {summary_file}")

    # Finalize output
    if log_fh:
        log_fh.close()
        sys.stdout = sys.__stdout__
        if partial_file.exists():
            partial_file.rename(output_file)
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

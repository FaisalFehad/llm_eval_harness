#!/usr/bin/env python3
"""
Optimized MLX eval for V16 (and any MLX-based pipeline).

Uses mlx_lm.batch_generate for batched parallel inference.
Optionally reuses a static KV-cache prefix computed once and deep-copied per job.
Truncates job descriptions to reduce context length.

Usage:
    python3 versions/v16/scripts/eval_optimized.py \
        --model mlx-community/Qwen3-4B-bf16 \
        --adapter versions/v15/adapters/0000700_adapters.safetensors \
        --prompt versions/v16/prompts/student.txt \
        --test-file versions/v12/data/v12_original/test_labeled_audited.jsonl \
        --batch-size 32 \
        --output-dir eval_results/v16_optimized \
        --save-predictions
"""

import argparse
import datetime
import json
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.models.cache import make_prompt_cache

# Reuse V16 scoring logic (extends V7 with fine-grained comp tokens)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from semantic_tokens_v16 import (
    validate_prediction, compute_from_tokens,
    V7_TOKEN_FIELDS, V7_RAW_FIELDS,
)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "mlx-community/Qwen3-4B-bf16"
DEFAULT_TEST_FILE = "versions/v12/data/v12_original/test_labeled_audited.jsonl"
DEFAULT_PROMPT = "versions/v16/prompts/student.txt"
MAX_TOKENS = 1000
TOKEN_FIELDS = ("loc", "arr", "sen", "tech", "comp")


# ── Helpers (identical to eval_student_v7.py) ─────────────────────────────────

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def greedy_sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def parse_json_output(text: str) -> dict | None:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    text = re.sub(r'\{(\s*)(\w+)":', r'{"\2":', text)
    text = re.sub(r'\{(\s*)(\w+):', r'{"\2":', text)
    text = re.sub(r',(\s*)(\w+):', r',"\2":', text)
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
    match = re.search(r'\{[\s\S]*', text)
    if match:
        truncated = match.group()
        if truncated.count('"') % 2 != 0:
            truncated += '"'
        if '"tech"' in truncated and '[' in truncated.split('"tech"')[-1]:
            after_tech = truncated.split('"tech"')[-1]
            if after_tech.count('[') > after_tech.count(']'):
                truncated += ']'
        truncated += '}'
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    return None


def compare_tech_arrays(pred_tech, golden_tech) -> bool:
    if isinstance(pred_tech, str):
        pred_tech = [pred_tech]
    if isinstance(golden_tech, str):
        golden_tech = [golden_tech]
    return sorted(pred_tech) == sorted(golden_tech)


def score_v7_result(predicted: dict, golden: dict) -> dict:
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
    pred_computed = compute_from_tokens(pred)
    golden_computed = compute_from_tokens(golden)
    field_matches = {}
    for field in TOKEN_FIELDS:
        if field == "tech":
            field_matches[f"{field}_match"] = compare_tech_arrays(
                pred.get(field, []), golden.get(field, []))
        else:
            field_matches[f"{field}_match"] = pred.get(field) == golden.get(field)
        field_matches[f"pred_{field}"] = pred.get(field)
        field_matches[f"golden_{field}"] = golden.get(field)
    return {
        "valid": True,
        "fuzzy_fixes": fuzzy_fixes,
        **field_matches,
        "label_match": pred_computed["label"] == golden_computed["label"],
        "score_match": pred_computed["score"] == golden_computed["score"],
        "pred_label": pred_computed["label"],
        "golden_label": golden_computed["label"],
        "pred_score": pred_computed["score"],
        "golden_score": golden_computed["score"],
        "pred_loc_score": pred_computed["loc_score"],
        "pred_role_score": pred_computed["role_score"],
        "pred_tech_score": pred_computed["tech_score"],
        "pred_comp_score": pred_computed["comp_score"],
        **{f"pred_{f}": pred.get(f, "") for f in V7_RAW_FIELDS},
    }


# ── Prompt building ───────────────────────────────────────────────────────────

def build_formatted_prompt(tokenizer, prompt_template: str, job: dict, no_think: bool) -> str:
    raw_location = job.get("job_location", job.get("location", ""))
    prompt_text = (prompt_template
        .replace("{{job_title}}", job["title"])
        .replace("{{job_location}}", raw_location)
        .replace("{{jd_text}}", job["jd_text"]))

    messages = [
        {"role": "system", "content": "Respond with JSON only."},
        {"role": "user", "content": prompt_text},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if no_think:
        kwargs["enable_thinking"] = False
    formatted = tokenizer.apply_chat_template(messages, **kwargs)

    is_qwen3 = "qwen3" in str(tokenizer.name_or_path).lower() or "qwen3" in getattr(tokenizer, 'model_id', '').lower()
    # Fallback heuristic if name doesn't contain qwen3
    if not is_qwen3:
        try:
            # Check vocab for qwen3 thinking tokens
            THINK_START_TOKEN = "<think>"
            if hasattr(tokenizer, 'convert_tokens_to_ids') and tokenizer.convert_tokens_to_ids(THINK_START_TOKEN) not in (None, 0):
                is_qwen3 = True
        except Exception:
            pass
    if not is_qwen3 or no_think:
        formatted += "{"
    return formatted


# ── Static prefix cache ────────────────────────────────────────────────────────

def find_common_prefix_length(tokenizer, prompts: list[str]) -> int:
    """Return the length (in tokens) of the longest common prefix across prompts."""
    token_lists = [tokenizer.encode(p) for p in prompts]
    if not token_lists:
        return 0
    prefix_len = 0
    while prefix_len < min(len(t) for t in token_lists):
        if all(t[prefix_len] == token_lists[0][prefix_len] for t in token_lists):
            prefix_len += 1
        else:
            break
    return prefix_len


def deep_copy_cache(cache):
    """Deep-copy an mlx_lm KV cache for reuse."""
    new_cache = []
    for c in cache:
        cls = type(c)
        new_c = cls.__new__(cls)
        state = c.state
        if isinstance(state, tuple):
            new_state = tuple(mx.array(s) for s in state)
        elif isinstance(state, list):
            new_state = [mx.array(s) for s in state]
        else:
            new_state = mx.array(state)
        new_c.state = new_state
        if hasattr(c, 'meta_state'):
            new_c.meta_state = c.meta_state
        new_cache.append(new_c)
    return new_cache


def build_static_cache(model, tokenizer, prefix_tokens: list[int]):
    """Run the static prefix through the model once and return the KV cache."""
    cache = make_prompt_cache(model)
    y = mx.array(prefix_tokens)
    model(y[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--job", type=str, default=None,
                        help="Run specific job(s) by line number (1-indexed)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-think", action="store_true", default=True,
                        help="Disable Qwen3 thinking mode (default: True)")
    parser.add_argument("--think", action="store_true",
                        help="Enable Qwen3 thinking mode (overrides --no-think)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default 32)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Only evaluate first N jobs (for quick tests)")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Disable static KV-cache prefix reuse")
    args = parser.parse_args()

    if args.think:
        args.no_think = False

    # ── Output setup ──────────────────────────────────────────────────────────
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
        output_dir = Path(args.output_dir) / adapter_folder / "optimized"
        output_file = output_dir / f"{timestamp}_{test_stem}_{prompt_stem}_{checkpoint_stem}_batch{args.batch_size}.txt"
    else:
        adapter_folder = "baseline"
        checkpoint_stem = model_stem
        output_dir = Path(args.output_dir) / "baseline" / "optimized"
        output_file = output_dir / f"{timestamp}_{test_stem}_{prompt_stem}_{model_stem}_batch{args.batch_size}.txt"

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
    print(f"Model: {args.model}")
    adapter_path = None
    _tmp_adapter_dir = None  # Track temp dir for cleanup
    if args.adapter:
        p = Path(args.adapter)
        if p.is_file() and p.name != "adapters.safetensors":
            # Copy to a temp directory to avoid overwriting the live adapters.safetensors
            _tmp_adapter_dir = tempfile.mkdtemp(prefix="mlx_adapter_")
            # Copy adapter config (needed by mlx_lm.load)
            config_src = p.parent / "adapter_config.json"
            if config_src.exists():
                shutil.copy2(config_src, Path(_tmp_adapter_dir) / "adapter_config.json")
            shutil.copy2(p, Path(_tmp_adapter_dir) / "adapters.safetensors")
            adapter_path = _tmp_adapter_dir
            print(f"Adapter: {args.adapter} -> {adapter_path}/adapters.safetensors (temp)")
        else:
            adapter_path = str(p.parent) if p.is_file() else str(p)
            print(f"Adapter: {adapter_path}")
    else:
        print("No adapter (baseline)")
    model, tokenizer = load(args.model, adapter_path=adapter_path)
    print("-" * 60)

    # Load test examples
    all_examples = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

    if args.n_jobs:
        all_examples = all_examples[:args.n_jobs]

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

    print(f"Running optimized V7 eval on {len(test_examples)} jobs...")
    print(f"  batch_size: {args.batch_size}")
    print(f"  no_think:   {args.no_think}")
    print(f"  use_cache:  {not args.skip_cache}")
    print("-" * 60)

    # ── Discover static prefix ────────────────────────────────────────────────
    prefix_len = 0
    static_cache = None
    if not args.skip_cache and len(test_examples) > 1:
        sample_jobs = test_examples[:min(4, len(test_examples))]
        sample_prompts = [build_formatted_prompt(tokenizer, prompt_template, job, args.no_think) for _, job in sample_jobs]
        prefix_len = find_common_prefix_length(tokenizer, sample_prompts)
        print(f"Static prefix tokens: {prefix_len}")
        if prefix_len > 0:
            prefix_tokens = tokenizer.encode(sample_prompts[0])[:prefix_len]
            t_cache = time.time()
            static_cache = build_static_cache(model, tokenizer, prefix_tokens)
            print(f"  prefix-cache built in {time.time() - t_cache:.2f}s")
        print("-" * 60)

    # ── Batch inference ───────────────────────────────────────────────────────
    results = []
    predictions = []
    parse_failures = 0
    invalid_tokens = 0
    fuzzy_total = 0
    seq = 0

    eval_start_time = time.time()
    total_prompt_tokens = 0
    total_gen_tokens = 0

    for batch_start in range(0, len(test_examples), args.batch_size):
        batch = test_examples[batch_start:batch_start + args.batch_size]

        # Build prompt tokens for the batch
        full_prompts = [build_formatted_prompt(tokenizer, prompt_template, job, args.no_think) for _, job in batch]
        full_tokens_list = [tokenizer.encode(p) for p in full_prompts]

        # If using prefix cache, take suffixes
        if static_cache:
            prompt_tokens_list = [tokens[prefix_len:] for tokens in full_tokens_list]
            prompt_caches = [deep_copy_cache(static_cache) for _ in batch]
        else:
            prompt_tokens_list = full_tokens_list
            prompt_caches = None

        # Batch generate
        t0 = time.time()
        response = batch_generate(
            model, tokenizer,
            prompts=prompt_tokens_list,
            prompt_caches=prompt_caches,
            max_tokens=args.max_tokens,
            sampler=greedy_sampler,
            prefill_step_size=4096,
            verbose=False,
        )
        batch_elapsed = time.time() - t0
        per_job_elapsed = batch_elapsed / len(batch)

        stats = response.stats
        total_prompt_tokens += stats.prompt_tokens
        total_gen_tokens += stats.generation_tokens

        # Score each job
        is_qwen3 = "qwen3" in args.model.lower()
        for i, (orig_idx, job) in enumerate(batch):
            seq += 1
            elapsed = per_job_elapsed

            raw_text = response.texts[i]

            # Pre-fill brace restoration
            if not is_qwen3 or args.no_think:
                raw_text = "{" + raw_text
            else:
                raw_text = re.sub(r"  思考.*?思考\s*", "", raw_text, flags=re.DOTALL)

            parsed = parse_json_output(raw_text)

            if parsed is None:
                parse_failures += 1
                print(f"[{orig_idx:3d}/{len(all_examples)}] !  {elapsed:4.1f}s  "
                      f"{job['title'][:42]:<42}  PARSE FAIL")
                print(f"           Raw: {raw_text[:300]}")
                results.append({"label_match": False, "parse_fail": True,
                                "golden_label": job.get("label", "?")})
                predictions.append({
                    "job_index": orig_idx, "title": job["title"],
                    "parse_fail": True, "raw_output": raw_text[:500],
                })
                continue

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
            label_col = (f"{scored['golden_label']}->{scored['pred_label']}"
                         if not scored["label_match"] else scored["golden_label"])
            prefix = f"[{orig_idx:3d}/{len(all_examples)}] {status}  {elapsed:4.1f}s  "
            print(f"{prefix}{job['title'][:42]:<42}  {label_col:<20}  {running_pct:3.0f}%")

            if not scored["label_match"] or args.verbose:
                indent = " " * len(prefix)
                parts = []
                for field in TOKEN_FIELDS:
                    g = scored.get(f"golden_{field}", "?")
                    p = scored.get(f"pred_{field}", "?")
                    icon = "\u2713" if scored.get(f"{field}_match") else "\u2717"
                    if field == "tech":
                        g_str = ",".join(g) if isinstance(g, list) else str(g)
                        p_str = ",".join(p) if isinstance(p, list) else str(p)
                        parts.append(f"{field} {g_str}->{p_str} {icon}")
                    else:
                        parts.append(f"{field} {g}->{p} {icon}")
                print(f"{indent}{' | '.join(parts)}")
                raws = [scored.get(f"pred_{f}", "") for f in V7_RAW_FIELDS]
                raw_str = " | ".join(r for r in raws if r)
                if raw_str:
                    print(f"{indent}{raw_str[:180]}")
                print()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    valid = [r for r in results if not r.get("parse_fail") and not r.get("invalid_token")]
    label_correct = sum(r["label_match"] for r in valid)
    score_correct = sum(r.get("score_match", False) for r in valid)

    W = 72
    run_id = f"{adapter_folder}  {checkpoint_stem}  batch={args.batch_size}"
    print("\n" + "=" * W)
    print(f"  V7 OPTIMIZED EVAL  {run_id}")
    print(f"  eval set: {test_stem}  |  prompt: {prompt_stem}  |  n={n}")
    print("=" * W)

    issues = parse_failures + invalid_tokens
    print(f"  Parse failures: {parse_failures}   Invalid tokens: {invalid_tokens}   "
          f"Fuzzy corrections: {fuzzy_total}   Valid: {len(valid)}/{n}")

    # Timing stats
    eval_duration = time.time() - eval_start_time
    avg_per_job = eval_duration / len(test_examples) if test_examples else 0
    print(f"  Total time: {eval_duration:.1f}s  |  Avg per job: {avg_per_job:.2f}s")
    print(f"  Prompt tokens: {total_prompt_tokens}  |  Gen tokens: {total_gen_tokens}")
    print()

    field_pcts = {}
    if valid:
        pct = label_correct / len(valid) * 100
        score_pct = score_correct / len(valid) * 100
        target_met = pct >= 90.0

        lp_by_label = {}
        for lbl in ("good_fit", "maybe", "bad_fit"):
            lbl_r = [r for r in valid if r["golden_label"] == lbl]
            lp_by_label[lbl] = (sum(r["label_match"] for r in lbl_r) / len(lbl_r) * 100) if lbl_r else 0.0
        macro_fit = sum(lp_by_label.values()) / len(lp_by_label)

        status_icon = "MET" if target_met else "NOT MET"
        print(f"  Label accuracy:  {label_correct}/{len(valid)} = {pct:.1f}%   {status_icon}  (target >=90%)")
        print(f"  Score accuracy:  {score_correct}/{len(valid)} = {score_pct:.1f}%")
        print()
        macro_icon = "MET" if macro_fit >= 90 else "NOT MET"
        print(f"  Fit accuracy (macro):  {macro_fit:.1f}%  {macro_icon}   "
              f"gf={lp_by_label['good_fit']:.0f}%  "
              f"maybe={lp_by_label['maybe']:.0f}%  "
              f"bad={lp_by_label['bad_fit']:.0f}%")
        print()

        print("  Field token accuracy:")
        for field in TOKEN_FIELDS:
            key = f"{field}_match"
            matched = [r[key] for r in valid if key in r]
            correct = sum(matched)
            fp = correct / len(matched) * 100 if matched else 0.0
            field_pcts[field] = fp
            bar_filled = int(fp / 5)
            bar = "#" * bar_filled + "." * (20 - bar_filled)
            print(f"    {field:<14} {bar}  {correct:>3}/{len(matched)}  {fp:.1f}%")
        print()

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

        print("-" * W)
        print(
            f"  {adapter_folder}/{checkpoint_stem}"
            f"  label={pct:.1f}%"
            f"  loc={field_pcts.get('loc', 0):.1f}%"
            f"  role={field_pcts.get('sen', 0):.1f}%"
            f"  tech={field_pcts.get('tech', 0):.1f}%"
            f"  comp={field_pcts.get('comp', 0):.1f}%"
            f"  wa={field_pcts.get('arr', 0):.1f}%"
            f"  gf={lp_by_label.get('good_fit', 0):.0f}%"
            f"  maybe={lp_by_label.get('maybe', 0):.0f}%"
            f"  bad={lp_by_label.get('bad_fit', 0):.0f}%"
        )
    print("=" * W)

    if predictions and not args.job:
        pred_file = output_file.with_suffix(".predictions.jsonl")
        with open(pred_file, "w") as pf:
            for rec in predictions:
                pf.write(json.dumps(rec) + "\n")
        print(f"\nPredictions: {pred_file}")

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
            "score_accuracy": round(score_correct / len(valid) * 100, 1) if valid else 0,
            "field_accuracy": {f: round(field_pcts.get(f, 0), 1) for f in TOKEN_FIELDS} if valid else {},
            "per_label": {lbl: round(lp_by_label.get(lbl, 0), 1) for lbl in ("good_fit", "maybe", "bad_fit")} if valid else {},
            "batch_size": args.batch_size,
            "prefix_cache_tokens": prefix_len,
            "avg_seconds_per_job": round(avg_per_job, 2),
            "total_seconds": round(eval_duration, 1),
        }
        with open(summary_file, "w") as sf:
            json.dump(summary, sf, indent=2)
        print(f"Summary: {summary_file}")

    if log_fh:
        log_fh.close()
        sys.stdout = sys.__stdout__
        if partial_file.exists():
            partial_file.rename(output_file)
        print(f"\nSaved: {output_file}")

    print(f"\nDuration: {eval_duration:.1f}s ({avg_per_job:.2f}s/job, {len(test_examples)} jobs)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout = sys.__stdout__
        print("\n\nInterrupted.")
        for p in Path("eval_results").rglob("*.partial"):
            p.unlink(missing_ok=True)
        sys.exit(1)

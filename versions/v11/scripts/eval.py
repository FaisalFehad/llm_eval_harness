#!/usr/bin/env python3
"""
Eval V11 student variants with optional cleaning, hint injection, and postprocessing.
- Supports 10-field (raw) and 5-field token-only outputs; scoring uses token fields.
- Cleaning: sentence-level filter matching build-v11 (drops boilerplate, keeps factual sentences, caps 28).
- Hint: prepend deterministic HINT line derived from regex heuristics.
- Postprocess: adjusts tokens using text heuristics (currency->comp, non-UK->loc, AI terms->tech, ensure tech not empty).

Usage examples:
  .venv/bin/python finetune/eval_student_v11.py \
      --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
      --adapter finetune/adapters_v11 \
      --prompt prompts/student_v11.txt \
      --test-file data/v11/valid_labeled.jsonl \
      --hint --postprocess

  .venv/bin/python finetune/eval_student_v11.py \
      --adapter finetune/adapters_v11_tokens \
      --prompt prompts/student_v11_tokens.txt \
      --test-file data/v7/test_labeled.jsonl \
      --clean --hint --postprocess
"""

import argparse
import datetime
import json
import re
import sys
import time
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent))
from semantic_tokens_v7 import (
    validate_prediction,
    compute_from_tokens,
    fuzzy_match,
    V7_TOKEN_FIELDS,
    V7_RAW_FIELDS,
)
from v11_preproc import clean_text, build_hint, apply_postprocess


def greedy_sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def parse_json_output(text: str) -> dict | None:
    text = text.strip().replace("```json", "").replace("```", "").strip()
    text = re.sub(r"\{(\s*)(\w+)\":", r"{\"\2\":", text)
    text = re.sub(r",(\s*)(\w+):", r",\"\2\":", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[\s\S]*", text)
    if match:
        truncated = match.group()
        if truncated.count('"') % 2 != 0:
            truncated += '"'
        truncated += "}"
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    # Fallback: regex salvage for key fields
    salvage: dict[str, object] = {}
    for field in ("loc", "arr", "sen", "comp"):
        m = re.search(rf'"?{field}"?\s*:\s*"([^"\n]+)"', text, re.I)
        if m:
            salvage[field] = m.group(1).strip().upper()
    tech_match = re.search(r'"?tech"?\s*:\s*\[([^\]]+)\]', text, re.I)
    if tech_match:
        items = re.findall(r'"([^"\]]+)"', "[" + tech_match.group(1) + "]")
        if items:
            salvage["tech"] = [s.strip().upper() for s in items]
    if set(["loc", "arr", "sen", "comp", "tech"]).issubset(salvage.keys()):
        return salvage
    return None


ALLOWED_LOC = ["IN_LONDON", "REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNK"]
ALLOWED_ARR = ["REMOTE", "HYBRID", "IN_OFFICE", "UNK"]
ALLOWED_SEN = ["LEVEL_3", "LEVEL_2", "LEVEL_1"]
ALLOWED_COMP = [
    "NO_GBP",
    "UP_TO_ONLY",
    "BELOW_45K",
    "RANGE_45_54K",
    "RANGE_55_74K",
    "RANGE_75_99K",
    "ABOVE_100K",
]
ALLOWED_TECH = ["NODE", "REACT", "JS_TS", "AI_ML", "OOS"]


def salvage_tokens(text: str) -> dict | None:
    t = text.upper()
    def find_one(options):
        for opt in options:
            if opt in t:
                return opt
        return None

    loc = find_one(ALLOWED_LOC)
    arr = find_one(ALLOWED_ARR)
    sen = find_one(ALLOWED_SEN)
    comp = find_one(ALLOWED_COMP)
    tech_found = [opt for opt in ALLOWED_TECH if opt in t]
    if tech_found and "OOS" in tech_found and len(tech_found) > 1:
        tech_found = [x for x in tech_found if x != "OOS"]
    if not tech_found:
        tech_found = ["OOS"]
    if loc and arr and sen and comp:
        return {"loc": loc, "arr": arr, "sen": sen, "comp": comp, "tech": sorted(set(tech_found))}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--hint", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    with open(args.prompt) as f:
        prompt_template = f.read()

    adapter_path = None
    if args.adapter:
        p = Path(args.adapter)
        adapter_path = str(p.parent) if p.is_file() else str(p)

    model, tokenizer = load(args.model, adapter_path=adapter_path)
    system_msg = "Respond with JSON only."

    jobs = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.test_file).stem
    pred_path = out_dir / f"{timestamp}_{stem}_preds.jsonl"

    log_to_stdout = os.environ.get("LOG_TO_STDOUT") == "1"
    if log_to_stdout:
        partial_log = None
    else:
        partial_log = out_dir / f"{timestamp}_{stem}.txt.partial"
        log_fh = open(partial_log, "w")
        sys.stdout = log_fh

    print(f"Adapter: {adapter_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Hint: {args.hint}  Postprocess: {args.postprocess}  Clean: {args.clean}")
    print(f"Jobs: {len(jobs)}")
    print("-" * 60)

    predictions = []
    results = []
    parse_failures = 0
    invalid_tokens = 0
    fuzzy_total = 0
    field_matches: dict[str, list[bool]] = {f: [] for f in V7_TOKEN_FIELDS}
    for seq, job in enumerate(jobs, 1):
        jd_text = job.get("jd_text", "")
        if args.clean:
            jd_text = clean_text(jd_text)
        if args.hint:
            jd_text = f"HINT: {build_hint(jd_text)}\n\n{jd_text}"

        raw_location = job.get("job_location", job.get("location", ""))
        prompt_text = (
            prompt_template
            .replace("{{job_title}}", job.get("title", ""))
            .replace("{{job_location}}", raw_location)
            .replace("{{jd_text}}", jd_text)
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted += "{"

        t0 = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=args.max_tokens,
            verbose=False,
            prefill_step_size=4096,
        )
        response = "{" + response
        elapsed = time.time() - t0

        parsed = parse_json_output(response)
        tokens = None
        if parsed is None:
            salvage = salvage_tokens(response)
            if salvage:
                tokens = salvage
            else:
                parse_failures += 1
                results.append({"label_match": False, "parse_fail": True, "golden_label": job.get("label")})
                predictions.append({"job_index": seq, "parse_fail": True, "raw_output": response[:400]})
                continue
        if tokens is None:
            validation = validate_prediction(parsed)
            if not validation["valid"]:
                salvage = salvage_tokens(response)
                if salvage:
                    tokens = salvage
                else:
                    invalid_tokens += 1
                    results.append({"label_match": False, "invalid_token": True, "golden_label": job.get("label"), "errors": validation["errors"]})
                    predictions.append({"job_index": seq, "invalid_token": True, "raw_output": response[:400]})
                    continue
            else:
                tokens = validation["corrected"]
                if validation.get("fuzzy_corrections"):
                    fuzzy_total += 1

        if args.postprocess:
            tokens = apply_postprocess(tokens, jd_text)

        computed = compute_from_tokens(tokens)
        label_match = computed["label"] == job.get("label")
        # Field accuracy tracking
        for f in V7_TOKEN_FIELDS:
            gold = job.get(f)
            pred_val = tokens.get(f)
            if gold is None:
                continue
            if f == "tech":
                def norm(v):
                    if isinstance(v, str):
                        return [v]
                    return sorted(v)
                field_matches[f].append(norm(pred_val) == norm(gold))
            else:
                field_matches[f].append(pred_val == gold)

        results.append({"label_match": label_match, "pred_label": computed["label"], "golden_label": job.get("label"), "elapsed": elapsed})
        predictions.append({
            "job_index": seq,
            "title": job.get("title", ""),
            "pred_tokens": {f: tokens.get(f) for f in V7_TOKEN_FIELDS},
            "pred_label": computed["label"],
            "pred_score": computed["score"],
            "golden_label": job.get("label"),
            "golden_score": job.get("score"),
            **{f"pred_{f}": tokens.get(f) for f in V7_RAW_FIELDS},
        })

        if seq % 50 == 0:
            correct_so_far = sum(r.get("label_match", False) for r in results)
            pct = correct_so_far / len(results) * 100
            print(f"[{seq}/{len(jobs)}] {pct:.1f}% acc | elapsed {elapsed:4.1f}s")

    valid = [r for r in results if not r.get("parse_fail") and not r.get("invalid_token")]
    label_correct = sum(r["label_match"] for r in valid)
    pct = label_correct / len(valid) * 100 if valid else 0

    print("=" * 60)
    print(f"Label accuracy: {pct:.1f}%  ({label_correct}/{len(valid)})")
    print(f"Parse failures: {parse_failures}  Invalid tokens: {invalid_tokens}  Fuzzy: {fuzzy_total}")
    if valid:
        field_acc: dict[str, float] = {}
        for f in V7_TOKEN_FIELDS:
            matches = field_matches.get(f, [])
            if matches:
                field_acc[f] = sum(matches) / len(matches) * 100
        print(f"Field accuracy: {field_acc}")
    print("=" * 60)

    if args.save_predictions:
        with open(pred_path, "w") as pf:
            for rec in predictions:
                pf.write(json.dumps(rec) + "\n")
        print(f"Predictions: {pred_path}")

    if not log_to_stdout and partial_log is not None:
        log_fh.close()
        final_log = partial_log.with_suffix(".txt")
        partial_log.rename(final_log)
        sys.stdout = sys.__stdout__
        print(f"Saved log: {final_log}")
        if args.save_predictions:
            print(f"Saved predictions: {pred_path}")
    else:
        if args.save_predictions:
            print(f"Predictions: {pred_path}")


if __name__ == "__main__":
    main()

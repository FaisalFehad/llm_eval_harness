#!/usr/bin/env python3
"""
Fast V7 teacher inference for distillation/filtering.
Outputs per-row predictions with job_index to align with source file lines.

Usage:
  .venv/bin/python finetune/infer_v7_teacher.py \
    --input data/v9/train_labeled.jsonl \
    --output data/v9/train_labeled_v7preds.jsonl \
    --adapter finetune/adapters_v7 \
    --prompt prompts/student_v7.txt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

sys.path.insert(0, str(Path(__file__).parent))
from semantic_tokens_v7 import validate_prediction, compute_from_tokens  # noqa: E402
from v11_preproc import clean_text


def greedy_sampler(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def parse_json_output(text: str) -> dict | None:
    """Robust JSON extraction (borrowed from eval_student_v7)."""
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    # Fix missing quotes on first key
    text = re.sub(r'\{(\s*)(\w+)":', r'{"\2":', text)
    text = re.sub(r'\{(\s*)(\w+):', r'{"\2":', text)
    text = re.sub(r',\s*(\w+):', r',"\1":', text)

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
        if '"tech"' in truncated and '[' in truncated.split('"tech"')[-1]:
            after = truncated.split('"tech"')[-1]
            if after.count('[') > after.count(']'):
                truncated += ']'
        truncated += '}'
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass
    return None


def run(args):
    with open(args.prompt) as f:
        prompt_template = f.read()

    print(f"Loading model {args.model} adapter={args.adapter}")
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    system_msg = "Respond with JSON only."

    jobs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))
    print(f"Jobs: {len(jobs)} from {args.input}")

    out_f = open(args.output, "w")
    limit = args.limit if args.limit and args.limit > 0 else len(jobs)

    for idx, job in enumerate(jobs[:limit], 1):
        raw_location = job.get("job_location", job.get("location", ""))
        jd_text = clean_text(job.get("jd_text", ""))
        prompt_text = (
            prompt_template
            .replace("{{job_title}}", job["title"])
            .replace("{{job_location}}", raw_location)
            .replace("{{jd_text}}", jd_text)
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted += "{"  # prefill opening brace

        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=args.max_tokens,
            verbose=False,
            prefill_step_size=4096,
            sampler=greedy_sampler,
        )
        response = "{" + response
        parsed = parse_json_output(response)
        pred_label = None
        pred_score = None
        parse_fail = False
        invalid_token = False
        raw_output = response[:400]
        if parsed is None:
            parse_fail = True
        else:
            validation = validate_prediction(parsed)
            if not validation["valid"]:
                invalid_token = True
            else:
                tokens = validation["corrected"]
                computed = compute_from_tokens(tokens)
                pred_label = computed["label"]
                pred_score = computed["score"]

        record = {
            "job_index": idx,
            "golden_label": job.get("label"),
            "golden_score": job.get("score"),
            "pred_label": pred_label,
            "pred_score": pred_score,
            "parse_fail": parse_fail,
            "invalid_token": invalid_token,
        }
        if parse_fail or invalid_token:
            record["raw_output"] = raw_output
        out_f.write(json.dumps(record) + "\n")
        if idx % 25 == 0:
            out_f.flush()
        if idx % 100 == 0:
            print(f"[{idx}/{len(jobs)}] done", flush=True)
    out_f.close()
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of rows to process")
    args = parser.parse_args()
    run(args)

#!/usr/bin/env python3
"""
Run V7 teacher on (unlabeled) jobs and emit predicted tokens/label/score.
Usage:
  .venv/bin/python finetune/infer_v7_teacher_tokens.py \
    --input data/v9/train_unlabeled_batch1.jsonl \
    --output data/v9/teacher_preds_batch1.jsonl \
    --adapter finetune/adapters_v7 \
    --prompt prompts/student_v7.txt \
    --max-tokens 180 --limit 2000
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
    text = text.strip().replace("```json", "").replace("```", "").strip()
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
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--max-tokens", type=int, default=180)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.prompt) as f:
        prompt_template = f.read()
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    system_msg = "Respond with JSON only."

    jobs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))
    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]
    print(f"Jobs: {len(jobs)} from {args.input}")

    out_f = open(args.output, "w")
    parse_fail = invalid = 0
    for idx, job in enumerate(jobs, 1):
        raw_location = job.get("job_location", job.get("location", ""))
        jd_text = job.get("jd_text", "")[:3500]
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

        response = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=args.max_tokens,
            verbose=False,
            sampler=greedy_sampler,
            prefill_step_size=4096,
        )
        response = "{" + response
        parsed = parse_json_output(response)
        if parsed is None:
            parse_fail += 1
            continue
        validation = validate_prediction(parsed)
        if not validation["valid"]:
            invalid += 1
            continue
        tokens = validation["corrected"]
        computed = compute_from_tokens(tokens)
        record = {
            "job_id": job.get("job_id", f"unlabeled_{idx}"),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "job_location": raw_location,
            "jd_text": jd_text,
            "loc": tokens.get("loc"),
            "arr": tokens.get("arr"),
            "sen": tokens.get("sen"),
            "tech": tokens.get("tech"),
            "comp": tokens.get("comp"),
            "score": computed["score"],
            "label": computed["label"],
            "source_file": job.get("source_file", "teacher_aug"),
        }
        out_f.write(json.dumps(record) + "\n")
        if idx % 5 == 0:
            out_f.flush()
            print(f"[{idx}/{len(jobs)}] written")
    out_f.close()
    print(f"Done. parse_fail={parse_fail} invalid={invalid} wrote={len(jobs)-parse_fail-invalid}")


if __name__ == "__main__":
    main()

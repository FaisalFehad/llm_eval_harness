#!/usr/bin/env python3
"""
Generate teacher labels for knowledge distillation.

Runs the fine-tuned teacher model on raw/golden JSONL jobs and writes scored
output compatible with format-finetune-training-data-for-mlx.ts.

Usage:
  source .venv/bin/activate

  # Label new raw jobs with the teacher (v9.8 prompt, v2b adapter)
  python finetune/generate_teacher_labels.py \
      --adapter finetune/adapters_v2b \
      --input data/real_linkedin_500.jsonl \
      --output data/distillation/teacher_labeled.jsonl \
      --prompt prompts/scorer_v9.8.txt

  # Re-label existing golden jobs with teacher's judgement
  python finetune/generate_teacher_labels.py \
      --adapter finetune/adapters_v2b \
      --input data/finetune/train.jsonl \
      --output data/distillation/train_relabeled.jsonl \
      --prompt prompts/scorer_v9.8.txt
"""

import argparse
import json
import re
from pathlib import Path

from mlx_lm import load, generate

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID   = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_TOKENS = 90


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output, handling extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def validate_label(parsed: dict) -> bool:
    """Check that parsed output has all required fields with valid values."""
    required = ["loc", "role", "tech", "comp", "score", "label"]
    if not all(k in parsed for k in required):
        return False
    if parsed["label"] not in ("good_fit", "maybe", "bad_fit"):
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher labels for knowledge distillation")
    parser.add_argument("--model", type=str, default=MODEL_ID,
                        help=f"Model ID or local path (default: {MODEL_ID})")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to adapter checkpoint (omit for baseline)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file (needs: title, location, jd_text)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file (scored jobs)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Path to prompt template .txt file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each prediction")
    args = parser.parse_args()

    # Load prompt template
    with open(args.prompt) as f:
        prompt_template = f.read()
    print(f"Prompt: {args.prompt}")

    # Resolve adapter path
    adapter_path = None
    if args.adapter:
        p = Path(args.adapter)
        adapter_path = str(p.parent) if p.is_file() else str(p)
        print(f"Loading model with adapter: {adapter_path}")
    else:
        print("Loading baseline model (no adapter)")

    print(f"Model: {args.model}")

    # Load model
    model, tokenizer = load(args.model, adapter_path=adapter_path)

    # Load input jobs
    jobs = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    print(f"Loaded {len(jobs)} jobs from {args.input}")
    print("─" * 60)

    # Run inference
    results = []
    parse_failures = []

    for i, job in enumerate(jobs, 1):
        prompt_text = (prompt_template
            .replace("{{job_title}}", job["title"])
            .replace("{{job_location}}", job.get("location", ""))
            .replace("{{jd_text}}", job["jd_text"]))

        system_msg = "/no_think" if "qwen3" in args.model.lower() else "Respond with JSON only."
        messages = [
            {"role": "system",  "content": system_msg},
            {"role": "user",    "content": prompt_text},
        ]

        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        parsed = parse_json_output(response)

        if parsed is None or not validate_label(parsed):
            parse_failures.append({
                "index": i,
                "title": job["title"],
                "raw_output": response[:200],
            })
            if args.verbose:
                print(f"[{i:3d}/{len(jobs)}] PARSE FAIL — {job['title'][:50]}")
                print(f"      Raw: {response[:100]}")
            continue

        # Build scored output (compatible with format-finetune-training-data-for-mlx.ts)
        scored_job = {
            "title":     job["title"],
            "company":   job.get("company", ""),
            "location":  job.get("location", ""),
            "jd_text":   job["jd_text"],
            "label":     parsed["label"],
            "score":     parsed["score"],
            "loc":       parsed["loc"],
            "role":      parsed["role"],
            "tech":      parsed["tech"],
            "comp":      parsed["comp"],
            "reasoning": parsed.get("reasoning", ""),
        }

        # Preserve job_id if present in input
        if "job_id" in job:
            scored_job["job_id"] = job["job_id"]

        results.append(scored_job)

        if args.verbose:
            print(f"[{i:3d}/{len(jobs)}] {parsed['label']:<10} "
                  f"{job['title'][:40]:<40} "
                  f"score={parsed['score']} "
                  f"loc={parsed['loc']} role={parsed['role']} "
                  f"tech={parsed['tech']} comp={parsed['comp']}")

        # Progress indicator every 20 jobs (when not verbose)
        if not args.verbose and i % 20 == 0:
            print(f"  [{i:3d}/{len(jobs)}] {len(results)} labeled, "
                  f"{len(parse_failures)} failures...")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for job in results:
            f.write(json.dumps(job) + "\n")

    # Write parse failures to separate file
    if parse_failures:
        fail_path = output_path.with_suffix(".failures.jsonl")
        with open(fail_path, "w") as f:
            for fail in parse_failures:
                f.write(json.dumps(fail) + "\n")

    # Summary
    print("\n" + "═" * 60)
    print("TEACHER LABELING COMPLETE")
    print("═" * 60)
    print(f"Input:          {len(jobs)} jobs")
    print(f"Labeled:        {len(results)}")
    print(f"Parse failures: {len(parse_failures)}")
    print(f"Output:         {output_path}")

    if parse_failures:
        print(f"Failures log:   {output_path.with_suffix('.failures.jsonl')}")

    # Label distribution
    from collections import Counter
    dist = Counter(r["label"] for r in results)
    print(f"\nLabel distribution:")
    for lbl in ["good_fit", "maybe", "bad_fit"]:
        print(f"  {lbl:<10}: {dist.get(lbl, 0)}")


if __name__ == "__main__":
    main()

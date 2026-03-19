#!/usr/bin/env python3
"""
GGUF eval script for V14 — uses llama-cpp-python for quantized model inference.
Compatible with IQ2_XXS, Q2_K, Q4_K_M etc. Outputs same format as eval_student_v14.py.

Usage:
    python3 finetune/eval_student_v14_gguf.py \
        --model ~/qwen3_4B_v14_IQ2_XXS.gguf \
        --test-file data/v12/test_labeled_audited.jsonl \
        --prompt prompts/student_v14.txt \
        --output-dir eval_results/v14_gguf_IQ2_XXS
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama


# ── Reuse V14 token vocab + parsing ──────────────────────────────────────────

VALID_LOC   = {"IN_LONDON", "REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNK"}
VALID_ARR   = {"REMOTE", "HYBRID", "IN_OFFICE", "UNK"}
VALID_SEN   = {"LEVEL_1", "LEVEL_2", "LEVEL_3"}
VALID_TECH  = {"NODE", "REACT", "JS_TS", "AI_ML", "OOS"}
VALID_COMP  = {"NO_GBP", "UP_TO_ONLY", "BELOW_45K", "RANGE_45_54K",
               "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K"}


def parse_json_output(text: str) -> dict | None:
    """Extract and validate JSON from model output."""
    text = text.strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    required = {"loc", "arr", "sen", "tech", "comp"}
    if not required.issubset(data):
        return None
    if data["loc"] not in VALID_LOC:
        return None
    if data["arr"] not in VALID_ARR:
        return None
    if data["sen"] not in VALID_SEN:
        return None
    if not isinstance(data["tech"], list) or not data["tech"]:
        return None
    if not all(t in VALID_TECH for t in data["tech"]):
        return None
    if data["comp"] not in VALID_COMP:
        return None
    return data


def build_prompt(prompt_template: str, job: dict) -> str:
    raw_location = job.get("location") or job.get("loc_raw") or ""
    return (prompt_template
            .replace("{{job_title}}", job.get("title", ""))
            .replace("{{job_location}}", raw_location)
            .replace("{{jd_text}}", job.get("jd_text", "")))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GGUF eval for V14 quantized models")
    parser.add_argument("--model", required=True, help="Path to .gguf file")
    parser.add_argument("--test-file", default="data/v12/test_labeled_audited.jsonl")
    parser.add_argument("--prompt", default="prompts/student_v14.txt")
    parser.add_argument("--output-dir", default="eval_results/v14_gguf")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="-1 = all layers on GPU")
    parser.add_argument("--ctx-size", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=600)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_stem = Path(args.model).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    pred_file = output_dir / f"{timestamp}_{model_stem}.predictions.jsonl"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading GGUF model: {args.model}")
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.ctx_size,
        verbose=False,
        # No chat_format override — use the Qwen3 Jinja template embedded in the
        # GGUF. The embedded template correctly handles /no_think and thinking mode.
    )
    print(f"Model loaded. Context: {args.ctx_size} tokens\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(args.test_file) as f:
        examples = [json.loads(l) for l in f if l.strip()]
    with open(args.prompt) as f:
        prompt_template = f.read()

    print(f"Test set: {len(examples)} jobs")
    print(f"Prompt:   {args.prompt}")
    print(f"Output:   {pred_file}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    results = []
    parse_failures = 0
    correct = 0
    start_all = time.time()

    with open(pred_file, "w") as out_f:
        for i, job in enumerate(examples, 1):
            t0 = time.time()
            prompt_text = build_prompt(prompt_template, job)

            messages = [
                # /no_think disables Qwen3 thinking mode (pre-fill doesn't work
                # with create_chat_completion — it closes the turn with <|im_end|>
                # causing empty output).
                # NOTE: explicit field list in system prompt HURTS accuracy (91.6%
                # vs 97.9% hybrid) — shifts model attention from job content to
                # schema, scrambling sen predictions. Keep prompt minimal.
                {"role": "system", "content": "Respond with JSON only. /no_think"},
                {"role": "user",   "content": prompt_text},
            ]

            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            raw = response["choices"][0]["message"]["content"].strip()
            elapsed = time.time() - t0

            parsed = parse_json_output(raw)
            label_true = job.get("label")
            label_pred = None

            if parsed is None:
                parse_failures += 1
                print(f"[{i:3d}/{len(examples)}] !  {elapsed:4.1f}s  "
                      f"{job['title'][:42]:<42}  PARSE FAIL")
                print(f"           Raw: {raw[:200]}")
            else:
                # Quick label check (model-only, no hybrid)
                label_match = (label_true == job.get("label"))
                result_str = "✓" if label_match else "✗"
                if label_match:
                    correct += 1
                running_pct = correct / i * 100
                print(f"[{i:3d}/{len(examples)}] {result_str}  {elapsed:4.1f}s  "
                      f"{job['title'][:42]:<42}  {running_pct:.0f}%")

            record = {
                "job_index": i,                    # 1-based; required by compute_hybrid_v13_1.py
                "job_id": job.get("job_id", i),
                "title": job.get("title", ""),
                "pred_tokens": parsed,             # field name expected by hybrid scorer
                "raw_response": raw,
                "parse_fail": parsed is None,
            }
            out_f.write(json.dumps(record) + "\n")
            results.append(record)

    total_time = time.time() - start_all
    n = len(examples)
    print(f"\n{'='*60}")
    print(f"Model:         {model_stem}")
    print(f"Jobs:          {n}")
    print(f"Parse failures:{parse_failures} ({parse_failures/n*100:.1f}%)")
    print(f"Time:          {total_time:.0f}s ({total_time/n:.1f}s/job)")
    print(f"Predictions:   {pred_file}")
    print(f"\nNext: python3 finetune/compute_hybrid_v13_1.py \\")
    print(f"        --test-file {args.test_file} \\")
    print(f"        --predictions {pred_file} --v12")


if __name__ == "__main__":
    main()

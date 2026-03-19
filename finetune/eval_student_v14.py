#!/usr/bin/env python3
"""
Eval HuggingFace PEFT models against V7 semantic token eval set.

Drop-in replacement for eval_student_v7.py for V14 (PyTorch/HF) models.
All V7 scoring logic is identical — only inference backend changes (HF vs MLX).

Key differences from eval_student_v7.py:
  - Loads base model + PEFT adapter via transformers/peft (not mlx_lm)
  - --adapter is a checkpoint DIRECTORY (e.g. finetune/adapters_v14/checkpoint-1800)
  - Greedy decoding via model.generate(do_sample=False)
  - No thinking mode (Qwen2.5 doesn't have it; --no-think accepted but ignored)

Output format is identical to eval_student_v7.py (.txt, .predictions.jsonl,
.summary.json) — compatible with compute_hybrid_v13_1.py and compare_evals.py.

Usage:
    python3 finetune/eval_student_v14.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter finetune/adapters_v14/checkpoint-1800 \\
        --test-file data/v12/test_labeled_audited.jsonl \\
        --prompt prompts/student_v14.txt \\
        --output-dir eval_results/v14 \\
        --save-predictions
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse all V7 scoring logic unchanged
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


# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TEST_FILE = "data/v12/test_labeled_audited.jsonl"
DEFAULT_PROMPT = "prompts/student_v14.txt"
MAX_TOKENS = 1000

TOKEN_FIELDS = ("loc", "arr", "sen", "tech", "comp")
SCORE_FIELDS = ("loc", "sen", "tech", "comp")


# ── Helpers (identical to eval_student_v7.py) ─────────────────────────────────

def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output, handling extra text and truncation."""
    text = text.strip()
    # Qwen3 thinking mode: strip <think>...</think> block (appears before final JSON)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="PEFT checkpoint directory (e.g. finetune/adapters_v14/checkpoint-1800)")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST_FILE)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--job", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--system-msg", type=str, default=None)
    parser.add_argument("--no-think", action="store_true",
                        help="Disable Qwen3 thinking mode (faster, fewer tokens). No-op for Qwen2.5.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size (default 32 — batching is ~15x faster than seq)")
    args = parser.parse_args()

    # ── Output file setup ─────────────────────────────────────────────────────
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    test_stem = Path(args.test_file).stem
    prompt_stem = Path(args.prompt).stem

    if args.adapter:
        adapter_path = Path(args.adapter)
        adapter_folder = adapter_path.parent.name
        dir_name = adapter_path.name
        checkpoint_stem = dir_name.replace("checkpoint-", "") if dir_name.startswith("checkpoint-") else dir_name
        output_dir = Path(args.output_dir) / adapter_path.parent.name
    else:
        adapter_folder = "baseline"
        checkpoint_stem = args.model.split("/")[-1]
        output_dir = Path(args.output_dir) / "baseline"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{timestamp}_{test_stem}_{prompt_stem}_{checkpoint_stem}.txt"
    partial_file = output_file.with_suffix(".txt.partial")

    log_fh = None
    if not args.job:
        log_fh = open(partial_file, "w")
        sys.stdout = Tee(sys.__stdout__, log_fh)
    print(f"Output: {output_file}")

    with open(args.prompt) as f:
        prompt_template = f.read()
    print(f"Prompt: {args.prompt}")

    # ── Load model + adapter ──────────────────────────────────────────────────
    print(f"Model:   {args.model}")
    print(f"Adapter: {args.adapter or 'none (baseline)'}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-padding required for decoder-only batched generation: all sequences must
    # end at the same position so generated tokens immediately follow the input.
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    if args.adapter:
        model = PeftModel.from_pretrained(base_model, args.adapter)
        print(f"PEFT adapter loaded: {args.adapter}")
    else:
        model = base_model
        print("No adapter (baseline)")

    model.train(False)      # disable dropout for inference (equivalent to model.eval())
    print("-" * 60)

    system_msg = args.system_msg or "Respond with JSON only."

    # ── Load test examples ────────────────────────────────────────────────────
    all_examples = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

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
    seq = 0  # running job counter across batches

    batch_size = 1 if args.job else args.batch_size  # single-job mode stays sequential
    print(f"Inference batch size: {batch_size}\n")

    for batch_start in range(0, len(test_examples), batch_size):
        batch = test_examples[batch_start:batch_start + batch_size]

        # ── Build formatted prompts for the whole batch ───────────────────────
        formatted_prompts = []
        for orig_idx, job in batch:
            raw_location = job.get("job_location", job.get("location", ""))
            prompt_text = (prompt_template
                .replace("{{job_title}}", job["title"])
                .replace("{{job_location}}", raw_location)
                .replace("{{jd_text}}", job["jd_text"]))
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text},
            ]
            template_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if args.no_think:
                template_kwargs["enable_thinking"] = False  # Qwen3: skip <think> block
            formatted = tokenizer.apply_chat_template(messages, **template_kwargs)

            formatted_prompts.append(formatted)

        # ── Tokenize batch (left-padded) and generate ─────────────────────────
        inputs = tokenizer(
            formatted_prompts, return_tensors="pt",
            padding=True, truncation=False,
        ).to(model.device)
        padded_input_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,                    # greedy, matches MLX greedy_sampler
                pad_token_id=tokenizer.eos_token_id,
            )
        batch_elapsed = time.time() - t0
        per_job_elapsed = batch_elapsed / len(batch)

        # ── Decode and score each job in the batch ────────────────────────────
        for i, (orig_idx, job) in enumerate(batch):
            seq += 1
            elapsed = per_job_elapsed

            # Slice generated tokens: skip left-padded input, take only new tokens
            generated_ids = output_ids[i][padded_input_len:]
            raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response = raw_response                          # model outputs full JSON including opening brace

            parsed = parse_json_output(response)

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
    run_id = f"{adapter_folder}  step {checkpoint_stem}"
    print("\n" + "=" * W)
    print(f"  V14 EVAL  {run_id}")
    print(f"  test set: {test_stem}  |  prompt: {prompt_stem}  |  n={n}")
    print("=" * W)

    print(f"  Parse failures: {parse_failures}   Invalid tokens: {invalid_tokens}   "
          f"Fuzzy corrections: {fuzzy_total}   Valid: {len(valid)}/{n}")
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

        any_transitions = False
        transition_lines = []
        for field in TOKEN_FIELDS:
            transitions = {}
            for r in valid:
                g = r.get(f"golden_{field}")
                p = r.get(f"pred_{field}")
                if field == "tech":
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

        over_scores = sum(1 for r in valid if not r["label_match"] and
                         r["pred_label"] in ("good_fit", "maybe") and
                         r["golden_label"] == "bad_fit")
        under_scores = sum(1 for r in valid if not r["label_match"] and
                          r["pred_label"] == "bad_fit" and
                          r["golden_label"] in ("good_fit", "maybe"))
        print(f"  Error direction:  over-scoring {over_scores}  |  under-scoring {under_scores}")

        print()
        print("-" * W)
        print(
            f"  {adapter_folder}/step{checkpoint_stem}"
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout = sys.__stdout__
        print("\n\nInterrupted.")
        for p in Path("eval_results").rglob("*.partial"):
            p.unlink(missing_ok=True)
        sys.exit(1)

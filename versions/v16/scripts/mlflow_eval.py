#!/usr/bin/env python3
"""
V15 MLflow Evaluation — run the model and score results through MLflow.

This creates a predict_fn that calls the V15 model, defines custom scorers
for token accuracy, and runs mlflow.genai.evaluate() so results appear in
the MLflow UI under the Evaluations tab.

Usage:
    .venv/bin/python3 finetune/mlflow_eval_v15.py

    # With a different prompt:
    .venv/bin/python3 finetune/mlflow_eval_v15.py --prompt prompts/student_v15_fix5.txt

    # Quick test on 10 jobs:
    .venv/bin/python3 finetune/mlflow_eval_v15.py --limit 10
"""

import argparse
import json
import re
import sys

import mlflow
import pandas as pd

# ── Scoring maps (same as semantic_tokens_v7.py) ──────────────────────────
LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_MAP = {"NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10, "OOS": 0}
COMP_MAP = {"NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_30K": -30, "30_40K": -20, "40_50K": -10,
            "50_60K": 0, "60_70K": 5, "70_80K": 10, "80_90K": 15, "90_100K": 20,
            "100_120K": 25, "120_140K": 25, "140_160K": 25, "160_180K": 25,
            "180_200K": 25, "ABOVE_200K": 25,
            # Legacy V1 coarse buckets (fallback)
            "BELOW_45K": -30, "RANGE_45_54K": 0, "RANGE_55_74K": 5,
            "RANGE_75_99K": 15, "ABOVE_100K": 25}

def compute_label(tokens):
    loc_score = LOCATION_MAP.get(tokens.get("loc", "UNK"), 0)
    tech = tokens.get("tech", ["OOS"])
    if not isinstance(tech, list): tech = ["OOS"]
    is_oos = "OOS" in tech or len(tech) == 0
    role_score = 0 if is_oos else SENIORITY_MAP.get(tokens.get("sen", "LEVEL_1"), 0)
    tech_score = 0 if is_oos else sum(TECH_MAP.get(t, 0) for t in tech)
    comp_score = COMP_MAP.get(tokens.get("comp", "NO_GBP"), 0)
    score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
    label = "good_fit" if score >= 70 else ("maybe" if score >= 50 else "bad_fit")
    return label, score


# ── Global model (loaded once) ─────────────────────────────────────────────
_model = None
_tokenizer = None
_generate = None
_prompt_template = None


def load_model(model_name, adapter_path, prompt_path):
    global _model, _tokenizer, _generate, _prompt_template
    from mlx_lm import load, generate
    _generate = generate
    print(f"Loading {model_name} + {adapter_path}...")
    _model, _tokenizer = load(model_name, adapter_path=adapter_path)
    with open(prompt_path) as f:
        _prompt_template = f.read()
    print("Model ready.")


def predict_fn(title: str = "", location: str = "", jd_text: str = "") -> dict:
    """Predict function for mlflow.genai.evaluate().

    MLflow calls this with keyword args matching the input keys.
    """

    # Build prompt
    user_msg = _prompt_template.replace("{{job_title}}", title)
    user_msg = user_msg.replace("{{job_location}}", location)
    user_msg = user_msg.replace("{{jd_text}}", jd_text)

    messages = [
        {"role": "system", "content": "Respond with JSON only."},
        {"role": "user", "content": user_msg},
    ]

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt += "{"

    response = _generate(_model, _tokenizer, prompt=prompt, max_tokens=500, verbose=False)
    raw = "{" + response

    # Parse JSON
    try:
        # Try fixing unquoted keys
        fixed = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', raw)
        parsed = json.loads(fixed)
        tokens = {
            "loc": parsed.get("loc", "UNK"),
            "arr": parsed.get("arr", "UNK"),
            "sen": parsed.get("sen", "LEVEL_2"),
            "tech": parsed.get("tech", ["OOS"]),
            "comp": parsed.get("comp", "NO_GBP"),
        }
        label, score = compute_label(tokens)
        return {
            "response": raw[:500],
            "pred_label": label,
            "pred_score": score,
            "pred_loc": tokens["loc"],
            "pred_arr": tokens["arr"],
            "pred_sen": tokens["sen"],
            "pred_tech": json.dumps(tokens["tech"]),
            "pred_comp": tokens["comp"],
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "response": raw[:500],
            "pred_label": "PARSE_FAIL",
            "pred_score": 0,
            "pred_loc": "", "pred_arr": "", "pred_sen": "",
            "pred_tech": "", "pred_comp": "",
        }


# ── Custom Scorers ─────────────────────────────────────────────────────────

@mlflow.genai.scorer
def label_accuracy(inputs, outputs, expectations):
    """Does the predicted label match the golden label?"""
    return outputs.get("pred_label") == expectations.get("golden_label")


@mlflow.genai.scorer
def field_loc(inputs, outputs, expectations):
    """Does loc match?"""
    return outputs.get("pred_loc") == expectations.get("golden_loc")


@mlflow.genai.scorer
def field_tech(inputs, outputs, expectations):
    """Does tech match?"""
    pred = outputs.get("pred_tech", "")
    golden = expectations.get("golden_tech", "")
    try:
        return json.loads(pred) == json.loads(golden)
    except:
        return pred == golden


@mlflow.genai.scorer
def field_comp(inputs, outputs, expectations):
    """Does comp match?"""
    return outputs.get("pred_comp") == expectations.get("golden_comp")


@mlflow.genai.scorer
def field_sen(inputs, outputs, expectations):
    """Does sen match?"""
    return outputs.get("pred_sen") == expectations.get("golden_sen")


@mlflow.genai.scorer
def parse_success(inputs, outputs, expectations):
    """Did the model produce parseable output?"""
    return outputs.get("pred_label") != "PARSE_FAIL"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V15 MLflow Evaluation")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-bf16")
    parser.add_argument("--adapter", default="finetune/adapters_v15_4B")
    parser.add_argument("--prompt", default="prompts/student_v15_fix4.txt")
    parser.add_argument("--test-file", default="versions/v12/data/v12_original/test_labeled_audited.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N jobs")
    parser.add_argument("--experiment", default="V15-Student-Model")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(args.experiment)

    # Load model
    load_model(args.model, args.adapter, args.prompt)

    # Load test set
    test_jobs = []
    with open(args.test_file) as f:
        for line in f:
            test_jobs.append(json.loads(line))

    if args.limit:
        test_jobs = test_jobs[:args.limit]

    # Build eval dataset
    eval_data = []
    for j in test_jobs:
        golden_label, golden_score = compute_label(j)
        eval_data.append({
            "inputs": {
                "title": j.get("title", ""),
                "location": j.get("job_location", ""),
                "jd_text": j.get("jd_text", ""),
            },
            "expectations": {
                "golden_label": golden_label,
                "golden_loc": j.get("loc", ""),
                "golden_arr": j.get("arr", ""),
                "golden_sen": j.get("sen", ""),
                "golden_tech": json.dumps(sorted(j.get("tech", ["OOS"]))),
                "golden_comp": j.get("comp", ""),
            },
        })

    print(f"Evaluating {len(eval_data)} jobs...")
    print(f"Prompt: {args.prompt}")

    # Run predictions sequentially FIRST (Metal GPU is not thread-safe)
    import time
    for i, entry in enumerate(eval_data):
        start = time.time()
        inp = entry["inputs"]
        output = predict_fn(
            title=inp["title"],
            location=inp["location"],
            jd_text=inp["jd_text"],
        )
        entry["outputs"] = output
        elapsed = time.time() - start
        status = "OK" if output.get("pred_label") != "PARSE_FAIL" else "FAIL"
        print(f"  [{i+1:3d}/{len(eval_data)}] {elapsed:.1f}s  {inp['title'][:45]:45s} {status} {output.get('pred_label','?')}")

    # Now pass pre-computed results to MLflow (no predict_fn — no threading)
    prompt_name = args.prompt.split("/")[-1].replace(".txt", "")
    run_name = args.run_name or f"eval-{prompt_name}-{len(eval_data)}jobs"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": args.model,
            "adapter": args.adapter,
            "prompt": args.prompt,
            "test_file": args.test_file,
            "n_jobs": len(eval_data),
        })

        results = mlflow.genai.evaluate(
            data=eval_data,
            scorers=[
                label_accuracy,
                field_loc,
                field_tech,
                field_comp,
                field_sen,
                parse_success,
            ],
        )

        print(f"\nResults:")
        for k, v in results.metrics.items():
            print(f"  {k}: {v}")

    print(f"\nDone! View results at http://localhost:5000")
    print(f"Go to Experiments → {args.experiment} → click the run → Evaluation tab")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze wrong predictions from MLX 6-bit eval.
Shows every job where the model got something wrong, with full JD text for manual review.
"""

import json
import sys

PREDICTIONS_FILE = (
    "/Users/faisal/Code/automation/ai_eval_harness/eval_results/v14_mlx6bit/baseline/"
    "2026-03-19_120501_test_labeled_audited_student_v14_qwen3_4B_v14_mlx6bit.predictions.jsonl"
)
GOLDEN_FILE = "/Users/faisal/Code/automation/ai_eval_harness/data/v12/test_labeled_audited.jsonl"

FIELDS = ["loc", "arr", "sen", "tech", "comp"]

# ── Load golden test set (0-based line index = job_index - 1) ──────────────────
golden = []
with open(GOLDEN_FILE) as f:
    for line in f:
        golden.append(json.loads(line.strip()))

# ── Load predictions ───────────────────────────────────────────────────────────
parse_failures = []
wrong_preds = []

with open(PREDICTIONS_FILE) as f:
    for line in f:
        p = json.loads(line.strip())
        if "parse_fail" in p:
            parse_failures.append(p)
        else:
            # Check for field-level errors even when label_match is True
            g_tokens = p["golden_tokens"]
            pred_tokens = p.get("pred_tokens", {})
            wrong_fields = []
            for field in FIELDS:
                g_val = g_tokens.get(field)
                p_val = pred_tokens.get(field)
                if g_val != p_val:
                    wrong_fields.append((field, p_val, g_val))
            if wrong_fields or not p.get("label_match", True):
                wrong_preds.append((p, wrong_fields))

# ── Helper: get golden record by job_index ─────────────────────────────────────
def get_golden(job_index):
    idx = job_index - 1  # job_index is 1-based in predictions
    if 0 <= idx < len(golden):
        return golden[idx]
    # Fallback: scan (in case index is 0-based)
    for g in golden:
        if g.get("job_id") == job_index:
            return g
    return None

SEP = "=" * 80

# ── Section 1: Parse failures ──────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"SECTION 1: PARSE FAILURES ({len(parse_failures)} jobs)")
print(SEP)

for i, p in enumerate(parse_failures, 1):
    job_index = p["job_index"]
    g = get_golden(job_index)

    print(f"\n{'─'*60}")
    print(f"[PARSE FAIL {i}/{len(parse_failures)}]  Job #{job_index}  |  {p.get('title', 'N/A')}")
    if g:
        print(f"  Location field (golden):  {g.get('job_location', 'N/A')}")
        print(f"  Golden tokens:  loc={g['loc']}  arr={g['arr']}  sen={g['sen']}  tech={g['tech']}  comp={g['comp']}")
        print(f"  Golden label:   {g['label']}  (score={g['score']})")
    print(f"\n  RAW MODEL OUTPUT:")
    raw = p.get("raw_output", "(none)")
    print("  " + raw.replace("\n", "\n  "))
    if g:
        jd = g.get("jd_text", "(no JD text)")
        print(f"\n  FULL JD TEXT:")
        print("  " + jd.replace("\n", "\n  "))

# ── Section 2: Wrong field predictions (parsed OK but fields wrong) ────────────
print(f"\n{SEP}")
print(f"SECTION 2: WRONG FIELD PREDICTIONS ({len(wrong_preds)} jobs)")
print(SEP)

for i, (p, wrong_fields) in enumerate(wrong_preds, 1):
    job_index = p["job_index"]
    g = get_golden(job_index)

    label_match = p.get("label_match", True)
    label_flag = "  *** LABEL WRONG ***" if not label_match else ""

    print(f"\n{'─'*60}")
    print(f"[WRONG {i}/{len(wrong_preds)}]  Job #{job_index}  |  {p.get('title', 'N/A')}{label_flag}")
    if g:
        print(f"  Location field (golden):  {g.get('job_location', 'N/A')}")
    print(f"  Golden label:   {p['golden_label']}  (score={p['golden_score']})")
    print(f"  Pred label:     {p['pred_label']}  (score={p['pred_score']})")

    print(f"\n  WRONG FIELDS:")
    for field, pred_val, golden_val in wrong_fields:
        print(f"    {field:6s}:  pred={pred_val}  vs  golden={golden_val}")

    print(f"\n  ALL PRED TOKENS:   {p.get('pred_tokens', {})}")
    print(f"  ALL GOLDEN TOKENS: {p['golden_tokens']}")

    print(f"\n  PRED RAW FIELDS (what model extracted verbatim):")
    print(f"    loc_raw:  {p.get('pred_loc_raw', 'N/A')}")
    print(f"    arr_raw:  {p.get('pred_arr_raw', 'N/A')}")
    print(f"    sen_raw:  {p.get('pred_sen_raw', 'N/A')}")
    print(f"    tech_raw: {p.get('pred_tech_raw', 'N/A')}")
    print(f"    comp_raw: {p.get('pred_comp_raw', 'N/A')}")

    if g:
        jd = g.get("jd_text", "(no JD text)")
        print(f"\n  FULL JD TEXT:")
        print("  " + jd.replace("\n", "\n  "))

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
print(f"  Total test jobs:          239")
print(f"  Predictions file:         {len(parse_failures) + len(wrong_preds) + (239 - len(parse_failures) - len(wrong_preds))} loaded")
print(f"  Parse failures:           {len(parse_failures)}")
print(f"  Wrong field predictions:  {len(wrong_preds)}")
print(f"    of which label wrong:   {sum(1 for p, _ in wrong_preds if not p.get('label_match', True))}")
print(f"  Total problems:           {len(parse_failures) + len(wrong_preds)}")
print(SEP)

#!/usr/bin/env python3
"""
Analyze whether V7 0.5B parse failures (27 excluded jobs) have a
different golden-label distribution than the 212 valid predictions.

If parse failures are disproportionately good_fit/maybe, the reported
84.9% accuracy on valid-only is inflated.

Also compares against the 1.5B model (0 parse failures, 85.4%) as
a sanity check.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── File paths ──────────────────────────────────────────────────────────────

TEST_FILE = BASE / "data/v7/test_labeled.jsonl"
PRED_05B = BASE / "eval_results/adapters_v7/2026-03-11_044934_test_labeled_student_v7_final.predictions.jsonl"
PRED_15B = BASE / "eval_results/adapters_v7_1.5B/2026-03-11_085836_test_labeled_student_v7_final.predictions.jsonl"
SUMMARY_05B = BASE / "eval_results/adapters_v7/2026-03-11_044934_test_labeled_student_v7_final.summary.json"
SUMMARY_15B = BASE / "eval_results/adapters_v7_1.5B/2026-03-11_085836_test_labeled_student_v7_final.summary.json"

# ── Load data ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_json(path):
    with open(path) as f:
        return json.load(f)

# Token -> score mappings (from semantic_tokens_v7.py)
LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_INDIVIDUAL_MAP = {"NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10, "OOS": 0}
COMP_MAP = {"NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0, "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25}

def compute_label(job):
    """Compute label from token fields in the test set."""
    loc_score = LOCATION_MAP.get(job["loc"], 0)
    tech = job.get("tech", ["OOS"])
    is_oos = "OOS" in tech or len(tech) == 0
    role_score = 0 if is_oos else SENIORITY_MAP.get(job["sen"], 0)
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    comp_score = COMP_MAP.get(job["comp"], 0)
    score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
    if score >= 70:
        return "good_fit", score
    elif score >= 50:
        return "maybe", score
    else:
        return "bad_fit", score


print("=" * 72)
print("  PARSE FAILURE SELECTION BIAS ANALYSIS — V7 0.5B")
print("=" * 72)

# ── 1. Load test set and index by job_id ────────────────────────────────────

test_jobs = load_jsonl(TEST_FILE)
print(f"\nTest set: {len(test_jobs)} jobs")

# The predictions use job_index (1-indexed line number in test file)
# Build a map from job_index to test job
test_by_index = {i + 1: job for i, job in enumerate(test_jobs)}

# Also verify golden labels
for idx, job in test_by_index.items():
    label, score = compute_label(job)
    job["_computed_label"] = label
    job["_computed_score"] = score

# ── 2. Load 0.5B predictions and identify excluded jobs ────────────────────

preds_05b = load_jsonl(PRED_05B)
summary_05b = load_json(SUMMARY_05B)
print(f"\n0.5B predictions file: {len(preds_05b)} records")
print(f"  Summary: n_total={summary_05b['n_total']}, n_valid={summary_05b['n_valid']}, "
      f"parse_failures={summary_05b['parse_failures']}, invalid_tokens={summary_05b['invalid_tokens']}")

# Separate parse failures from valid predictions
parse_fail_indices = set()
valid_pred_indices = set()

for pred in preds_05b:
    idx = pred["job_index"]
    if pred.get("parse_fail"):
        parse_fail_indices.add(idx)
    else:
        valid_pred_indices.add(idx)

# Jobs NOT in predictions at all = invalid token jobs
all_indices = set(test_by_index.keys())
in_pred_file = parse_fail_indices | valid_pred_indices
missing_indices = all_indices - in_pred_file  # These are the invalid token jobs

print(f"\n  Parse failures (in predictions file): {len(parse_fail_indices)}")
print(f"  Valid predictions (in predictions file): {len(valid_pred_indices)}")
print(f"  Missing from predictions (invalid tokens): {len(missing_indices)}")
print(f"  Total excluded (parse + invalid): {len(parse_fail_indices) + len(missing_indices)}")

excluded_indices = parse_fail_indices | missing_indices

# ── 3. Golden label distribution: excluded vs valid ─────────────────────────

print("\n" + "-" * 72)
print("  GOLDEN LABEL DISTRIBUTION")
print("-" * 72)

# Overall test set distribution
all_labels = Counter(job["_computed_label"] for job in test_jobs)
print(f"\n  Full test set ({len(test_jobs)} jobs):")
for lbl in ("good_fit", "maybe", "bad_fit"):
    n = all_labels[lbl]
    pct = n / len(test_jobs) * 100
    print(f"    {lbl:<10}  {n:>3}  ({pct:5.1f}%)")

# Excluded jobs distribution
excluded_jobs = [test_by_index[i] for i in sorted(excluded_indices)]
excluded_labels = Counter(job["_computed_label"] for job in excluded_jobs)
print(f"\n  EXCLUDED jobs ({len(excluded_jobs)} jobs — parse failures + invalid tokens):")
for lbl in ("good_fit", "maybe", "bad_fit"):
    n = excluded_labels.get(lbl, 0)
    pct = n / len(excluded_jobs) * 100 if excluded_jobs else 0
    print(f"    {lbl:<10}  {n:>3}  ({pct:5.1f}%)")

# Valid jobs distribution
valid_jobs = [test_by_index[i] for i in sorted(valid_pred_indices)]
valid_labels = Counter(job["_computed_label"] for job in valid_jobs)
print(f"\n  VALID jobs ({len(valid_jobs)} jobs):")
for lbl in ("good_fit", "maybe", "bad_fit"):
    n = valid_labels.get(lbl, 0)
    pct = n / len(valid_jobs) * 100 if valid_jobs else 0
    print(f"    {lbl:<10}  {n:>3}  ({pct:5.1f}%)")

# ── 3b. Break down by parse_fail vs invalid_token ──────────────────────────

print(f"\n  PARSE FAILURE jobs ({len(parse_fail_indices)}):")
pf_labels = Counter(test_by_index[i]["_computed_label"] for i in parse_fail_indices)
for lbl in ("good_fit", "maybe", "bad_fit"):
    n = pf_labels.get(lbl, 0)
    pct = n / len(parse_fail_indices) * 100 if parse_fail_indices else 0
    print(f"    {lbl:<10}  {n:>3}  ({pct:5.1f}%)")

print(f"\n  INVALID TOKEN jobs ({len(missing_indices)}):")
it_labels = Counter(test_by_index[i]["_computed_label"] for i in missing_indices)
for lbl in ("good_fit", "maybe", "bad_fit"):
    n = it_labels.get(lbl, 0)
    pct = n / len(missing_indices) * 100 if missing_indices else 0
    print(f"    {lbl:<10}  {n:>3}  ({pct:5.1f}%)")

# ── 4. Chi-square-like comparison ──────────────────────────────────────────

print("\n" + "-" * 72)
print("  OVER/UNDER-REPRESENTATION IN EXCLUDED JOBS")
print("-" * 72)

print(f"\n  {'Label':<10}  {'Test %':>7}  {'Excluded %':>11}  {'Delta':>7}  {'Direction':>12}")
print(f"  {'-'*10}  {'-'*7}  {'-'*11}  {'-'*7}  {'-'*12}")
for lbl in ("good_fit", "maybe", "bad_fit"):
    test_pct = all_labels[lbl] / len(test_jobs) * 100
    excl_pct = excluded_labels.get(lbl, 0) / len(excluded_jobs) * 100 if excluded_jobs else 0
    delta = excl_pct - test_pct
    direction = "OVER" if delta > 2 else ("UNDER" if delta < -2 else "~same")
    print(f"  {lbl:<10}  {test_pct:>6.1f}%  {excl_pct:>10.1f}%  {delta:>+6.1f}%  {direction:>12}")

# ── 5. What-if: assume all excluded jobs are WRONG ──────────────────────────

print("\n" + "-" * 72)
print("  WHAT-IF ANALYSIS: WORST-CASE AND BEST-CASE SCENARIOS")
print("-" * 72)

n_valid = len(valid_pred_indices)
n_excluded = len(excluded_indices)
n_total = len(test_jobs)

# From the valid predictions, count correct labels
valid_correct = 0
for pred in preds_05b:
    if pred.get("parse_fail"):
        continue
    if pred.get("label_match"):
        valid_correct += 1

print(f"\n  Valid predictions: {n_valid}, correct: {valid_correct} ({valid_correct/n_valid*100:.1f}%)")
print(f"  Excluded (parse + invalid): {n_excluded}")

# Worst case: all excluded = wrong
worst_case_correct = valid_correct + 0
worst_case_pct = worst_case_correct / n_total * 100
print(f"\n  WORST CASE (all {n_excluded} excluded = wrong):")
print(f"    {worst_case_correct}/{n_total} = {worst_case_pct:.1f}%")

# Best case: all excluded = correct
best_case_correct = valid_correct + n_excluded
best_case_pct = best_case_correct / n_total * 100
print(f"\n  BEST CASE (all {n_excluded} excluded = correct):")
print(f"    {best_case_correct}/{n_total} = {best_case_pct:.1f}%")

# Expected case based on base rate (valid accuracy)
valid_acc = valid_correct / n_valid
expected_correct_excl = round(n_excluded * valid_acc)
expected_total = valid_correct + expected_correct_excl
expected_pct = expected_total / n_total * 100
print(f"\n  EXPECTED CASE (excluded have same ~{valid_acc*100:.1f}% acc as valid):")
print(f"    {expected_total}/{n_total} = {expected_pct:.1f}%")

# ── 6. 1.5B comparison (0 parse failures) ──────────────────────────────────

print("\n" + "-" * 72)
print("  1.5B MODEL COMPARISON (0 PARSE FAILURES)")
print("-" * 72)

preds_15b = load_jsonl(PRED_15B)
summary_15b = load_json(SUMMARY_15B)

# The 1.5B has 0 parse failures but 13 invalid tokens
# So predictions file should have ~226 records (all valid, no parse fail markers)
print(f"\n  1.5B: n_total={summary_15b['n_total']}, n_valid={summary_15b['n_valid']}, "
      f"parse_failures={summary_15b['parse_failures']}, invalid_tokens={summary_15b['invalid_tokens']}")

# Check what 1.5B predicted on the same 27 jobs excluded by 0.5B
valid_15b_indices = set()
for pred in preds_15b:
    valid_15b_indices.add(pred["job_index"])

missing_15b = all_indices - valid_15b_indices  # Invalid token jobs for 1.5B

print(f"\n  1.5B valid predictions: {len(valid_15b_indices)}")
print(f"  1.5B missing (invalid tokens): {len(missing_15b)}")

# How did 1.5B do on the 27 jobs excluded by 0.5B?
overlap_with_excluded = excluded_indices & valid_15b_indices
not_in_15b_either = excluded_indices - valid_15b_indices

print(f"\n  Of the {len(excluded_indices)} jobs excluded by 0.5B:")
print(f"    1.5B has valid predictions for: {len(overlap_with_excluded)}")
print(f"    1.5B also excludes: {len(not_in_15b_either)}")

if overlap_with_excluded:
    # Check 1.5B accuracy on these jobs
    correct_on_excluded = 0
    for pred in preds_15b:
        if pred["job_index"] in overlap_with_excluded:
            if pred.get("label_match"):
                correct_on_excluded += 1

    pct_on_excluded = correct_on_excluded / len(overlap_with_excluded) * 100
    print(f"\n  1.5B accuracy on the {len(overlap_with_excluded)} excluded-by-0.5B jobs:")
    print(f"    {correct_on_excluded}/{len(overlap_with_excluded)} = {pct_on_excluded:.1f}%")
    print(f"    vs. 1.5B overall: {summary_15b['label_accuracy']}%")

    if pct_on_excluded < summary_15b['label_accuracy'] - 5:
        print(f"\n    >> YES, these excluded jobs ARE harder — 1.5B scores {summary_15b['label_accuracy'] - pct_on_excluded:.1f}pp lower on them")
    elif pct_on_excluded > summary_15b['label_accuracy'] + 5:
        print(f"\n    >> NO, these excluded jobs are EASIER — 1.5B scores {pct_on_excluded - summary_15b['label_accuracy']:.1f}pp higher on them")
    else:
        print(f"\n    >> These excluded jobs are SIMILAR difficulty to the rest")

# Per-label breakdown of 1.5B on excluded jobs
print(f"\n  1.5B results on excluded-by-0.5B jobs, by golden label:")
label_results_on_excluded = defaultdict(lambda: {"correct": 0, "total": 0})
for pred in preds_15b:
    if pred["job_index"] in overlap_with_excluded:
        golden = pred.get("golden_label", "?")
        label_results_on_excluded[golden]["total"] += 1
        if pred.get("label_match"):
            label_results_on_excluded[golden]["correct"] += 1

for lbl in ("good_fit", "maybe", "bad_fit"):
    r = label_results_on_excluded[lbl]
    if r["total"] > 0:
        pct = r["correct"] / r["total"] * 100
        print(f"    {lbl:<10}  {r['correct']}/{r['total']} = {pct:.0f}%")
    else:
        print(f"    {lbl:<10}  0/0")

# ── 7. Detailed listing of excluded jobs ────────────────────────────────────

print("\n" + "-" * 72)
print("  DETAILED LISTING OF EXCLUDED JOBS (sorted by golden label)")
print("-" * 72)

excluded_details = []
for idx in sorted(excluded_indices):
    job = test_by_index[idx]
    reason = "PARSE_FAIL" if idx in parse_fail_indices else "INVALID_TOKEN"

    # Check if 1.5B got it right
    result_15b = "N/A"
    for pred in preds_15b:
        if pred["job_index"] == idx:
            result_15b = "CORRECT" if pred.get("label_match") else "WRONG"
            break
    else:
        result_15b = "EXCLUDED_15B"

    excluded_details.append({
        "idx": idx,
        "title": job["title"][:45],
        "label": job["_computed_label"],
        "score": job["_computed_score"],
        "loc": job["loc"],
        "tech": job["tech"],
        "comp": job["comp"],
        "reason": reason,
        "result_15b": result_15b,
    })

# Sort by label then index
excluded_details.sort(key=lambda x: ({"good_fit": 0, "maybe": 1, "bad_fit": 2}[x["label"]], x["idx"]))

print(f"\n  {'Idx':>4}  {'Reason':<14}  {'Label':<10}  {'Score':>5}  {'1.5B':>12}  {'Title':<45}")
print(f"  {'-'*4}  {'-'*14}  {'-'*10}  {'-'*5}  {'-'*12}  {'-'*45}")
for d in excluded_details:
    tech_str = ",".join(d["tech"]) if isinstance(d["tech"], list) else str(d["tech"])
    print(f"  {d['idx']:>4}  {d['reason']:<14}  {d['label']:<10}  {d['score']:>5}  {d['result_15b']:>12}  {d['title']}")

# ── 8. Final verdict ───────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  VERDICT")
print("=" * 72)

# Calculate the key numbers
excl_gf_pct = excluded_labels.get("good_fit", 0) / len(excluded_jobs) * 100 if excluded_jobs else 0
test_gf_pct = all_labels["good_fit"] / len(test_jobs) * 100
excl_maybe_pct = excluded_labels.get("maybe", 0) / len(excluded_jobs) * 100 if excluded_jobs else 0
test_maybe_pct = all_labels["maybe"] / len(test_jobs) * 100
excl_hard_pct = excl_gf_pct + excl_maybe_pct
test_hard_pct = test_gf_pct + test_maybe_pct

print(f"\n  Harder labels (good_fit + maybe) in test set: {test_hard_pct:.1f}%")
print(f"  Harder labels (good_fit + maybe) in excluded: {excl_hard_pct:.1f}%")

if overlap_with_excluded:
    print(f"\n  1.5B accuracy on excluded-by-0.5B jobs: {pct_on_excluded:.1f}%")
    print(f"  1.5B accuracy overall: {summary_15b['label_accuracy']}%")

print(f"\n  0.5B reported (valid only): {summary_05b['label_accuracy']}%")
print(f"  0.5B worst case (all {n_excluded} excluded wrong): {worst_case_pct:.1f}%")
print(f"  0.5B best case (all {n_excluded} excluded right): {best_case_pct:.1f}%")

bias = excl_hard_pct - test_hard_pct
if abs(bias) > 5:
    if bias > 0:
        print(f"\n  >> BIAS DETECTED: Excluded jobs are {bias:.1f}pp more likely to be")
        print(f"     good_fit/maybe (the harder labels). The 84.9% is likely inflated.")
    else:
        print(f"\n  >> INVERSE BIAS: Excluded jobs are {-bias:.1f}pp MORE likely to be bad_fit.")
        print(f"     The 84.9% is actually conservative (deflated).")
else:
    print(f"\n  >> NO SIGNIFICANT BIAS: Excluded jobs have similar label distribution")
    print(f"     ({bias:+.1f}pp difference). The 84.9% figure is not selection-biased.")

print()

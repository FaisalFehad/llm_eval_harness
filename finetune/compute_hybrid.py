#!/usr/bin/env python3
"""
Compute ACTUAL hybrid accuracy: V7 model (loc/arr/sen) + regex (tech/comp).

Also analyzes:
- Regex error patterns (what does the regex get wrong?)
- Correlation between model and regex errors
- Confidence intervals for all accuracy numbers
- Per-label hybrid accuracy

Usage:
    python3 finetune/compute_hybrid.py \
        --test-file data/v7/test_labeled.jsonl \
        --predictions eval_results/adapters_v7/2026-03-11_044934_test_labeled_student_v7_final.predictions.jsonl
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Import the deterministic baseline classifier
sys.path.insert(0, str(Path(__file__).parent))
from deterministic_baseline import classify_job, compute_score_and_label

# ── Score Maps (from semantic_tokens_v7.py) ──────────────────────────────────

LOCATION_MAP = {
    "IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10,
    "OUTSIDE_UK": -50, "UNK": 0,
}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_INDIVIDUAL_MAP = {"OOS": 0, "NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10}
COMP_MAP = {
    "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30,
    "RANGE_45_54K": 0, "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
}


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))


def compute_label(loc: str, sen: str, tech: list, comp: str) -> dict:
    """Compute score and label from tokens."""
    loc_score = LOCATION_MAP.get(loc, 0)
    comp_score = COMP_MAP.get(comp, 0)
    is_oos = "OOS" in tech or len(tech) == 0
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    role_score = 0 if is_oos else SENIORITY_MAP.get(sen, 0)
    raw = loc_score + role_score + tech_score + comp_score
    score = max(0, min(100, raw))
    if score >= 70:
        label = "good_fit"
    elif score >= 50:
        label = "maybe"
    else:
        label = "bad_fit"
    return {"score": score, "label": label}


def main():
    parser = argparse.ArgumentParser(description="Compute hybrid accuracy (model + regex)")
    parser.add_argument("--test-file", required=True, help="Golden test set")
    parser.add_argument("--predictions", required=True, help="V7 model predictions .jsonl")
    parser.add_argument("--output", default=None, help="Save summary JSON to this path")
    parser.add_argument("--v12", action="store_true",
                        help="V12 hybrid mode: regex loc/tech/comp + model sen/arr")
    args = parser.parse_args()

    # Load test set
    test_jobs = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                test_jobs.append(json.loads(line))

    # Load model predictions
    model_preds = []
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                model_preds.append(json.loads(line))

    print(f"Test jobs: {len(test_jobs)}")
    print(f"Model predictions: {len(model_preds)}")

    # Index model predictions by job_index
    pred_by_idx = {}
    for p in model_preds:
        pred_by_idx[p["job_index"]] = p

    # Classify all test jobs with regex
    regex_preds = []
    for job in test_jobs:
        regex_preds.append(classify_job(job))

    # ── Compute approaches ──────────────────────────────────────────────
    # 1. Model only (handling parse failures)
    # 2. Regex only
    # 3. Hybrid A: model loc/arr/sen + regex tech/comp
    # 4. Hybrid B: model loc/sen + regex tech/comp/arr
    # 5. V12 Hybrid: regex loc/tech/comp + model sen/arr (--v12 flag)

    results = {
        "model_only": {"correct": 0, "total": 0, "field_correct": {f: 0 for f in ["loc","arr","sen","tech","comp"]}, "parse_fail": 0},
        "regex_only": {"correct": 0, "total": 0, "field_correct": {f: 0 for f in ["loc","arr","sen","tech","comp"]}},
        "hybrid_A": {"correct": 0, "total": 0, "field_correct": {f: 0 for f in ["loc","arr","sen","tech","comp"]}, "parse_fail": 0},
        "hybrid_B": {"correct": 0, "total": 0, "field_correct": {f: 0 for f in ["loc","arr","sen","tech","comp"]}, "parse_fail": 0},
    }
    if args.v12:
        results["v12_hybrid"] = {"correct": 0, "total": 0, "field_correct": {f: 0 for f in ["loc","arr","sen","tech","comp"]}, "parse_fail": 0}
    v12_errors = []
    v12_per_label = {l: {"correct": 0, "total": 0} for l in ["good_fit", "maybe", "bad_fit"]}

    # Track per-job details for error analysis
    hybrid_errors = []
    regex_errors_detail = {f: [] for f in ["loc","arr","sen","tech","comp"]}
    both_wrong = 0  # Jobs where both model and regex get label wrong
    model_wrong_regex_right = 0
    regex_wrong_model_right = 0

    label_order = ["good_fit", "maybe", "bad_fit"]
    hybrid_confusion = Counter()
    hybrid_per_label = {l: {"correct": 0, "total": 0} for l in label_order}

    for i, job in enumerate(test_jobs):
        golden_label = job["label"]
        golden = {f: job[f] for f in ["loc", "arr", "sen", "tech", "comp"]}
        regex = regex_preds[i]

        # Model prediction for this job
        job_idx = i + 1  # predictions are 1-indexed
        mp = pred_by_idx.get(job_idx)
        has_model = mp is not None and not mp.get("parse_fail", False)

        # ── Regex only ───────────────────────────────────────────────
        regex_result = compute_label(regex["loc"], regex["sen"], regex["tech"], regex["comp"])
        regex_label = regex_result["label"]
        results["regex_only"]["total"] += 1
        if regex_label == golden_label:
            results["regex_only"]["correct"] += 1

        # Per-field regex accuracy
        for f in ["loc", "arr", "sen", "comp"]:
            if regex[f] == golden[f]:
                results["regex_only"]["field_correct"][f] += 1
        if sorted(regex["tech"]) == sorted(golden["tech"]):
            results["regex_only"]["field_correct"]["tech"] += 1
        else:
            regex_errors_detail["tech"].append({
                "idx": i, "golden": golden["tech"], "predicted": regex["tech"],
                "title": job.get("title", ""), "job_id": job.get("job_id", ""),
            })

        for f in ["loc", "arr", "sen", "comp"]:
            if regex[f] != golden[f]:
                regex_errors_detail[f].append({
                    "idx": i, "golden": golden[f], "predicted": regex[f],
                    "title": job.get("title", ""), "job_id": job.get("job_id", ""),
                })

        # ── Model only ───────────────────────────────────────────────
        if has_model:
            model_tokens = mp["pred_tokens"]
            results["model_only"]["total"] += 1
            if mp.get("label_match", False):
                results["model_only"]["correct"] += 1

            for f in ["loc", "arr", "sen", "comp"]:
                if model_tokens[f] == golden[f]:
                    results["model_only"]["field_correct"][f] += 1
            if sorted(model_tokens["tech"]) == sorted(golden["tech"]):
                results["model_only"]["field_correct"]["tech"] += 1
        else:
            results["model_only"]["parse_fail"] += 1

        # ── Hybrid A: model loc/arr/sen + regex tech/comp ────────────
        if has_model:
            model_tokens = mp["pred_tokens"]
            h_loc = model_tokens["loc"]
            h_arr = model_tokens["arr"]
            h_sen = model_tokens["sen"]
            h_tech = regex["tech"]
            h_comp = regex["comp"]

            h_result = compute_label(h_loc, h_sen, h_tech, h_comp)
            h_label = h_result["label"]

            results["hybrid_A"]["total"] += 1
            hybrid_per_label[golden_label]["total"] += 1
            hybrid_confusion[(golden_label, h_label)] += 1

            if h_label == golden_label:
                results["hybrid_A"]["correct"] += 1
                hybrid_per_label[golden_label]["correct"] += 1

            # Per-field: model fields
            for f in ["loc", "arr", "sen"]:
                if model_tokens[f] == golden[f]:
                    results["hybrid_A"]["field_correct"][f] += 1
            # Regex fields
            if regex["comp"] == golden["comp"]:
                results["hybrid_A"]["field_correct"]["comp"] += 1
            if sorted(regex["tech"]) == sorted(golden["tech"]):
                results["hybrid_A"]["field_correct"]["tech"] += 1

            # Track error patterns
            if h_label != golden_label:
                diffs = []
                for f in ["loc", "arr", "sen"]:
                    if model_tokens[f] != golden[f]:
                        diffs.append(f"{f}: {golden[f]}→{model_tokens[f]}")
                if sorted(regex["tech"]) != sorted(golden["tech"]):
                    diffs.append(f"tech: {golden['tech']}→{regex['tech']}")
                if regex["comp"] != golden["comp"]:
                    diffs.append(f"comp: {golden['comp']}→{regex['comp']}")

                hybrid_errors.append({
                    "idx": i, "title": job.get("title", ""),
                    "golden_label": golden_label, "hybrid_label": h_label,
                    "golden_score": job.get("score", "?"), "hybrid_score": h_result["score"],
                    "diffs": diffs,
                })

            # Track model vs regex correlation
            model_label_correct = mp.get("label_match", False)
            regex_label_correct = (regex_label == golden_label)
            if not model_label_correct and not regex_label_correct:
                both_wrong += 1
            elif not model_label_correct and regex_label_correct:
                model_wrong_regex_right += 1
            elif model_label_correct and not regex_label_correct:
                regex_wrong_model_right += 1
        else:
            results["hybrid_A"]["parse_fail"] += 1
            # For parse failures, use regex only as fallback
            h_result_fb = compute_label(regex["loc"], regex["sen"], regex["tech"], regex["comp"])
            results["hybrid_A"]["total"] += 1
            hybrid_per_label[golden_label]["total"] += 1
            hybrid_confusion[(golden_label, h_result_fb["label"])] += 1
            if h_result_fb["label"] == golden_label:
                results["hybrid_A"]["correct"] += 1
                hybrid_per_label[golden_label]["correct"] += 1

        # ── Hybrid B: model loc/sen + regex tech/comp/arr ────────────
        if has_model:
            model_tokens = mp["pred_tokens"]
            hb_loc = model_tokens["loc"]
            hb_arr = regex["arr"]  # regex arr instead of model
            hb_sen = model_tokens["sen"]
            hb_tech = regex["tech"]
            hb_comp = regex["comp"]

            hb_result = compute_label(hb_loc, hb_sen, hb_tech, hb_comp)
            results["hybrid_B"]["total"] += 1
            if hb_result["label"] == golden_label:
                results["hybrid_B"]["correct"] += 1
        else:
            results["hybrid_B"]["parse_fail"] += 1
            results["hybrid_B"]["total"] += 1
            h_fb = compute_label(regex["loc"], regex["sen"], regex["tech"], regex["comp"])
            if h_fb["label"] == golden_label:
                results["hybrid_B"]["correct"] += 1

        # ── V12 Hybrid: regex loc/tech/comp + model sen/arr ────────────
        if args.v12:
            v12_loc = regex["loc"]  # regex loc (100% accuracy)
            v12_tech = regex["tech"]  # regex tech
            v12_comp = regex["comp"]  # regex comp

            if has_model:
                model_tokens = mp["pred_tokens"]
                v12_sen = model_tokens["sen"]  # model sen
                v12_arr = model_tokens["arr"]  # model arr
            else:
                # Parse failure fallback: regex for all fields
                results["v12_hybrid"]["parse_fail"] += 1
                v12_sen = regex["sen"]
                v12_arr = regex["arr"]

            v12_result = compute_label(v12_loc, v12_sen, v12_tech, v12_comp)
            v12_label = v12_result["label"]
            results["v12_hybrid"]["total"] += 1
            v12_per_label[golden_label]["total"] += 1

            if v12_label == golden_label:
                results["v12_hybrid"]["correct"] += 1
                v12_per_label[golden_label]["correct"] += 1
            else:
                diffs = []
                if v12_loc != golden["loc"]:
                    diffs.append(f"loc:{golden['loc']}->{v12_loc}")
                if has_model and model_tokens["sen"] != golden["sen"]:
                    diffs.append(f"sen:{golden['sen']}->{model_tokens['sen']}")
                if sorted(v12_tech) != sorted(golden["tech"]):
                    diffs.append(f"tech:{golden['tech']}->{v12_tech}")
                if v12_comp != golden["comp"]:
                    diffs.append(f"comp:{golden['comp']}->{v12_comp}")
                v12_errors.append({
                    "idx": i, "title": job.get("title", ""),
                    "golden_label": golden_label, "v12_label": v12_label,
                    "golden_score": job.get("score", "?"), "v12_score": v12_result["score"],
                    "diffs": diffs,
                })

            # Per-field accuracy for V12
            if v12_loc == golden["loc"]:
                results["v12_hybrid"]["field_correct"]["loc"] += 1
            if v12_arr == golden["arr"]:
                results["v12_hybrid"]["field_correct"]["arr"] += 1
            if (v12_sen if has_model else regex["sen"]) == golden["sen"]:
                results["v12_hybrid"]["field_correct"]["sen"] += 1
            if sorted(v12_tech) == sorted(golden["tech"]):
                results["v12_hybrid"]["field_correct"]["tech"] += 1
            if v12_comp == golden["comp"]:
                results["v12_hybrid"]["field_correct"]["comp"] += 1

    # ── Print Results ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HYBRID ACCURACY COMPUTATION — ACTUAL MEASURED RESULTS")
    print("=" * 70)

    n = len(test_jobs)
    for name, r in results.items():
        acc = 100 * r["correct"] / r["total"] if r["total"] > 0 else 0
        lo, hi = wilson_ci(r["correct"], r["total"])
        pf = r.get("parse_fail", 0)
        pf_str = f" (parse_fail={pf}, fallback to regex)" if pf > 0 else ""
        print(f"\n{name}: {r['correct']}/{r['total']} = {acc:.1f}%  95% CI: [{100*lo:.1f}%, {100*hi:.1f}%]{pf_str}")

        if "field_correct" in r:
            # For model_only: field accuracy is on valid predictions only (total already excludes parse fails)
            # For regex: field accuracy is on all 239 jobs
            # For hybrids: field accuracy is on valid model predictions (parse fails use regex fallback)
            if name == "regex_only" or name == "model_only":
                field_denom = r["total"]  # model_only total already excludes parse fails
            else:
                field_denom = r["total"] - r.get("parse_fail", 0)
            for f in ["loc", "arr", "sen", "tech", "comp"]:
                fc = r["field_correct"][f]
                fa = 100 * fc / field_denom if field_denom > 0 else 0
                print(f"  {f}: {fa:.1f}% ({fc}/{field_denom})")

    # ── Hybrid A per-label accuracy ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("HYBRID A — Per-Label Accuracy:")
    for lbl in label_order:
        c = hybrid_per_label[lbl]["correct"]
        t = hybrid_per_label[lbl]["total"]
        acc = 100 * c / t if t > 0 else 0
        lo, hi = wilson_ci(c, t)
        print(f"  {lbl:<12}: {c}/{t} = {acc:.1f}%  95% CI: [{100*lo:.1f}%, {100*hi:.1f}%]")

    # ── Confusion matrix ─────────────────────────────────────────────────
    print(f"\nHybrid A Confusion Matrix (rows=golden, cols=predicted):")
    print(f"  {'':>12} {'good_fit':>10} {'maybe':>10} {'bad_fit':>10}")
    for g in label_order:
        row = [hybrid_confusion[(g, p)] for p in label_order]
        print(f"  {g:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    # ── Error correlation ────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("ERROR CORRELATION (model vs regex, on valid predictions only):")
    valid_count = results["model_only"]["total"]
    both_right = results["model_only"]["correct"] - model_wrong_regex_right  # approximate
    # Recalculate properly
    model_correct = results["model_only"]["correct"]
    model_wrong = valid_count - model_correct
    regex_correct_on_valid = 0
    for i, job in enumerate(test_jobs):
        job_idx = i + 1
        mp = pred_by_idx.get(job_idx)
        if mp and not mp.get("parse_fail", False):
            regex_result = compute_label(regex_preds[i]["loc"], regex_preds[i]["sen"],
                                         regex_preds[i]["tech"], regex_preds[i]["comp"])
            if regex_result["label"] == job["label"]:
                regex_correct_on_valid += 1

    both_correct = model_correct - regex_wrong_model_right
    print(f"  Model correct, regex correct : {both_correct}")
    print(f"  Model correct, regex wrong   : {regex_wrong_model_right}")
    print(f"  Model wrong,   regex correct : {model_wrong_regex_right}")
    print(f"  Model wrong,   regex wrong   : {both_wrong}")
    print(f"  Total valid predictions       : {valid_count}")

    if both_wrong > 0:
        print(f"\n  Correlation insight: {both_wrong} jobs are wrong for BOTH model and regex")
        print(f"  → These are the 'hard' jobs that neither approach solves")
        print(f"  → Hybrid can't help on these — they set the floor")

    # ── Hybrid A error details ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"HYBRID A ERRORS ({len(hybrid_errors)} label mismatches):")
    for e in hybrid_errors[:30]:  # Show first 30
        print(f"\n  Job {e['idx']+1}: {e['title']}")
        print(f"    Golden: {e['golden_label']} (score={e['golden_score']})")
        print(f"    Hybrid: {e['hybrid_label']} (score={e['hybrid_score']})")
        print(f"    Field diffs: {', '.join(e['diffs']) if e['diffs'] else 'NONE (score calc diff)'}")

    # ── Regex error analysis ─────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("REGEX ERROR ANALYSIS:")
    for f in ["loc", "arr", "sen", "tech", "comp"]:
        errors = regex_errors_detail[f]
        n_err = len(errors)
        acc = 100 * (n - n_err) / n
        print(f"\n  {f}: {n_err} errors ({acc:.1f}% accuracy)")

        # Show transition distribution
        transitions = Counter()
        for e in errors:
            g = str(e["golden"])
            p = str(e["predicted"])
            transitions[f"{g} → {p}"] += 1
        for t, count in transitions.most_common(10):
            print(f"    {t}: {count}")

    # ── Confidence Intervals Summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("CONFIDENCE INTERVALS (95% Wilson score)")
    print(f"{'=' * 70}")

    comparisons = [
        ("V7 Model", results["model_only"]["correct"], results["model_only"]["total"]),
        ("Regex Only", results["regex_only"]["correct"], results["regex_only"]["total"]),
        ("Hybrid A (model sen/loc/arr + regex tech/comp)", results["hybrid_A"]["correct"], results["hybrid_A"]["total"]),
        ("Hybrid B (model sen/loc + regex tech/comp/arr)", results["hybrid_B"]["correct"], results["hybrid_B"]["total"]),
    ]
    if args.v12:
        comparisons.append(
            ("V12 Hybrid (regex loc/tech/comp + model sen/arr)", results["v12_hybrid"]["correct"], results["v12_hybrid"]["total"]),
        )

    for name, c, t in comparisons:
        acc = 100 * c / t if t > 0 else 0
        lo, hi = wilson_ci(c, t)
        print(f"  {name:<55} {acc:5.1f}%  [{100*lo:.1f}%, {100*hi:.1f}%]")

    print(f"\n  Note: With n=239, 1pp ≈ 2.4 jobs. Differences <3pp are within noise.")

    # ── V12 Hybrid details ─────────────────────────────────────────────
    if args.v12:
        r = results["v12_hybrid"]
        print(f"\n{'=' * 70}")
        print("V12 HYBRID DETAILS (regex loc/tech/comp + model sen/arr)")
        print(f"{'=' * 70}")
        acc = 100 * r["correct"] / r["total"]
        lo, hi = wilson_ci(r["correct"], r["total"])
        print(f"\n  Accuracy: {r['correct']}/{r['total']} = {acc:.1f}%  95% CI: [{100*lo:.1f}%, {100*hi:.1f}%]")
        print(f"  Parse failures (regex fallback): {r['parse_fail']}")
        print(f"\n  Per-field:")
        for f in ["loc", "arr", "sen", "tech", "comp"]:
            fc = r["field_correct"][f]
            fa = 100 * fc / r["total"]
            print(f"    {f}: {fa:.1f}% ({fc}/{r['total']})")
        print(f"\n  Per-label:")
        for lbl in ["good_fit", "maybe", "bad_fit"]:
            c = v12_per_label[lbl]["correct"]
            t = v12_per_label[lbl]["total"]
            a = 100 * c / t if t > 0 else 0
            print(f"    {lbl:<12}: {c}/{t} = {a:.1f}%")
        print(f"\n  Errors ({len(v12_errors)}):")
        for e in v12_errors:
            diff_str = ", ".join(e["diffs"]) if e["diffs"] else "score_calc_only"
            print(f"    Job {e['idx']+1}: {e['golden_label']}->{e['v12_label']} (score {e['golden_score']}->{e['v12_score']}) [{diff_str}] {e['title'][:50]}")

    # ── Save summary JSON ───────────────────────────────────────────────
    if args.output:
        summary = {}
        for name, r in results.items():
            summary[name] = {
                "correct": r["correct"],
                "total": r["total"],
                "accuracy_pct": round(100 * r["correct"] / r["total"], 1) if r["total"] > 0 else 0,
                "parse_fail": r.get("parse_fail", 0),
                "field_accuracy": {f: round(100 * r["field_correct"][f] / r["total"], 1)
                                   for f in ["loc","arr","sen","tech","comp"]}
                                   if r["total"] > 0 else {},
            }
        if args.v12:
            summary["v12_errors"] = [
                {"job_index": e["idx"]+1, "golden": e["golden_label"],
                 "predicted": e["v12_label"], "diffs": e["diffs"]}
                for e in v12_errors
            ]
            summary["v12_per_label"] = {
                lbl: {"correct": v12_per_label[lbl]["correct"],
                      "total": v12_per_label[lbl]["total"]}
                for lbl in ["good_fit", "maybe", "bad_fit"]
            }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()

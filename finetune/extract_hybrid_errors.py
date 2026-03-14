#!/usr/bin/env python3
"""Extract hybrid error job IDs for Phase 0B audit."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deterministic_baseline import classify_job

LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_INDIVIDUAL_MAP = {"OOS": 0, "NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10}
COMP_MAP = {
    "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0,
    "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
}


def compute_label(loc, sen, tech, comp):
    loc_score = LOCATION_MAP.get(loc, 0)
    comp_score = COMP_MAP.get(comp, 0)
    is_oos = "OOS" in tech or len(tech) == 0
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    role_score = 0 if is_oos else SENIORITY_MAP.get(sen, 0)
    raw = loc_score + role_score + tech_score + comp_score
    score = max(0, min(100, raw))
    label = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
    return label, score


def main():
    test_file = sys.argv[1] if len(sys.argv) > 1 else "data/v7/test_labeled.jsonl"
    pred_file = sys.argv[2] if len(sys.argv) > 2 else "eval_results/adapters_v7_1.5B/2026-03-11_085836_test_labeled_student_v7_final.predictions.jsonl"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "data/v12/hybrid_error_jobs.txt"

    test_jobs = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                test_jobs.append(json.loads(line))

    model_preds = []
    with open(pred_file) as f:
        for line in f:
            if line.strip():
                model_preds.append(json.loads(line))

    pred_by_idx = {p["job_index"]: p for p in model_preds}
    regex_preds = [classify_job(job) for job in test_jobs]

    errors = []
    boundary_non_errors = []  # score 50-74 that are correct (for Phase 0B audit)

    for i, job in enumerate(test_jobs):
        golden_label = job["label"]
        regex = regex_preds[i]
        job_idx = i + 1
        mp = pred_by_idx.get(job_idx)
        has_model = mp is not None and not mp.get("parse_fail", False)

        if has_model:
            mt = mp["pred_tokens"]
            h_label, h_score = compute_label(mt["loc"], mt["sen"], regex["tech"], regex["comp"])
        else:
            h_label, h_score = compute_label(regex["loc"], regex["sen"], regex["tech"], regex["comp"])

        if h_label != golden_label:
            # Categorize error source
            diffs = []
            if has_model:
                mt = mp["pred_tokens"]
                if mt["loc"] != job["loc"]:
                    diffs.append(f"loc:{job['loc']}->{mt['loc']}")
                if mt["sen"] != job["sen"]:
                    diffs.append(f"sen:{job['sen']}->{mt['sen']}")
            else:
                diffs.append("NO_MODEL(regex_fallback)")
                if regex["loc"] != job["loc"]:
                    diffs.append(f"loc:{job['loc']}->{regex['loc']}")
                if regex["sen"] != job["sen"]:
                    diffs.append(f"sen:{job['sen']}->{regex['sen']}")
            if sorted(regex["tech"]) != sorted(job["tech"]):
                diffs.append(f"tech:{job['tech']}->{regex['tech']}")
            if regex["comp"] != job["comp"]:
                diffs.append(f"comp:{job['comp']}->{regex['comp']}")

            errors.append({
                "job_index": job_idx,
                "title": job.get("title", ""),
                "golden_label": golden_label,
                "hybrid_label": h_label,
                "golden_score": job.get("score", "?"),
                "hybrid_score": h_score,
                "has_model": has_model,
                "diffs": diffs,
            })
        else:
            # Track boundary-zone correct jobs (score 50-74) for audit
            golden_score = job.get("score", 0)
            if 50 <= golden_score <= 74:
                boundary_non_errors.append({
                    "job_index": job_idx,
                    "title": job.get("title", ""),
                    "golden_label": golden_label,
                    "golden_score": golden_score,
                })

    # Save error job IDs
    with open(output_file, "w") as f:
        f.write(f"# V12 Phase 0A: Hybrid error jobs ({len(errors)} total)\n")
        f.write(f"# Baseline: {239 - len(errors)}/239 = {(239 - len(errors)) / 239 * 100:.1f}%\n")
        f.write(f"# V7 1.5B model + regex hybrid (Hybrid A)\n\n")
        f.write("## Error Jobs\n")
        f.write("# idx | golden      | hybrid      | g_score | h_score | diffs | title\n\n")
        for e in errors:
            diff_str = ", ".join(e["diffs"]) if e["diffs"] else "score_calc_only"
            f.write(f"{e['job_index']:3d} | {e['golden_label']:<11} | {e['hybrid_label']:<11} | {e['golden_score']:>7} | {e['hybrid_score']:>7} | {diff_str} | {e['title']}\n")
        f.write(f"\n## Boundary-Zone Non-Error Jobs (score 50-74, correct) — {len(boundary_non_errors)} total\n")
        f.write("# idx | golden      | g_score | title\n\n")
        for b in boundary_non_errors:
            f.write(f"{b['job_index']:3d} | {b['golden_label']:<11} | {b['golden_score']:>7} | {b['title']}\n")

    print(f"Saved {len(errors)} error jobs + {len(boundary_non_errors)} boundary jobs to {output_file}")
    print(f"\n=== ERROR JOBS ({len(errors)}) ===")
    for e in errors:
        diff_str = ", ".join(e["diffs"]) if e["diffs"] else "score_calc_only"
        print(f"  Job {e['job_index']:3d}: {e['golden_label']:<11} -> {e['hybrid_label']:<11} (score {e['golden_score']:>3} -> {e['hybrid_score']:>3}) [{diff_str}]")
        print(f"          {e['title'][:80]}")

    print(f"\n=== BOUNDARY-ZONE CORRECT JOBS ({len(boundary_non_errors)}) ===")
    for b in boundary_non_errors[:25]:
        print(f"  Job {b['job_index']:3d}: {b['golden_label']:<11} score={b['golden_score']} {b['title'][:60]}")
    if len(boundary_non_errors) > 25:
        print(f"  ... and {len(boundary_non_errors) - 25} more")


if __name__ == "__main__":
    main()

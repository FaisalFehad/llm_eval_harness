#!/usr/bin/env python3
"""
V12 Phase 2A: Build training dataset.

Steps:
1. Start with V7's 713 labeled jobs
2. Remove 212 generated_v7 jobs (99.5% NODE bias) → 501 real jobs
3. Add V9 real jobs not already in V7 (decontaminated against test set)
4. Apply regex corrections to tech/comp/loc fields
5. Recompute scores and labels
6. Verify distribution

Output: data/v12/train_labeled.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deterministic_baseline import classify_job, _is_plausible_salary

# Score maps for recomputation
LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_INDIVIDUAL_MAP = {"OOS": 0, "NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10}
COMP_MAP = {
    "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0,
    "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
}


def recompute_score_label(job):
    """Recompute score and label from token fields."""
    loc_score = LOCATION_MAP.get(job["loc"], 0)
    comp_score = COMP_MAP.get(job["comp"], 0)
    tech = job["tech"]
    is_oos = "OOS" in tech or len(tech) == 0
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    role_score = 0 if is_oos else SENIORITY_MAP.get(job["sen"], 0)
    raw = loc_score + role_score + tech_score + comp_score
    score = max(0, min(100, raw))
    label = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
    job["score"] = score
    job["label"] = label
    return job


def job_fingerprint(job):
    """Create a fingerprint for deduplication (title + first 200 chars of JD)."""
    title = (job.get("title") or "").strip().lower()
    jd = (job.get("jd_text") or "")[:200].strip().lower()
    return f"{title}||{jd}"


def apply_regex_corrections(job, regex_pred, corrections_log):
    """Apply regex corrections to tech/comp/loc where regex is more reliable."""
    changed = False

    # TECH correction: only when regex has HIGH confidence
    # High confidence = specific keyword match, not bare \bai\b
    golden_tech = job["tech"]
    regex_tech = regex_pred["tech"]
    if sorted(golden_tech) != sorted(regex_tech):
        # Only correct specific known error patterns:
        # 1. Teacher added AI_ML but regex says no → remove AI_ML (boilerplate false positive)
        if "AI_ML" in golden_tech and "AI_ML" not in regex_tech:
            new_tech = [t for t in golden_tech if t != "AI_ML"]
            if not new_tech:
                new_tech = ["OOS"]
            corrections_log.append({
                "job_id": job.get("job_id", "?"),
                "field": "tech",
                "old": golden_tech,
                "new": new_tech,
                "reason": "regex: AI_ML not found (likely boilerplate)",
            })
            job["tech"] = new_tech
            changed = True

        # 2. Teacher missed NODE but regex found it (bare "node" pattern)
        if "NODE" not in golden_tech and "NODE" in regex_tech and "OOS" not in regex_tech:
            new_tech = [t for t in golden_tech if t != "OOS"] + ["NODE"]
            new_tech = sorted(set(new_tech))
            corrections_log.append({
                "job_id": job.get("job_id", "?"),
                "field": "tech",
                "old": golden_tech,
                "new": new_tech,
                "reason": "regex: NODE found (bare node pattern)",
            })
            job["tech"] = new_tech
            changed = True

    # COMP correction: only when regex has clear GBP salary and teacher disagrees
    golden_comp = job["comp"]
    regex_comp = regex_pred["comp"]
    if golden_comp != regex_comp:
        # TC/total compensation disqualifier
        jd_lower = (job.get("jd_text") or "").lower()
        has_tc = bool(re.search(r'\btotal\s+comp(?:ensation)?\b|\btotal\s+package\b', jd_lower))
        if has_tc and golden_comp != "NO_GBP":
            corrections_log.append({
                "job_id": job.get("job_id", "?"),
                "field": "comp",
                "old": golden_comp,
                "new": "NO_GBP",
                "reason": "TC/total compensation disqualifier",
            })
            job["comp"] = "NO_GBP"
            changed = True

    # LOC correction: regex loc is 100% accurate
    golden_loc = job["loc"]
    regex_loc = regex_pred["loc"]
    if golden_loc != regex_loc:
        corrections_log.append({
            "job_id": job.get("job_id", "?"),
            "field": "loc",
            "old": golden_loc,
            "new": regex_loc,
            "reason": f"regex loc (100% accuracy): {golden_loc}->{regex_loc}",
        })
        job["loc"] = regex_loc
        changed = True

    if changed:
        recompute_score_label(job)

    return job


def main():
    v7_file = "data/v7/train_labeled.jsonl"
    v9_file = "data/v9/train_labeled.jsonl"
    test_file = "data/v12/test_labeled_audited.jsonl"
    output_file = "data/v12/train_labeled.jsonl"
    report_file = "data/v12/build_report.json"

    # ── Step 1: Load V7 training data ──────────────────────────────────
    v7_jobs = []
    with open(v7_file) as f:
        for line in f:
            if line.strip():
                v7_jobs.append(json.loads(line))
    print(f"V7 training data: {len(v7_jobs)} jobs")

    # ── Step 2: Remove generated_v7 jobs ──────────────────────────────
    real_v7 = [j for j in v7_jobs if j.get("source_file") != "generated_v7"]
    removed = len(v7_jobs) - len(real_v7)
    print(f"Removed {removed} generated_v7 jobs → {len(real_v7)} real V7 jobs")

    # ── Step 3: Load V9 and find new real jobs ─────────────────────────
    v9_jobs = []
    with open(v9_file) as f:
        for line in f:
            if line.strip():
                v9_jobs.append(json.loads(line))
    print(f"V9 training data: {len(v9_jobs)} jobs")

    # Load test set for decontamination
    test_jobs = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                test_jobs.append(json.loads(line))
    test_fingerprints = set(job_fingerprint(j) for j in test_jobs)
    print(f"Test set: {len(test_jobs)} jobs ({len(test_fingerprints)} unique fingerprints)")

    # Build V7 fingerprint set for deduplication
    v7_fingerprints = set(job_fingerprint(j) for j in real_v7)

    # Filter V9: keep only real jobs not in V7 and not in test set
    synthetic_sources = {"generated_v7", "generated_v9", "node_variants_v7", "node_variants_v6"}
    v9_new = []
    v9_skipped = Counter()
    for job in v9_jobs:
        source = job.get("source_file", "unknown")
        if source in synthetic_sources:
            v9_skipped["synthetic"] += 1
            continue
        fp = job_fingerprint(job)
        if fp in v7_fingerprints:
            v9_skipped["duplicate_v7"] += 1
            continue
        if fp in test_fingerprints:
            v9_skipped["test_contaminated"] += 1
            continue
        v9_new.append(job)
        v7_fingerprints.add(fp)  # prevent V9 internal duplicates

    print(f"V9 new real jobs: {len(v9_new)} (skipped: {dict(v9_skipped)})")

    # ── Step 4: Combine ──────────────────────────────────────────────
    combined = real_v7 + v9_new
    print(f"Combined: {len(combined)} jobs")

    # ── Step 5: Apply regex corrections ──────────────────────────────
    corrections_log = []
    for job in combined:
        regex_pred = classify_job(job)
        apply_regex_corrections(job, regex_pred, corrections_log)

    # Recompute all scores/labels after corrections
    for job in combined:
        recompute_score_label(job)

    print(f"\nRegex corrections applied: {len(corrections_log)}")
    field_counts = Counter(c["field"] for c in corrections_log)
    for field, count in field_counts.most_common():
        print(f"  {field}: {count}")

    if len(corrections_log) > 100:
        print("  WARNING: >100 corrections — investigate before proceeding!")

    # ── Step 6: Distribution verification ────────────────────────────
    print(f"\n{'=' * 60}")
    print("DISTRIBUTION VERIFICATION")
    print(f"{'=' * 60}")

    label_dist = Counter(j["label"] for j in combined)
    total = len(combined)
    print(f"\nLabel distribution (target: ~25% gf, ~28% maybe, ~47% bad):")
    for lbl in ["good_fit", "maybe", "bad_fit"]:
        pct = 100 * label_dist[lbl] / total
        print(f"  {lbl:<12}: {label_dist[lbl]:>4} ({pct:.1f}%)")

    # Tech distribution
    oos_count = sum(1 for j in combined if "OOS" in j["tech"])
    oos_pct = 100 * oos_count / total
    print(f"\nOOS: {oos_count}/{total} ({oos_pct:.1f}%) (target: ~15%)")

    # Comp distribution
    print("\nComp distribution:")
    comp_dist = Counter(j["comp"] for j in combined)
    for comp in ["NO_GBP", "UP_TO_ONLY", "BELOW_45K", "RANGE_45_54K",
                  "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K"]:
        pct = 100 * comp_dist[comp] / total
        print(f"  {comp:<15}: {comp_dist[comp]:>4} ({pct:.1f}%)")

    # Seniority distribution
    print("\nSeniority distribution (target: ~35%/30%/35%):")
    sen_dist = Counter(j["sen"] for j in combined)
    for sen in ["LEVEL_1", "LEVEL_2", "LEVEL_3"]:
        pct = 100 * sen_dist[sen] / total
        print(f"  {sen:<10}: {sen_dist[sen]:>4} ({pct:.1f}%)")

    # Tech token distribution
    print("\nTech token distribution:")
    tech_dist = Counter()
    for j in combined:
        for t in j["tech"]:
            tech_dist[t] += 1
    for tech in ["NODE", "REACT", "JS_TS", "AI_ML", "OOS"]:
        pct = 100 * tech_dist[tech] / total
        print(f"  {tech:<8}: {tech_dist[tech]:>4} ({pct:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────
    with open(output_file, "w") as f:
        for job in combined:
            f.write(json.dumps(job) + "\n")
    print(f"\nSaved {len(combined)} jobs to {output_file}")

    # Save build report
    report = {
        "v7_total": len(v7_jobs),
        "v7_real": len(real_v7),
        "v7_generated_removed": removed,
        "v9_total": len(v9_jobs),
        "v9_new_added": len(v9_new),
        "v9_skipped": dict(v9_skipped),
        "combined_total": len(combined),
        "corrections_total": len(corrections_log),
        "corrections_by_field": dict(field_counts),
        "corrections_detail": corrections_log[:50],  # first 50 for review
        "distribution": {
            "labels": dict(label_dist),
            "oos_count": oos_count,
            "comp": dict(comp_dist),
            "sen": dict(sen_dist),
            "tech_tokens": dict(tech_dist),
        },
    }
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved build report to {report_file}")


if __name__ == "__main__":
    main()

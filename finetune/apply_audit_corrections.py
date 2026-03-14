#!/usr/bin/env python3
"""
Apply Phase 0B corrections to test set and re-run hybrid baseline.

Known corrections:
- Job 17: comp ABOVE_100K -> NO_GBP (TC = total compensation, teacher rule violation)
- Job 33: comp RANGE_55_74K -> RANGE_75_99K (£70k-£85k midpoint = £77.5k)
- Job 36: comp ABOVE_100K -> NO_GBP (TC = total compensation, teacher rule violation)
"""

import json
from copy import deepcopy
from pathlib import Path

LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_INDIVIDUAL_MAP = {"OOS": 0, "NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10}
COMP_MAP = {
    "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0,
    "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
}


def recompute(job):
    loc_score = LOCATION_MAP.get(job["loc"], 0)
    comp_score = COMP_MAP.get(job["comp"], 0)
    tech = job["tech"]
    is_oos = "OOS" in tech or len(tech) == 0
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    role_score = 0 if is_oos else SENIORITY_MAP.get(job["sen"], 0)
    raw = loc_score + role_score + tech_score + comp_score
    score = max(0, min(100, raw))
    label = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
    return score, label


CORRECTIONS = {
    17: {"field": "comp", "old": "ABOVE_100K", "new": "NO_GBP",
         "reason": "TC/total compensation is a disqualifier per teacher_v7.txt rule"},
    33: {"field": "comp", "old": "RANGE_55_74K", "new": "RANGE_75_99K",
         "reason": "£70k-£85k midpoint = £77.5k = RANGE_75_99K, not RANGE_55_74K"},
    36: {"field": "comp", "old": "ABOVE_100K", "new": "NO_GBP",
         "reason": "£150k-£200k TC = total compensation is a disqualifier per teacher_v7.txt rule"},
}


def main():
    test_file = "data/v7/test_labeled.jsonl"
    output_file = "data/v12/test_labeled_audited.jsonl"

    jobs = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))

    print(f"Loaded {len(jobs)} jobs from {test_file}")
    print(f"\nApplying {len(CORRECTIONS)} corrections:\n")

    for idx, fix in sorted(CORRECTIONS.items()):
        job = jobs[idx - 1]
        field = fix["field"]

        # Verify current value
        if job[field] != fix["old"]:
            print(f"  WARNING: Job {idx} {field} is {job[field]}, expected {fix['old']}. Skipping.")
            continue

        old_score, old_label = recompute(job)
        job[field] = fix["new"]
        new_score, new_label = recompute(job)
        job["score"] = new_score
        job["label"] = new_label

        print(f"  Job {idx}: {job['title'][:60]}")
        print(f"    {field}: {fix['old']} -> {fix['new']}")
        print(f"    score: {old_score} -> {new_score}, label: {old_label} -> {new_label}")
        print(f"    Reason: {fix['reason']}")
        print()

    # Save
    with open(output_file, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    print(f"Saved corrected test set to {output_file}")

    # Summary stats
    labels = {"good_fit": 0, "maybe": 0, "bad_fit": 0}
    for job in jobs:
        labels[job["label"]] += 1
    print(f"\nLabel distribution: {labels}")


if __name__ == "__main__":
    main()

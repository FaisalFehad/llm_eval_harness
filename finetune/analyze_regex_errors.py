#!/usr/bin/env python3
"""Analyze specific regex error patterns to calibrate Phase 1 fixes."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from deterministic_baseline_v13_1 import classify_job, classify_loc, classify_tech, classify_comp

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TEST_FILE = str(REPO / "versions/v12/data/v12_original/test_labeled_audited.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Analyze regex error patterns")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE,
                        help="Path to audited test set (default: shared V12 test set)")
    args = parser.parse_args()

    jobs = []
    with open(args.test_file) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))

    print("=" * 70)
    print("LOC ERRORS (14 total)")
    print("=" * 70)

    loc_errors = []
    for i, job in enumerate(jobs):
        pred_loc = classify_loc(job.get("job_location", ""))
        if pred_loc != job["loc"]:
            loc_errors.append((i+1, job, pred_loc))

    for idx, job, pred in sorted(loc_errors, key=lambda x: f"{x[2]}->{x[1]['loc']}"):
        loc = job.get("job_location", "?")
        jd_lower = job.get("jd_text", "").lower()
        has_remote_jd = bool(re.search(r'\bremote\b', jd_lower))
        print(f"\n  Job {idx}: golden={job['loc']} pred={pred}")
        print(f"    Location: {loc}")
        print(f"    Title: {job.get('title', '?')[:70]}")
        print(f"    JD has 'remote': {has_remote_jd}")

    print("\n" + "=" * 70)
    print("TECH ERRORS — NODE misses (golden has NODE, regex doesn't)")
    print("=" * 70)

    for i, job in enumerate(jobs):
        pred = classify_tech(job.get("jd_text", ""))
        golden = job["tech"]
        if "NODE" in golden and "NODE" not in pred:
            jd_lower = job.get("jd_text", "").lower()
            has_bare_node = bool(re.search(r'\bnode\b', jd_lower))
            contexts = []
            for m in re.finditer(r'\bnode\b', jd_lower):
                start = max(0, m.start() - 30)
                end = min(len(jd_lower), m.end() + 30)
                contexts.append(jd_lower[start:end])
            print(f"\n  Job {i+1}: golden={golden} pred={pred}")
            print(f"    Title: {job.get('title', '?')[:70]}")
            print(f"    bare 'node': {has_bare_node}")
            for ctx in contexts[:3]:
                print(f"    Context: ...{ctx}...")

    print("\n" + "=" * 70)
    print("TECH ERRORS — AI_ML false positives (regex adds AI_ML, golden doesn't)")
    print("=" * 70)

    for i, job in enumerate(jobs):
        pred = classify_tech(job.get("jd_text", ""))
        golden = job["tech"]
        if "AI_ML" in pred and "AI_ML" not in golden:
            jd_lower = job.get("jd_text", "").lower()
            ai_contexts = []
            for m in re.finditer(r'\bai\b', jd_lower):
                start = max(0, m.start() - 50)
                end = min(len(jd_lower), m.end() + 50)
                ai_contexts.append(jd_lower[start:end])
            print(f"\n  Job {i+1}: golden={golden} pred={pred}")
            print(f"    Title: {job.get('title', '?')[:70]}")
            for ctx in ai_contexts[:3]:
                print(f"    AI context: ...{ctx}...")

    print("\n" + "=" * 70)
    print("TECH ERRORS — JS_TS misses (golden has JS_TS, regex doesn't)")
    print("=" * 70)

    for i, job in enumerate(jobs):
        pred = classify_tech(job.get("jd_text", ""))
        golden = job["tech"]
        if "JS_TS" in golden and "JS_TS" not in pred:
            jd_lower = job.get("jd_text", "").lower()
            has_js = bool(re.search(r'\bjs\b', jd_lower))
            has_ts = bool(re.search(r'\bts\b', jd_lower))
            print(f"\n  Job {i+1}: golden={golden} pred={pred}")
            print(f"    Title: {job.get('title', '?')[:70]}")
            print(f"    has \\bjs\\b: {has_js}, has \\bts\\b: {has_ts}")

    print("\n" + "=" * 70)
    print("COMP ERRORS — label-flipping (sorted by impact)")
    print("=" * 70)

    COMP_MAP = {
        "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0,
        "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
    }

    for i, job in enumerate(jobs):
        pred_comp = classify_comp(job.get("jd_text", ""))
        golden_comp = job["comp"]
        if pred_comp != golden_comp:
            jd = job.get("jd_text", "")
            salary_ctx = []
            for m in re.finditer(r'.{0,50}£.{0,50}', jd):
                salary_ctx.append(m.group()[:100])
            print(f"\n  Job {i+1}: golden={golden_comp} pred={pred_comp}")
            print(f"    Title: {job.get('title', '?')[:70]}")
            for ctx in salary_ctx[:2]:
                print(f"    Salary: ...{ctx}...")


if __name__ == "__main__":
    main()
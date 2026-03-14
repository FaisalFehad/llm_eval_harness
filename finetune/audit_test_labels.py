#!/usr/bin/env python3
"""
Phase 0B: Audit test set labels for known and suspected teacher errors.

Examines:
1. The 26 hybrid error jobs — are any caused by wrong test labels?
2. The 3 known teacher errors (Jobs 17, 33, 196) from V6_DIAGNOSTIC_FINDINGS.md
3. Boundary-zone jobs (score 50-74) — most fragile, single-field errors flip labels

Outputs a corrected test set and a detailed audit report.
"""

import json
import re
import sys
from copy import deepcopy
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


def compute_score_and_label(loc, sen, tech, comp):
    loc_score = LOCATION_MAP.get(loc, 0)
    comp_score = COMP_MAP.get(comp, 0)
    is_oos = "OOS" in tech or len(tech) == 0
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    role_score = 0 if is_oos else SENIORITY_MAP.get(sen, 0)
    raw = loc_score + role_score + tech_score + comp_score
    score = max(0, min(100, raw))
    label = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
    return score, label


def audit_job(idx, job, jd_text):
    """
    Audit a single job's labels against teacher_v7.txt rules.
    Returns (corrections_dict, reasoning) or (None, "OK").
    """
    corrections = {}
    reasons = []

    title = job.get("title", "")
    location = job.get("job_location", "")
    jd_lower = jd_text.lower()
    title_lower = title.lower()

    # --- TECH audit ---
    golden_tech = job["tech"]
    # Check: does "node" / "node.js" / "nodejs" appear in JD?
    has_node = bool(re.search(r'\bnode\.?js\b|\bnodejs\b|\bnode\s+js\b|\bexpress\.?js\b|\bnest\.?js\b|\bfastify\b|\bkoa\b', jd_lower))
    has_bare_node = bool(re.search(r'\bnode\b', jd_lower))
    has_react = bool(re.search(r'\breact\b|\breactjs\b|\breact\.js\b|\bnext\.?js\b|\bnextjs\b', jd_lower))
    has_js_ts = bool(re.search(r'\btypescript\b|\bjavascript\b|\bjs\b|\bts\b|\bangular\b|\bvue\.?js\b|\bsvelte\b', jd_lower))
    has_ai_ml = bool(re.search(
        r'\bmachine\s+learning\b|\bdeep\s+learning\b|\bartificial\s+intelligence\b'
        r'|\bnatural\s+language\s+processing\b|\bcomputer\s+vision\b'
        r'|\bneural\s+network\b|\bllm\b|\blarge\s+language\s+model\b'
        r'|\btensorflow\b|\bpytorch\b|\bml\s+engineer\b'
        r'|\bfine[- ]?tuning\b|\bprompt\s+engineering\b'
        r'|\bnlp\b',
        jd_lower
    ))
    # Bare "AI" is ambiguous — check for hiring boilerplate
    has_bare_ai = bool(re.search(r'\bai\b', jd_lower))
    ai_in_boilerplate = bool(re.search(r'(equal\s+opportunity|diversity|inclusion|we\s+use\s+ai\s+to)', jd_lower))

    expected_tech = []
    if has_node or has_bare_node:
        expected_tech.append("NODE")
    if has_react:
        expected_tech.append("REACT")
    if has_js_ts:
        expected_tech.append("JS_TS")
    if has_ai_ml or (has_bare_ai and not ai_in_boilerplate):
        expected_tech.append("AI_ML")
    if not expected_tech:
        expected_tech = ["OOS"]

    # Don't flag minor tech differences — only flag if it changes the label
    if sorted(golden_tech) != sorted(expected_tech):
        # Check if the difference matters for scoring
        old_score, old_label = compute_score_and_label(job["loc"], job["sen"], golden_tech, job["comp"])
        new_score, new_label = compute_score_and_label(job["loc"], job["sen"], expected_tech, job["comp"])
        if old_label != new_label:
            reasons.append(f"TECH: golden={golden_tech} but JD text suggests {expected_tech} (label would change {old_label}->{new_label})")
            # Don't auto-correct tech — too many edge cases with bare \bnode\b etc.

    # --- COMP audit ---
    golden_comp = job["comp"]
    # Check for known patterns
    # "Total Compensation" / "TC" → NO_GBP per teacher rule
    has_tc = bool(re.search(r'\btotal\s+comp(ensation)?\b|\btc\b', jd_lower))
    has_ote = bool(re.search(r'\bote\b', jd_lower))
    has_daily_rate = bool(re.search(r'per\s+day|/day|\bday\s+rate\b|\bdaily\b', jd_lower))

    # Look for GBP salary
    gbp_range = re.search(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?\s*[-–—to]+\s*£?\s*([\d,]+(?:\.\d+)?)\s*[kK]?', jd_text)
    gbp_single = re.search(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?', jd_text)

    if has_tc or has_ote or has_daily_rate:
        if golden_comp not in ("NO_GBP",):
            reasons.append(f"COMP: has TC/OTE/daily rate but golden={golden_comp}, should be NO_GBP")

    # --- LOC audit ---
    golden_loc = job["loc"]
    loc_lower = location.lower()
    # "Anywhere" check
    if "anywhere" in loc_lower:
        if "remote" in jd_lower:
            if golden_loc != "REMOTE":
                reasons.append(f"LOC: location='Anywhere' + JD has 'remote', golden={golden_loc}, should be REMOTE")

    return reasons


def main():
    test_file = "data/v7/test_labeled.jsonl"
    output_file = "data/v12/test_labeled_audited.jsonl"
    report_file = "data/v12/audit_report.txt"

    jobs = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))

    print(f"Loaded {len(jobs)} jobs from {test_file}")

    # Known teacher errors from V6_DIAGNOSTIC_FINDINGS.md
    known_errors = {
        17: {
            "finding": "Job 17: 'Total Compensation' -> teacher labeled ABOVE_100K but rule says NO_GBP",
            "field": "comp",
            "old_value": "ABOVE_100K",
            "new_value": "NO_GBP",
        },
        33: {
            "finding": "Job 33: £70k-£85k midpoint = £77.5k = RANGE_75_99K, teacher labeled RANGE_55_74K",
            "field": "comp",
            "old_value": "RANGE_55_74K",
            "new_value": "RANGE_75_99K",
        },
        196: {
            "finding": "Job 196: JS/TS + AI explicitly named, teacher labeled OOS",
            "field": "tech",
            "old_value": ["OOS"],
            "new_value": None,  # needs manual inspection
        },
    }

    # Error job indices from Phase 0A
    error_indices = [10, 16, 17, 69, 82, 103, 108, 109, 122, 150, 156, 157, 172, 181,
                     183, 199, 204, 210, 213, 217, 218, 221, 230, 232, 239]

    # Boundary zone indices (score 50-74)
    boundary_indices = []
    for i, job in enumerate(jobs):
        score = job.get("score", 0)
        if 50 <= score <= 74:
            boundary_indices.append(i + 1)

    # All indices to audit
    audit_indices = sorted(set(error_indices + list(known_errors.keys()) + boundary_indices))

    print(f"Auditing {len(audit_indices)} jobs ({len(error_indices)} errors, {len(boundary_indices)} boundary, {len(known_errors)} known errors)")

    corrections_applied = []
    audit_findings = []

    # First pass: apply known teacher errors
    for idx, fix in known_errors.items():
        job = jobs[idx - 1]
        field = fix["field"]
        old_val = fix["old_value"]
        actual_val = job[field]

        if fix["new_value"] is None:
            # Needs manual inspection — examine JD
            jd = job.get("jd_text", "").lower()
            title = job.get("title", "").lower()
            # Job 196: check for JS/TS and AI
            has_js = bool(re.search(r'\bjavascript\b|\btypescript\b|\bjs\b|\bts\b', jd))
            has_ai = bool(re.search(r'\bai\b|\bmachine\s+learning\b|\bllm\b|\bartificial\s+intelligence\b', jd))
            if has_js and has_ai:
                fix["new_value"] = ["JS_TS", "AI_ML"]
            elif has_js:
                fix["new_value"] = ["JS_TS"]
            elif has_ai:
                fix["new_value"] = ["AI_ML"]
            else:
                audit_findings.append(f"Job {idx}: Known error but could not determine correction. Skipping.")
                continue

        if isinstance(actual_val, list):
            match = sorted(actual_val) == sorted(old_val)
        else:
            match = actual_val == old_val

        if match:
            old_score, old_label = compute_score_and_label(job["loc"], job["sen"], job["tech"], job["comp"])
            # Apply correction
            job_copy = deepcopy(job)
            job_copy[field] = fix["new_value"]
            new_score, new_label = compute_score_and_label(
                job_copy["loc"], job_copy["sen"], job_copy["tech"], job_copy["comp"]
            )
            job_copy["score"] = new_score
            job_copy["label"] = new_label

            corrections_applied.append({
                "job_index": idx,
                "title": job["title"],
                "field": field,
                "old": old_val,
                "new": fix["new_value"],
                "old_label": old_label,
                "new_label": new_label,
                "old_score": old_score,
                "new_score": new_score,
                "reason": fix["finding"],
            })
            jobs[idx - 1] = job_copy
        else:
            audit_findings.append(f"Job {idx}: Expected {field}={old_val} but found {actual_val}. Known error may already be fixed or different.")

    # Second pass: automated audit of all target jobs
    for idx in audit_indices:
        job = jobs[idx - 1]
        jd = job.get("jd_text", "")
        findings = audit_job(idx, job, jd)
        if findings:
            for f in findings:
                audit_findings.append(f"Job {idx} ({job['title'][:50]}): {f}")

    # Save corrected test set
    with open(output_file, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    # Save audit report
    with open(report_file, "w") as f:
        f.write("# V12 Phase 0B: Test Set Audit Report\n\n")
        f.write(f"## Corrections Applied ({len(corrections_applied)})\n\n")
        for c in corrections_applied:
            f.write(f"### Job {c['job_index']}: {c['title']}\n")
            f.write(f"  Field: {c['field']}\n")
            f.write(f"  Old: {c['old']} -> New: {c['new']}\n")
            f.write(f"  Label: {c['old_label']} (score {c['old_score']}) -> {c['new_label']} (score {c['new_score']})\n")
            f.write(f"  Reason: {c['reason']}\n\n")

        f.write(f"\n## Audit Findings ({len(audit_findings)})\n\n")
        for finding in audit_findings:
            f.write(f"- {finding}\n")

    print(f"\nCorrections applied: {len(corrections_applied)}")
    for c in corrections_applied:
        print(f"  Job {c['job_index']}: {c['field']} {c['old']} -> {c['new']} ({c['old_label']} -> {c['new_label']})")

    print(f"\nAudit findings: {len(audit_findings)}")
    for f in audit_findings[:20]:
        print(f"  {f}")

    print(f"\nSaved corrected test set to {output_file}")
    print(f"Saved audit report to {report_file}")


if __name__ == "__main__":
    main()

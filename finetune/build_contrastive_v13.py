"""
Build contrastive training examples for V13 fresh retrain.

Generates ~50 targeted examples by taking existing training jobs and creating
variants with modified titles, locations, and seniority labels. Focuses on
the specific error patterns found in the 0.6B model evaluation.

Usage:
    python3 finetune/build_contrastive_v13.py \
        --input data/v12/train_labeled.jsonl \
        --output data/v13/contrastive.jsonl
"""

import json
import argparse
import random
import copy
from pathlib import Path

# Scoring maps (must match semantic_tokens_v7.py)
LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_MAP = {"NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10, "OOS": 0}
COMP_MAP = {
    "NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0,
    "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25,
}


def compute_score_and_label(job: dict) -> dict:
    """Recalculate score and label from token fields."""
    loc_score = LOCATION_MAP[job["loc"]]
    is_oos = "OOS" in job["tech"] or len(job["tech"]) == 0
    role_score = 0 if is_oos else SENIORITY_MAP[job["sen"]]
    tech_score = 0 if is_oos else sum(TECH_MAP.get(t, 0) for t in job["tech"])
    comp_score = COMP_MAP[job["comp"]]
    score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
    label = "good_fit" if score >= 70 else ("maybe" if score >= 50 else "bad_fit")

    job["loc_score"] = loc_score
    job["role_score"] = role_score
    job["tech_score"] = tech_score
    job["comp_score"] = comp_score
    job["score"] = score
    job["label"] = label
    return job


def make_variant(base: dict, title: str, sen: str, job_id_suffix: str,
                 loc: str | None = None, job_location: str | None = None,
                 sen_raw: str | None = None) -> dict:
    """Create a variant of an existing job with modified title/sen/loc."""
    v = copy.deepcopy(base)
    v["job_id"] = f"contrastive_{job_id_suffix}"
    v["title"] = title
    v["sen"] = sen
    v["sen_raw"] = sen_raw or title
    v["source_file"] = "contrastive_v13"
    if loc is not None:
        v["loc"] = loc
    if job_location is not None:
        v["job_location"] = job_location
        v["loc_raw"] = job_location
    return compute_score_and_label(v)


def build_contrastive(jobs: list[dict]) -> list[dict]:
    """Build contrastive examples from existing training data."""
    random.seed(42)
    contrastive = []

    # Partition base jobs by tech for diversity
    node_jobs = [j for j in jobs if "NODE" in j["tech"] and j["sen"] == "LEVEL_2"]
    react_jobs = [j for j in jobs if "REACT" in j["tech"] and j["sen"] == "LEVEL_2"]
    oos_jobs = [j for j in jobs if "OOS" in j["tech"] and j["sen"] == "LEVEL_2"]
    l3_jobs = [j for j in jobs if j["sen"] == "LEVEL_3"]
    l1_jobs = [j for j in jobs if j["sen"] == "LEVEL_1"]
    london_jobs = [j for j in jobs if j["loc"] == "IN_LONDON"]
    uk_other_jobs = [j for j in jobs if j["loc"] == "UK_OTHER"]

    random.shuffle(node_jobs)
    random.shuffle(react_jobs)
    random.shuffle(oos_jobs)
    random.shuffle(l3_jobs)
    random.shuffle(l1_jobs)
    random.shuffle(london_jobs)
    random.shuffle(uk_other_jobs)

    idx = 0

    # ── SEN: L2 reinforcement (generic titles over-promoted to L3) ──
    # These are the most common 0.6B error: generic titles → L3
    l2_titles = [
        ("Backend Engineer", "Backend Engineer"),
        ("Software Engineer", "Software Engineer"),
        ("Full Stack Developer", "Full Stack Developer"),
        ("Data Engineer", "Data Engineer"),
        ("Platform Engineer", "Platform Engineer"),
        ("DevOps Engineer", "DevOps Engineer"),
        ("Frontend Developer", "Frontend Developer"),
        ("Cloud Engineer", "Cloud Engineer"),
        ("QA Engineer", "QA Engineer"),
        ("Systems Engineer", "Systems Engineer"),
    ]

    base_pool = (node_jobs + react_jobs + oos_jobs)[:20]
    for i, (title, sen_raw) in enumerate(l2_titles):
        if i >= len(base_pool):
            break
        v = make_variant(base_pool[i], title, "LEVEL_2", f"l2_generic_{idx}", sen_raw=sen_raw)
        contrastive.append(v)
        idx += 1

    # ── SEN: L3 reinforcement (under-promoted) ──
    l3_titles = [
        ("Staff Software Engineer", "Staff Software Engineer"),
        ("Staff Backend Engineer", "Staff Backend Engineer"),
        ("Head of Engineering", "Head of Engineering"),
        ("Head of Platform", "Head of Platform"),
        ("Head of Data", "Head of Data"),
        ("Engineering Manager", "Engineering Manager"),
        ("Engineering Manager - Backend", "Engineering Manager"),
        ("Distinguished Engineer", "Distinguished Engineer"),
        ("Principal Software Engineer", "Principal Software Engineer"),
        ("VP of Engineering", "VP of Engineering"),
    ]

    for i, (title, sen_raw) in enumerate(l3_titles):
        if i >= len(base_pool):
            break
        v = make_variant(base_pool[i], title, "LEVEL_3", f"l3_keyword_{idx}", sen_raw=sen_raw)
        contrastive.append(v)
        idx += 1

    # ── SEN: L1 reinforcement (under/over-promoted) ──
    l1_titles = [
        ("Software Engineer - Internship", "Internship"),
        ("Junior Software Engineer", "Junior"),
        ("Junior Frontend Developer", "Junior"),
        ("Graduate Software Engineer", "Graduate"),
        ("Graduate Backend Developer", "Graduate"),
        ("Associate Software Engineer", "Associate"),
        ("Associate Developer", "Associate"),
        ("Software Engineer I", "I"),
        ("Trainee Developer", "Trainee"),
        ("Apprentice Software Engineer", "Apprentice"),
    ]

    for i, (title, sen_raw) in enumerate(l1_titles):
        if i >= len(base_pool):
            break
        v = make_variant(base_pool[i], title, "LEVEL_1", f"l1_keyword_{idx}", sen_raw=sen_raw)
        contrastive.append(v)
        idx += 1

    # ── SEN: Mid-Senior → L3 ──
    mid_senior_titles = [
        ("Mid-Senior Software Engineer", "Mid-Senior"),
        ("Software Engineer (Mid/Senior)", "Mid/Senior"),
        ("Mid-Senior Backend Developer", "Mid-Senior"),
    ]

    for i, (title, sen_raw) in enumerate(mid_senior_titles):
        if i >= len(l3_jobs):
            break
        v = make_variant(l3_jobs[i], title, "LEVEL_3", f"mid_senior_{idx}", sen_raw=sen_raw)
        contrastive.append(v)
        idx += 1

    # ── SEN: L2 with senior-sounding JD text (prevent JD over-promotion) ──
    # Take L3 jobs, change title to generic, set sen=L2
    # This teaches: "Title decides — ignore experience language in description"
    jd_override_titles = [
        "Software Engineer",
        "Backend Developer",
        "Full Stack Engineer",
        "Frontend Engineer",
    ]

    for i, title in enumerate(jd_override_titles):
        if i >= len(l3_jobs):
            break
        # Take a senior job's JD but set generic title → L2
        v = make_variant(l3_jobs[i + 3], title, "LEVEL_2", f"l2_jd_override_{idx}",
                         sen_raw=title)
        contrastive.append(v)
        idx += 1

    # ── LOC: REMOTE patterns ──
    remote_patterns = [
        ("Remote (UK)", "REMOTE"),
        ("UK Remote", "REMOTE"),
        ("Remote - United Kingdom", "REMOTE"),
        ("United Kingdom (Remote)", "REMOTE"),
        ("Fully Remote", "REMOTE"),
        ("Remote", "REMOTE"),
    ]

    for i, (loc_str, loc) in enumerate(remote_patterns):
        if i >= len(uk_other_jobs):
            break
        v = make_variant(uk_other_jobs[i], uk_other_jobs[i]["title"], uk_other_jobs[i]["sen"],
                         f"loc_remote_{idx}", loc=loc, job_location=loc_str)
        contrastive.append(v)
        idx += 1

    # ── LOC: UK_OTHER bare country ──
    uk_bare_patterns = [
        ("United Kingdom", "UK_OTHER"),
        ("England", "UK_OTHER"),
        ("Scotland", "UK_OTHER"),
        ("UK", "UK_OTHER"),
    ]

    for i, (loc_str, loc) in enumerate(uk_bare_patterns):
        if i + 6 >= len(uk_other_jobs):
            break
        v = make_variant(uk_other_jobs[i + 6], uk_other_jobs[i + 6]["title"],
                         uk_other_jobs[i + 6]["sen"],
                         f"loc_uk_bare_{idx}", loc=loc, job_location=loc_str)
        contrastive.append(v)
        idx += 1

    # ── LOC: UNK (empty/N/A) ──
    unk_patterns = [
        ("", "UNK"),
        ("N/A", "UNK"),
    ]

    for i, (loc_str, loc) in enumerate(unk_patterns):
        if i + 10 >= len(uk_other_jobs):
            break
        v = make_variant(uk_other_jobs[i + 10], uk_other_jobs[i + 10]["title"],
                         uk_other_jobs[i + 10]["sen"],
                         f"loc_unk_{idx}", loc=loc, job_location=loc_str)
        contrastive.append(v)
        idx += 1

    # ── LOC: OUTSIDE_UK with Remote (should NOT be REMOTE) ──
    outside_uk_remote = [
        ("Berlin, Germany (Remote)", "OUTSIDE_UK"),
        ("New York, USA - Remote", "OUTSIDE_UK"),
        ("Remote - India", "OUTSIDE_UK"),
    ]

    for i, (loc_str, loc) in enumerate(outside_uk_remote):
        if i + 12 >= len(uk_other_jobs):
            break
        v = make_variant(uk_other_jobs[i + 12], uk_other_jobs[i + 12]["title"],
                         uk_other_jobs[i + 12]["sen"],
                         f"loc_outside_remote_{idx}", loc=loc, job_location=loc_str)
        contrastive.append(v)
        idx += 1

    return contrastive


def main():
    parser = argparse.ArgumentParser(description="Build contrastive V13 training examples")
    parser.add_argument("--input", default="data/v12/train_labeled.jsonl")
    parser.add_argument("--output", default="data/v13/contrastive.jsonl")
    args = parser.parse_args()

    with open(args.input) as f:
        jobs = [json.loads(line) for line in f]
    print(f"Loaded {len(jobs)} base training jobs")

    contrastive = build_contrastive(jobs)

    # Stats
    sen_counts = {}
    loc_counts = {}
    for c in contrastive:
        sen_counts[c["sen"]] = sen_counts.get(c["sen"], 0) + 1
        loc_counts[c["loc"]] = loc_counts.get(c["loc"], 0) + 1

    print(f"\nGenerated {len(contrastive)} contrastive examples:")
    print(f"  Sen distribution: {json.dumps(sen_counts)}")
    print(f"  Loc distribution: {json.dumps(loc_counts)}")

    # Label distribution
    label_counts = {}
    for c in contrastive:
        label_counts[c["label"]] = label_counts.get(c["label"], 0) + 1
    print(f"  Label distribution: {json.dumps(label_counts)}")

    # Write
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for c in contrastive:
            f.write(json.dumps(c) + "\n")
    print(f"\nWrote to {args.output}")

    # Show samples
    print("\n=== Sample contrastive examples ===")
    for c in contrastive[:5]:
        print(f"  {c['job_id']}: \"{c['title']}\" | sen={c['sen']} | loc={c['loc']} | label={c['label']}")


if __name__ == "__main__":
    main()

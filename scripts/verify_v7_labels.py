#!/usr/bin/env python3
"""
Verify V7 Labels — Prompt Accuracy Check
==========================================
Compares V7-labeled data against:
1. Estimated tokens (from real data heuristics)
2. Recipe targets (for generated data)
3. V5 labels (for jobs that had V5 labels)

Reports accuracy, distribution match, and suspicious cases.

V7 uses short field names: loc, arr, sen, tech, comp
Tech is an array of individual tokens: ["NODE", "REACT", "JS_TS", "AI_ML"] or ["OOS"]

Usage:
    python3 scripts/verify_v7_labels.py \
        --labeled data/v7/val_labeled.jsonl \
        --unlabeled data/v7/val_unlabeled.jsonl \
        --csv data/v7/distribution_raw.csv \
        [--v5-pool data/v5/all_labeled_pool.jsonl]
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict


# V7 score tables (same as in semantic_tokens_v7.py)
V7_LOCATION_SCORES = {
    'IN_LONDON': 25, 'REMOTE': 25, 'UK_OTHER': 10,
    'OUTSIDE_UK': -50, 'UNK': 0
}
V7_SENIORITY_SCORES = {'LEVEL_3': 25, 'LEVEL_2': 15, 'LEVEL_1': 0}
V7_TECH_INDIVIDUAL_SCORES = {
    'OOS': 0, 'NODE': 10, 'REACT': 5, 'JS_TS': 5, 'AI_ML': 10,
}
V7_COMP_SCORES = {
    'NO_GBP': 0, 'UP_TO_ONLY': 0, 'BELOW_45K': -30, 'RANGE_45_54K': 0,
    'RANGE_55_74K': 5, 'RANGE_75_99K': 15, 'ABOVE_100K': 25
}


def compute_v7_score(job: dict) -> tuple:
    """Compute V7 score and label from tokens."""
    loc_score = V7_LOCATION_SCORES.get(job.get('loc', ''), 0)

    # Tech is an array — check for OOS
    tech = job.get('tech', [])
    if isinstance(tech, str):
        tech = [tech]
    is_oos = len(tech) == 0 or 'OOS' in tech

    if is_oos:
        role_score = 0
        tech_score = 0
    else:
        role_score = V7_SENIORITY_SCORES.get(job.get('sen', ''), 0)
        tech_score = sum(V7_TECH_INDIVIDUAL_SCORES.get(t, 0) for t in tech)

    comp_score = V7_COMP_SCORES.get(job.get('comp', ''), 0)

    score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
    if score >= 70:
        label = 'good_fit'
    elif score >= 50:
        label = 'maybe'
    else:
        label = 'bad_fit'

    return score, label


def load_distribution_targets(csv_path: str) -> dict:
    targets = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row['Field'].lower()
            token = row['Token']
            targets[field][token] = int(row['Total'])
    return dict(targets)


def main():
    parser = argparse.ArgumentParser(description='Verify V7 Labels')
    parser.add_argument('--labeled', type=str, default='data/v7/val_labeled.jsonl')
    parser.add_argument('--unlabeled', type=str, default='data/v7/val_unlabeled.jsonl')
    parser.add_argument('--csv', type=str, default='data/v7/distribution_raw.csv')
    parser.add_argument('--v5-pool', type=str, default='data/v5/all_labeled_pool.jsonl')
    parser.add_argument('--base-dir', type=str,
                        default='/Users/faisal/Code/automation/ai_eval_harness')
    parser.add_argument('--show-mismatches', type=int, default=10,
                        help='Number of mismatches to show per field')
    args = parser.parse_args()

    labeled_path = os.path.join(args.base_dir, args.labeled)
    unlabeled_path = os.path.join(args.base_dir, args.unlabeled)
    csv_path = os.path.join(args.base_dir, args.csv)
    v5_path = os.path.join(args.base_dir, args.v5_pool)

    print("=" * 70)
    print("V7 LABEL VERIFICATION")
    print("=" * 70)

    # Load labeled data
    labeled = {}
    with open(labeled_path) as f:
        for line in f:
            j = json.loads(line)
            labeled[j.get('job_id', '')] = j
    print(f"  Labeled: {len(labeled)} jobs")

    # Load unlabeled data (has _v7_est and recipe_targets)
    unlabeled = {}
    with open(unlabeled_path) as f:
        for line in f:
            j = json.loads(line)
            unlabeled[j.get('job_id', '')] = j
    print(f"  Unlabeled: {len(unlabeled)} jobs")

    # Load V5 pool for comparison
    v5_map = {}
    if os.path.exists(v5_path):
        with open(v5_path) as f:
            for line in f:
                j = json.loads(line)
                v5_map[j.get('job_id', '')] = j
        print(f"  V5 pool: {len(v5_map)} jobs")

    targets = load_distribution_targets(csv_path)

    # ── 1. Token Distribution ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[1] V7 LABELED DISTRIBUTION")
    print("=" * 70)

    v7_counts = defaultdict(Counter)
    for j in labeled.values():
        for field in ['loc', 'arr', 'sen', 'comp']:
            token = j.get(field, 'MISSING')
            v7_counts[field][token] += 1
        # Tech is an array — count each individual token
        tech = j.get('tech', [])
        if isinstance(tech, str):
            tech = [tech]
        for t in tech:
            v7_counts['tech'][t] += 1

    for field in ['loc', 'arr', 'sen', 'tech', 'comp']:
        print(f"\n  {field.upper()}")
        target_pct = {t: n for t, n in targets.get(field, {}).items()}
        total_target = sum(target_pct.values()) if target_pct else 1
        for token, count in sorted(v7_counts[field].items(), key=lambda x: -x[1]):
            pct = count / len(labeled) * 100
            target_total = target_pct.get(token, 0)
            target_pct_val = target_total / total_target * 100 if total_target else 0
            print(f"    {token:<25} count={count:>4}  ({pct:>5.1f}%)  target~{target_pct_val:>5.1f}%")

    # ── 2. Label Distribution ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[2] LABEL DISTRIBUTION")
    print("=" * 70)

    label_counts = Counter()
    score_ranges = defaultdict(list)
    for j in labeled.values():
        lab = j.get('label', 'unknown')
        label_counts[lab] += 1
        score_ranges[lab].append(j.get('score', 0))

    for lab in ['good_fit', 'maybe', 'bad_fit']:
        count = label_counts.get(lab, 0)
        pct = count / len(labeled) * 100
        scores = score_ranges.get(lab, [])
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {lab:<12} {count:>4} ({pct:>5.1f}%)  avg_score={avg:.1f}")

    # ── 3. Estimate vs V7 Comparison ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("[3] ESTIMATE vs V7 LABEL COMPARISON")
    print("=" * 70)

    fields_to_compare = ['loc', 'tech', 'comp']  # Fields we estimated
    for field in fields_to_compare:
        match = 0
        mismatch = 0
        mismatches = []

        for job_id, v7_job in labeled.items():
            unlabeled_job = unlabeled.get(job_id, {})
            est = unlabeled_job.get('_v7_est', {})
            recipe = unlabeled_job.get('recipe_targets', {})

            est_token = est.get(field) or recipe.get(field)
            v7_token = v7_job.get(field)

            if not est_token or est_token == 'UNSURE':
                continue

            if est_token == v7_token:
                match += 1
            else:
                mismatch += 1
                mismatches.append({
                    'job_id': job_id,
                    'title': v7_job.get('title', '')[:50],
                    'estimated': est_token,
                    'v7_actual': v7_token,
                    'raw': v7_job.get(f'{field}_raw', ''),
                    'is_generated': 'gen_v7' in job_id,
                })

        total = match + mismatch
        pct = match / total * 100 if total else 0
        print(f"\n  {field.upper()}: {match}/{total} match ({pct:.1f}%)")
        if mismatches:
            print(f"  Top mismatches:")
            for m in mismatches[:args.show_mismatches]:
                gen_tag = " [GEN]" if m['is_generated'] else ""
                print(f"    {m['job_id']}{gen_tag}: est={m['estimated']} -> v7={m['v7_actual']}")
                print(f"      Title: {m['title']}")
                print(f"      Raw: {m['raw'][:100]}")

    # ── 4. V5 vs V7 Comparison (for jobs with V5 labels) ─────────────────
    print("\n" + "=" * 70)
    print("[4] V5 vs V7 LABEL COMPARISON (backward compatibility)")
    print("=" * 70)

    v5_v7_pairs = []
    for job_id, v7_job in labeled.items():
        v5_job = v5_map.get(job_id)
        if v5_job:
            v5_v7_pairs.append((v5_job, v7_job))

    if not v5_v7_pairs:
        print("  No overlapping jobs with V5 labels.")
    else:
        print(f"  Found {len(v5_v7_pairs)} jobs with both V5 and V7 labels\n")

        # Compare computed scores
        score_diffs = []
        label_changes = Counter()
        for v5, v7 in v5_v7_pairs:
            v5_score = v5.get('score', 0)
            v7_score = v7.get('score', 0)
            v5_label = v5.get('label', '')
            v7_label = v7.get('label', '')
            score_diffs.append(v7_score - v5_score)
            if v5_label != v7_label:
                label_changes[f"{v5_label}->{v7_label}"] += 1

        avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
        label_match = len(v5_v7_pairs) - sum(label_changes.values())
        label_pct = label_match / len(v5_v7_pairs) * 100

        print(f"  Label agreement: {label_match}/{len(v5_v7_pairs)} ({label_pct:.1f}%)")
        print(f"  Avg score difference (V7-V5): {avg_diff:+.1f}")
        if label_changes:
            print(f"  Label changes:")
            for change, count in label_changes.most_common():
                print(f"    {change}: {count}")

        # Field-level comparison (comp is directly comparable)
        for field in ['comp']:
            match = 0
            for v5, v7 in v5_v7_pairs:
                if v5.get(field) == v7.get(field):
                    match += 1
            pct = match / len(v5_v7_pairs) * 100
            print(f"  {field} agreement: {match}/{len(v5_v7_pairs)} ({pct:.1f}%)")

    # ── 5. Suspicious Cases ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[5] SUSPICIOUS CASES")
    print("=" * 70)

    suspicious = []
    for job_id, v7_job in labeled.items():
        # Check for generated jobs where recipe target doesn't match V7 label
        unlabeled_job = unlabeled.get(job_id, {})
        recipe = unlabeled_job.get('recipe_targets', {})
        if recipe:
            for field in ['loc', 'tech', 'comp']:
                recipe_token = recipe.get(field)
                v7_token = v7_job.get(field)
                if recipe_token and v7_token and recipe_token != v7_token:
                    suspicious.append({
                        'type': 'recipe_mismatch',
                        'job_id': job_id,
                        'field': field,
                        'recipe': recipe_token,
                        'v7': v7_token,
                        'title': v7_job.get('title', '')[:50],
                    })

    recipe_mismatches = [s for s in suspicious if s['type'] == 'recipe_mismatch']
    print(f"\n  Recipe mismatches (generated jobs): {len(recipe_mismatches)}")
    if recipe_mismatches:
        # Group by field
        by_field = defaultdict(list)
        for s in recipe_mismatches:
            by_field[s['field']].append(s)
        for field, cases in by_field.items():
            print(f"    {field}: {len(cases)} mismatches")
            for c in cases[:3]:
                print(f"      {c['job_id']}: recipe={c['recipe']} -> v7={c['v7']}")
                print(f"        Title: {c['title']}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

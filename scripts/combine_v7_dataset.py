#!/usr/bin/env python3
"""
Combine V7 Curated Real + Generated Jobs
=========================================
Merges the selected real jobs with generated jobs, shuffles,
and verifies the combined dataset matches V7 distribution targets.

Usage:
    python3 scripts/combine_v7_dataset.py \
        --real data/v7/curated_dataset.jsonl \
        --generated data/v7/generated_jobs.jsonl \
        --csv data/v7/distribution_raw.csv \
        --output data/v7/v7_full_dataset.jsonl
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict


def load_distribution_targets(csv_path: str) -> dict:
    """Load exact distribution targets from CSV."""
    targets = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row['Field'].lower()
            token = row['Token']
            targets[field][token] = {
                'total': int(row['Total']),
                'train': int(row['Train']),
                'val': int(row['Val']),
                'test': int(row['Test']),
            }
    return dict(targets)


def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    jobs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def verify_distribution(jobs: list, targets: dict, label: str = "Dataset"):
    """Verify dataset distribution against targets. Returns True if all within tolerance."""
    print(f"\n  {label} ({len(jobs)} jobs):")

    all_ok = True
    for field in ['location', 'tech', 'comp', 'scope', 'work_arrangement']:
        if field not in targets:
            continue

        print(f"\n  {field.upper()}")
        print(f"  {'Token':<25} {'Target':>7} {'Have':>7} {'Diff':>7} {'Status'}")
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*10}")

        for token, target_info in sorted(targets[field].items(), key=lambda x: -x[1]['total']):
            target = target_info['total']
            # Count from estimated tokens (real jobs) or recipe targets (generated)
            have = 0
            for j in jobs:
                # Check estimated tokens
                est = j.get('_v7_est', {})
                recipe = j.get('recipe_targets', {})
                job_token = est.get(field) or recipe.get(field)
                if job_token == token:
                    have += 1

            diff = have - target
            pct_diff = abs(diff) / max(target, 1) * 100
            if pct_diff <= 10:
                status = '✅'
            elif pct_diff <= 20:
                status = '⚠️ ~' + str(diff)
                all_ok = False
            else:
                status = '❌ ' + str(diff)
                all_ok = False
            print(f"  {token:<25} {target:>7} {have:>7} {diff:>+7} {status}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Combine V7 Dataset')
    parser.add_argument('--real', type=str, default='data/v7/curated_dataset.jsonl')
    parser.add_argument('--generated', type=str, default='data/v7/generated_jobs.jsonl')
    parser.add_argument('--csv', type=str, default='data/v7/distribution_raw.csv')
    parser.add_argument('--output', type=str, default='data/v7/v7_full_dataset.jsonl')
    parser.add_argument('--base-dir', type=str,
                        default='/Users/faisal/Code/automation/ai_eval_harness')
    args = parser.parse_args()

    # Resolve paths
    real_path = os.path.join(args.base_dir, args.real)
    gen_path = os.path.join(args.base_dir, args.generated)
    csv_path = os.path.join(args.base_dir, args.csv)
    output_path = os.path.join(args.base_dir, args.output)

    print("=" * 70)
    print("V7 DATASET COMBINER")
    print("=" * 70)

    # Load data
    targets = load_distribution_targets(csv_path)
    real_jobs = load_jsonl(real_path)
    gen_jobs = load_jsonl(gen_path)

    print(f"  Real jobs: {len(real_jobs)}")
    print(f"  Generated jobs: {len(gen_jobs)}")
    print(f"  Total: {len(real_jobs) + len(gen_jobs)}")

    target_total = sum(t['total'] for t in targets['location'].values())
    print(f"  Target: {target_total}")

    # For generated jobs, copy recipe_targets to _v7_est for consistent counting
    for j in gen_jobs:
        if 'recipe_targets' in j and '_v7_est' not in j:
            j['_v7_est'] = j['recipe_targets']

    # Combine
    combined = real_jobs + gen_jobs

    # Verify distribution
    print("\n[1] Distribution verification (real + generated estimated):")
    ok = verify_distribution(combined, targets)

    if ok:
        print("\n  ✅ Distribution looks good!")
    else:
        print("\n  ⚠️ Some distribution mismatches (will be refined after V7 labeling)")

    # Source breakdown
    source_counts = Counter()
    for j in combined:
        src = j.get('_source', j.get('source_file', 'unknown'))
        if 'generated' in src or 'gen_v7' in j.get('job_id', ''):
            source_counts['generated'] += 1
        elif 'synth' in src or 'variant' in src:
            source_counts['synthetic_variant'] += 1
        else:
            source_counts['real'] += 1

    print(f"\n  Source breakdown:")
    for src, count in source_counts.most_common():
        pct = count / len(combined) * 100
        print(f"    {src}: {count} ({pct:.1f}%)")

    # Shuffle and output
    random.seed(42)
    random.shuffle(combined)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for j in combined:
            # Output clean format
            output = {
                'job_id': j.get('job_id', ''),
                'title': j.get('title', ''),
                'company': j.get('company', ''),
                'location': j.get('location', ''),
                'jd_text': j.get('jd_text', ''),
                'source_file': j.get('_source', j.get('source_file', 'unknown')),
            }
            # Keep estimated tokens for analysis (will be replaced by V7 labels)
            if '_v7_est' in j:
                output['_v7_est'] = j['_v7_est']
            if 'recipe_targets' in j:
                output['recipe_targets'] = j['recipe_targets']
            if 'augmentation_type' in j:
                output['augmentation_type'] = j['augmentation_type']
            f.write(json.dumps(output) + '\n')

    print(f"\n  Wrote {len(combined)} jobs to {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

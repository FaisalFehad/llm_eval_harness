#!/usr/bin/env python3
"""
Split V7 Dataset into Train/Val/Test
=====================================
Splits the combined dataset into train (60%), val (20%), test (20%).
Uses stratified sampling based on estimated location + tech to maintain
rough distribution balance across splits.

Usage:
    python3 scripts/split_v7_dataset.py \
        --input data/v7/v7_full_dataset.jsonl \
        --train-output data/v7/train_unlabeled.jsonl \
        --val-output data/v7/val_unlabeled.jsonl \
        --test-output data/v7/test_unlabeled.jsonl
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser(description='Split V7 Dataset')
    parser.add_argument('--input', type=str, default='data/v7/v7_full_dataset.jsonl')
    parser.add_argument('--train-output', type=str, default='data/v7/train_unlabeled.jsonl')
    parser.add_argument('--val-output', type=str, default='data/v7/val_unlabeled.jsonl')
    parser.add_argument('--test-output', type=str, default='data/v7/test_unlabeled.jsonl')
    parser.add_argument('--base-dir', type=str,
                        default='/Users/faisal/Code/automation/ai_eval_harness')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Resolve paths
    input_path = os.path.join(args.base_dir, args.input)
    train_path = os.path.join(args.base_dir, args.train_output)
    val_path = os.path.join(args.base_dir, args.val_output)
    test_path = os.path.join(args.base_dir, args.test_output)

    random.seed(args.seed)

    # Load data
    jobs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    print(f"Loaded {len(jobs)} jobs from {input_path}")

    # Create stratification key from estimated location + tech
    # This ensures each split has similar distribution of these critical fields
    strata = defaultdict(list)
    for j in jobs:
        est = j.get('_v7_est', {})
        recipe = j.get('recipe_targets', {})
        loc = est.get('location') or recipe.get('location', 'UNKNOWN')
        tech = est.get('tech') or recipe.get('tech', 'NONE')
        key = f"{loc}_{tech}"
        strata[key].append(j)

    # Stratified split: 60/20/20
    train, val, test = [], [], []
    for key, group in strata.items():
        random.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * 0.2))
        n_test = max(1, round(n * 0.2))
        n_train = n - n_val - n_test

        if n < 3:
            # Very small group — put all in train
            train.extend(group)
            continue

        val.extend(group[:n_val])
        test.extend(group[n_val:n_val + n_test])
        train.extend(group[n_val + n_test:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"\nSplit results:")
    print(f"  Train: {len(train)} ({len(train)/len(jobs)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(jobs)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(jobs)*100:.1f}%)")

    # Verify no overlap (excluding empty IDs)
    train_ids = set(j.get('job_id', '') for j in train) - {''}
    val_ids = set(j.get('job_id', '') for j in val) - {''}
    test_ids = set(j.get('job_id', '') for j in test) - {''}
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv:
        print(f"  WARNING: {len(overlap_tv)} Train/Val overlapping IDs")
    if overlap_tt:
        print(f"  WARNING: {len(overlap_tt)} Train/Test overlapping IDs")
    if overlap_vt:
        print(f"  WARNING: {len(overlap_vt)} Val/Test overlapping IDs")
    if not overlap_tv and not overlap_tt and not overlap_vt:
        print("  No overlap between splits ✅")

    # Distribution check per split
    for split_name, split_data in [('Train', train), ('Val', val), ('Test', test)]:
        counts = defaultdict(Counter)
        for j in split_data:
            est = j.get('_v7_est', {})
            recipe = j.get('recipe_targets', {})
            for field in ['location', 'tech', 'comp']:
                token = est.get(field) or recipe.get(field, 'UNKNOWN')
                counts[field][token] += 1

        print(f"\n  {split_name} distribution:")
        for field in ['location', 'tech', 'comp']:
            tokens_str = ", ".join(f"{t}={c}" for t, c in sorted(counts[field].items(), key=lambda x: -x[1]))
            print(f"    {field}: {tokens_str}")

    # Write outputs
    for split_path, split_data in [(train_path, train), (val_path, val), (test_path, test)]:
        os.makedirs(os.path.dirname(split_path), exist_ok=True)
        with open(split_path, 'w') as f:
            for j in split_data:
                f.write(json.dumps(j) + '\n')
        print(f"\n  Wrote {len(split_data)} jobs to {split_path}")


if __name__ == '__main__':
    main()

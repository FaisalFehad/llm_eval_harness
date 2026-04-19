#!/usr/bin/env python3
"""
Merge rebalanced V14 training data with V15 synthetic labeled data.

Takes:
1. Rebalanced V14 data (OOS downsampled, already in chat format)
2. V15 synthetic labeled data (in standard labeled JSONL format, NOT chat format)

The synthetic data needs to be converted to chat format first using format-for-mlx-v7.ts.
This script merges two chat-format JSONL files and produces train/valid splits.

Usage:
    python3 finetune/merge_v15_data.py \
        --base data/v15/train_rebalanced.jsonl \
        --augment data/v15/synthetic_chat.jsonl \
        --output-dir data/v15/ \
        --valid-ratio 0.1 \
        --seed 42
"""

import json
import argparse
import random
from collections import Counter
from pathlib import Path


def extract_labels(messages):
    """Extract the assistant's token predictions from a training example."""
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Merge V15 training data")
    parser.add_argument("--base", required=True, help="Base rebalanced training data (chat format)")
    parser.add_argument("--augment", required=True, help="Augmentation data (chat format)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load base data
    base = []
    with open(args.base) as f:
        for line in f:
            base.append(json.loads(line))
    print(f"Base data: {len(base)} examples")

    # Load augmentation data
    augment = []
    with open(args.augment) as f:
        for line in f:
            augment.append(json.loads(line))
    print(f"Augmentation data: {len(augment)} examples")

    # Merge
    all_data = base + augment
    print(f"Merged total: {len(all_data)} examples")

    # Shuffle
    random.shuffle(all_data)

    # Split into train/valid
    n_valid = max(1, int(len(all_data) * args.valid_ratio))
    valid_data = all_data[:n_valid]
    train_data = all_data[n_valid:]

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Distribution report
    print(f"\n--- Distribution Report ---")
    for split_name, data in [("TRAIN", train_data), ("VALID", valid_data)]:
        tech_dist = Counter()
        loc_dist = Counter()
        comp_dist = Counter()

        for ex in data:
            labels = extract_labels(ex["messages"])
            if labels:
                tech = labels.get("tech", [])
                if isinstance(tech, list):
                    tech_dist[tuple(sorted(tech))] += 1
                loc_dist[labels.get("loc", "?")] += 1
                comp_dist[labels.get("comp", "?")] += 1

        total = len(data)
        oos_count = tech_dist.get(("OOS",), 0)
        remote_count = loc_dist.get("REMOTE", 0)

        print(f"\n{split_name} ({total} examples):")
        print(f"  OOS: {oos_count} ({oos_count/total*100:.1f}%)")
        print(f"  loc=REMOTE: {remote_count} ({remote_count/total*100:.1f}%)")
        print(f"  Top 5 tech combos:")
        for combo, count in tech_dist.most_common(5):
            print(f"    {str(list(combo)):45s} {count:4d} ({count/total*100:.1f}%)")

    print(f"\nOutput:")
    print(f"  {train_path}")
    print(f"  {valid_path}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()

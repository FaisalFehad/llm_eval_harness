#!/usr/bin/env python3
"""
V12 Phase 1.5B + 2C: Preprocess training data for MLX formatting.

1. Apply JD preprocessing (html entities, whitespace, boilerplate removal)
2. Check for V9 _raw field truncation (50-char fingerprint)
3. Output cleaned data ready for format-for-mlx-v7.ts

Usage:
    python3 finetune/preprocess_v12_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_jd import preprocess_jd


def main():
    input_file = "data/v12/train_labeled.jsonl"
    output_file = "data/v12/train_labeled_preprocessed.jsonl"

    jobs = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))

    print(f"Loaded {len(jobs)} jobs from {input_file}")

    # Check for 50-char _raw truncation
    raw_fields = ["loc_raw", "arr_raw", "sen_raw", "tech_raw", "comp_raw"]
    truncation_count = 0
    for job in jobs:
        for field in raw_fields:
            val = job.get(field)
            if val is not None and isinstance(val, str) and len(val) == 50:
                truncation_count += 1
                # Don't fix — just flag. The _raw fields are from the teacher
                # and may legitimately be 50 chars. But flag if there are many.

    if truncation_count > 20:
        print(f"  WARNING: {truncation_count} _raw fields are exactly 50 chars (possible V9 truncation)")
    else:
        print(f"  _raw field 50-char check: {truncation_count} (OK)")

    # Apply JD preprocessing
    preprocessed_count = 0
    total_reduction = 0
    for job in jobs:
        original_len = len(job.get("jd_text", ""))
        job["jd_text"] = preprocess_jd(job.get("jd_text", ""))
        new_len = len(job["jd_text"])
        if new_len < original_len:
            preprocessed_count += 1
            total_reduction += original_len - new_len

    pct = 100 * preprocessed_count / len(jobs) if jobs else 0
    avg_reduction = total_reduction / preprocessed_count if preprocessed_count else 0
    print(f"  Preprocessed: {preprocessed_count}/{len(jobs)} jobs ({pct:.0f}%)")
    print(f"  Avg reduction: {avg_reduction:.0f} chars per affected job")

    # Save
    with open(output_file, "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge all V16 teacher label batches into single file."""
import json, glob
from pathlib import Path

OUTDIR = Path("versions/v16/data/full_relabel_v3")
OUTPUT = Path("versions/v16/data/v16_teacher_labels_v3.jsonl")

def merge():
    # Load all done indices and their data
    jobs = {}
    # Prefer full_relabel (has latest), fall back to ollama_100
    for pattern in [OUTDIR / "batch_*.jsonl", Path("versions/v16/data/ollama_100") / "batch_*.jsonl"]:
        for path in sorted(glob.glob(str(pattern))):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        idx = d.get("index")
                        if idx is not None and idx not in jobs:
                            jobs[idx] = d
    print(f"Loaded {len(jobs)} jobs")

    # Merge with original job data for full context
    merged_count = 0
    with open(OUTPUT, "w") as out_f, open("versions/v16/data/v15_jobs_extracted.jsonl") as orig_f:
        for line in orig_f:
            if line.strip():
                orig = json.loads(line)
                idx = orig.get("index")
                if idx in jobs:
                    merged = dict(orig)
                    # Add v16 labels
                    for field in ["loc_raw","loc","arr_raw","arr","sen_raw","sen","tech_raw","tech","comp_raw","comp"]:
                        merged[f"v16_{field}"] = jobs[idx].get(f"v16_{field}")
                    out_f.write(json.dumps(merged) + "\n")
                    merged_count += 1

    print(f"Wrote {merged_count} merged jobs to {OUTPUT}")

if __name__ == "__main__":
    merge()

#!/usr/bin/env python3
"""
Robust full V16 relabeling — runs multiple single-process batches in sequence
with progress tracking and resume capability.
Uses whichever backend is available (Ollama or OMLX).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROMPT = "versions/v16/prompts/teacher.txt"
PREFERRED_BACKENDS = [
    ("ollama", "gemma4:31b-cloud", "http://localhost:11434/v1", ""),
    ("ollama", "gemma4:31b-cloud", "http://localhost:11434/v1", ""),  # second worker if Ollama handles parallel
    ("omlx", "Qwen3.6-35B-A3B-oQ5", "http://127.0.0.1:8000/v1", ""),
]

def load_all_jobs():
    with open("versions/v16/data/v15_jobs_extracted.jsonl") as f:
        return [json.loads(line) for line in f]

def load_done_indices():
    done = set()
    # Check existing olama 100
    for p in [
        "versions/v16/data/ollama_100/ollama_100_merged.jsonl",
        "versions/v16/data/pilot_100/ollama_100_merged.jsonl",
        "versions/v16/data/full_relabel/batch_ollama_0.jsonl",
        "versions/v16/data/full_relabel/batch_ollama_1.jsonl",
        "versions/v16/data/full_relabel/batch_omlx_0.jsonl",
        "versions/v16/data/full_relabel/batch_omlx_1.jsonl",
    ]:
        if Path(p).exists():
            with open(p) as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        done.add(j.get("index"))
                    except:
                        pass
    return done

def run_single_batch(jobs, name, backend, model, url, api_key=""):
    """Run one batch through relabel_for_drift_v16.py."""
    output_dir = Path("versions/v16/data/full_relabel")
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"batch_{name}.jsonl"
    
    # Skip if already has all jobs
    existing = sum(1 for _ in open(out)) if out.exists() else 0
    if existing >= len(jobs):
        print(f"[{name}] Already complete ({existing}/{len(jobs)})")
        return name, existing, True
    
    # Filter to jobs not yet in this batch
    if out.exists() and existing > 0:
        with open(out) as f:
            done_in_batch = set(json.loads(line).get("index") for line in f)
        jobs = [j for j in jobs if j.get("index") not in done_in_batch]
    
    if not jobs:
        print(f"[{name}] No remaining jobs")
        return name, 0, True
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tf:
        for j in jobs:
            tf.write(json.dumps(j) + "\n")
        temp_in = tf.name
    
    cmd = [
        sys.executable,
        "finetune/relabel_for_drift_v16.py",
        "--input", temp_in,
        "--output", str(out),
        "--prompt", PROMPT,
        "--base-url", url,
        "--api-key", api_key,
        "--model", model,
    ]
    
    print(f"[{name}] Starting {len(jobs)} jobs via {backend} ({model})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    os.unlink(temp_in)
    
    # Count output
    count = sum(1 for _ in open(out)) if out.exists() else 0
    success = result.returncode == 0
    
    if not success:
        print(f"[{name}] FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:300]}")
    else:
        print(f"[{name}] Complete: {count} jobs")
    
    return name, count, success

def merge_outputs(output_path="versions/v16/data/v16_full_relabeled.jsonl"):
    """Merge all batch outputs into master file with deduplication."""
    seen = set()
    total = 0
    
    with open(output_path, "w") as out_f:
        # Include existing pilot/ollama data
        for source in [
            "versions/v16/data/ollama_100/ollama_100_merged.jsonl",
            "versions/v16/data/pilot_100/ollama_100_merged.jsonl",
        ]:
            if Path(source).exists():
                with open(source) as f:
                    for line in f:
                        try:
                            j = json.loads(line)
                            idx = j.get("index")
                            if idx not in seen:
                                seen.add(idx)
                                out_f.write(line)
                                total += 1
                        except:
                            pass
        
        # Include full_relabel batches
        for batch in Path("versions/v16/data/full_relabel").glob("batch_*.jsonl"):
            with open(batch) as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        idx = j.get("index")
                        if idx not in seen:
                            seen.add(idx)
                            out_f.write(line)
                            total += 1
                    except:
                        pass
    
    return total

def main():
    all_jobs = load_all_jobs()
    done = load_done_indices()
    remaining = [j for j in all_jobs if j.get("index") not in done]
    
    print(f"Total jobs: {len(all_jobs)}")
    print(f"Already done: {len(done)}")
    print(f"Remaining: {len(remaining)}")
    
    if len(remaining) == 0:
        print("Nothing to do — merging existing outputs...")
        total = merge_outputs()
        print(f"Done. Master file has {total} total jobs.")
        return
    
    # For simplicity: run remaining in chunks of 25 using a single backend at a time
    # This avoids parallelism complexity while still being fast enough
    chunk_size = 25
    chunks = [remaining[i:i+chunk_size] for i in range(0, len(remaining), chunk_size)]
    
    print(f"\nProcessing {len(chunks)} chunks of up to {chunk_size} jobs each...")
    
    total_done = 0
    for i, chunk in enumerate(chunks):
        backend, model, url, key = PREFERRED_BACKENDS[i % len(PREFERRED_BACKENDS)]
        name = f"{backend}_{i}"
        _, count, _ = run_single_batch(chunk, name, backend, model, url, key)
        total_done += count
        print(f"  Progress: {total_done}/{len(remaining)} done")
        
        # Save intermediate merge every 5 chunks
        if (i + 1) % 5 == 0:
            total = merge_outputs()
            print(f"  [SYNC] Master file now has {total} total jobs")
    
    total = merge_outputs()
    print(f"\n{'='*60}")
    print(f"FINAL: Master file has {total} jobs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

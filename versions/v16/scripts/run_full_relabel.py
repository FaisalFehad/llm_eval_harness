#!/usr/bin/env python3
"""
Generate ALL remaining V16 labels using 2×Ollama + 2×OMLX in parallel.
Tracks completed jobs by index to prevent duplicates.
"""
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_JOBS = "versions/v16/data/v15_jobs_extracted.jsonl"
DONE_FILE = "versions/v16/data/ollama_100/ollama_100_merged.jsonl"
PROMPT = "versions/v16/prompts/teacher.txt"
OLLAMA_MODEL = "gemma4:31b-cloud"
OMLX_MODEL = "Qwen3.6-35B-A3B-oQ5"
OLLAMA_URL = "http://localhost:11434/v1"
OMLX_URL = "http://127.0.0.1:8000/v1"
OUTPUT_DIR = "versions/v16/data/full_relabel"

def load_done_ids():
    """Load indices of already-completed jobs."""
    done = set()
    if Path(DONE_FILE).exists():
        with open(DONE_FILE) as f:
            for line in f:
                j = json.loads(line)
                done.add(j.get("index", j.get("job_id")))
    return done

def run_worker(name, jobs, backend, model, url, api_key=""):
    """Run one batch through a specific backend."""
    if not jobs:
        print(f"[{name}] No jobs assigned — skipping")
        return name, 0
    
    # Write temp input
    temp_in = f"/tmp/v16_batch_{name}.jsonl"
    out = Path(OUTPUT_DIR) / f"batch_{name}.jsonl"
    with open(temp_in, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    
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
    
    print(f"[{name}] Starting {len(jobs)} jobs → {out}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    count = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"[{name}] Finished: {count}/{len(jobs)} jobs (rc={result.returncode})")
    if result.returncode != 0:
        print(f"[{name}] stderr: {result.stderr[:300]}")
    
    return name, count

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load all jobs
    with open(INPUT_JOBS) as f:
        all_jobs = [json.loads(line) for line in f]
    
    # Load already-done indices
    done_ids = load_done_ids()
    print(f"Total jobs in source: {len(all_jobs)}")
    print(f"Already completed: {len(done_ids)}")
    
    # Filter remaining
    remaining = [j for j in all_jobs if j.get("index") not in done_ids]
    print(f"Remaining to label: {len(remaining)}")
    
    if len(remaining) == 0:
        print("Everything already done!")
        return
    
    # Split remaining into 4 batches: ollama_0, ollama_1, omlx_0, omlx_1
    n = len(remaining) // 4
    batches = [
        ("ollama_0", remaining[:n]),
        ("ollama_1", remaining[n:2*n]),
        ("omlx_0", remaining[2*n:3*n]),
        ("omlx_1", remaining[3*n:]),
    ]
    
    print(f"\nBatch distribution:")
    for name, jobs in batches:
        print(f"  {name}: {len(jobs)} jobs")
    
    # Launch all 4 in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, jobs in batches:
            if name.startswith("ollama"):
                future = executor.submit(run_worker, name, jobs, "ollama", OLLAMA_MODEL, OLLAMA_URL, "")
            else:
                future = executor.submit(run_worker, name, jobs, "omlx", OMLX_MODEL, OMLX_URL, "")
            futures[future] = name
        
        for future in as_completed(futures):
            name, count = future.result()
            results[name] = count
    
    # Merge all outputs into final master file
    master = Path(OUTPUT_DIR) / "v16_full_relabeled.jsonl"
    total = 0
    with open(master, "w") as out_f:
        for name, _ in batches:
            batch_path = Path(OUTPUT_DIR) / f"batch_{name}.jsonl"
            if batch_path.exists():
                with open(batch_path) as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total += 1
    
    # Also append existing 100 to master
    if Path(DONE_FILE).exists():
        with open(DONE_FILE) as f:
            existing = sum(1 for _ in f)
    else:
        existing = 0
    
    print(f"\n{'='*60}")
    print(f"Results: {results}")
    print(f"New jobs written: {total}")
    print(f"Existing jobs: {existing}")
    print(f"Master file: {master} (now has {total} new + {existing} existing = {total+existing} total)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

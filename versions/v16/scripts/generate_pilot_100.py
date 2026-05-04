#!/usr/bin/env python3
"""
Generate 100 NEW labeled jobs in 4 parallel batches using V16 teacher prompt.
Excludes the 42 already-validated jobs.
"""
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
BATCH_SIZE = 25  # 100 jobs / 4 batches
TOTAL_JOBS = 100
PROMPT_PATH = "versions/v16/prompts/teacher.txt"
MODEL = "gemma-4-31B-it-oQ4"

def run_batch(batch_idx, jobs, output_dir, api_key, model):
    """Run a single batch of jobs through OMLX API."""
    import subprocess
    
    # Write temp input file
    temp_input = f"/tmp/v16_batch_{batch_idx}_input.jsonl"
    output_path = os.path.join(output_dir, f"batch_{batch_idx}.jsonl")
    
    with open(temp_input, 'w') as f:
        for job in jobs:
            f.write(json.dumps(job) + '\n')
    
    cmd = [
        sys.executable, 
        "finetune/relabel_for_drift_v16.py",
        "--input", temp_input,
        "--output", output_path,
        "--prompt", PROMPT_PATH,
        "--api-key", api_key,
        "--model", model,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        'batch': batch_idx,
        'returncode': result.returncode,
        'stdout': result.stdout[-500:] if result.stdout else "",
        'stderr': result.stderr[-500:] if result.stderr else "",
        'output': output_path,
        'jobs': len(jobs)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="versions/v16/data/v15_jobs_extracted.jsonl")
    parser.add_argument("--exclude", default="versions/v16/data/sample_50_high_risk.jsonl")
    parser.add_argument("--output-dir", default="versions/v16/data/pilot_100")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gemma-4-31B-it-oQ4")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    # Load all jobs
    with open(args.input) as f:
        all_jobs = [json.loads(line) for line in f]
    
    # Load excluded job IDs
    excluded_ids = set()
    if Path(args.exclude).exists():
        with open(args.exclude) as f:
            for line in f:
                job = json.loads(line)
                excluded_ids.add(job.get('job_id', job.get('index')))
    
    # Filter out already-validated jobs
    available_jobs = [j for j in all_jobs if j.get('job_id', j.get('index')) not in excluded_ids]
    
    # Take first 100
    jobs_to_label = available_jobs[:TOTAL_JOBS]
    print(f"Total jobs: {len(all_jobs)}")
    print(f"Excluded: {len(excluded_ids)}")
    print(f"Available: {len(available_jobs)}")
    print(f"Selected: {len(jobs_to_label)}")
    
    if len(jobs_to_label) < TOTAL_JOBS:
        print(f"WARNING: Only {len(jobs_to_label)} jobs available, expected {TOTAL_JOBS}")
    
    # Split into batches
    actual_batch_size = max(1, len(jobs_to_label) // args.workers)
    batches = [jobs_to_label[i:i+actual_batch_size] for i in range(0, len(jobs_to_label), actual_batch_size)]
    
    # Ensure output dir exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run in parallel
    print(f"\nStarting {len(batches)} parallel batches with {args.workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, batch in enumerate(batches):
            future = executor.submit(run_batch, i, batch, str(output_dir), args.api_key, args.model)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "✅" if result['returncode'] == 0 else "❌"
            print(f"\n  {status} Batch {result['batch']}: {result['jobs']} jobs (rc={result['returncode']})")
            if result['returncode'] != 0:
                print(f"     stderr: {result['stderr'][:300]}")
                print(f"     stdout: {result['stdout'][:300]}")
    
    # Merge outputs
    merged_path = output_dir / "pilot_100_merged.jsonl"
    total_written = 0
    with open(merged_path, 'w') as out_f:
        for i in range(len(batches)):
            batch_path = output_dir / f"batch_{i}.jsonl"
            if batch_path.exists():
                with open(batch_path) as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_written += 1
    
    print(f"\n{'='*60}")
    print(f"Merged {total_written} jobs to {merged_path}")
    print(f"Individual batches in {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

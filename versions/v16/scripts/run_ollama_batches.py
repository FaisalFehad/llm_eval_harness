#!/usr/bin/env python3
"""
Run 2 parallel batch labeling jobs via Ollama gemma4:31b-cloud.
Splits V15 jobs into 2×50 batches.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROMPT = "versions/v16/prompts/teacher.txt"
MODEL = "gemma4:31b-cloud"
API_URL = "http://localhost:11434/v1"

def run_ollama_batch(batch_idx, input_path, output_path, api_key=""):
    cmd = [
        sys.executable,
        "finetune/relabel_for_drift_v16.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--prompt", PROMPT,
        "--base-url", API_URL,
        "--api-key", api_key,
        "--model", MODEL,
    ]
    print(f"[BATCH {batch_idx}] Starting: {input_path} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[BATCH {batch_idx}] Done (rc={result.returncode})")
    if result.returncode != 0:
        print(f"  stderr: {result.stderr[:300]}")
    return result

def main():
    # Load all jobs, exclude the 42 already validated
    with open("versions/v16/data/v15_jobs_extracted.jsonl") as f:
        all_jobs = [json.loads(line) for line in f]
    
    with open("versions/v16/data/sample_50_high_risk.jsonl") as f:
        excluded = set(json.loads(line).get('job_id', json.loads(line).get('index')) for line in f)
    
    available = [j for j in all_jobs if j.get('job_id', j.get('index')) not in excluded]
    selected = available[:100]
    
    print(f"Total: {len(all_jobs)}, Excluded: {len(excluded)}, Available: {len(available)}, Selected: {len(selected)}")
    
    # Split into 2 batches of 50
    mid = len(selected) // 2
    batches = [
        (0, selected[:mid]),
        (1, selected[mid:]),
    ]
    
    output_dir = Path("versions/v16/data/ollama_100")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write temp inputs and launch both
    procs = []
    for idx, jobs in batches:
        temp_in = f"/tmp/ollama_batch_{idx}.jsonl"
        out = output_dir / f"batch_{idx}.jsonl"
        with open(temp_in, "w") as f:
            for j in jobs:
                f.write(json.dumps(j) + "\n")
        
        proc = subprocess.Popen([
            sys.executable,
            "finetune/relabel_for_drift_v16.py",
            "--input", temp_in,
            "--output", str(out),
            "--prompt", PROMPT,
            "--base-url", API_URL,
            "--api-key", "",
            "--model", MODEL,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((idx, proc, out))
        print(f"Launched batch {idx}: {len(jobs)} jobs → {out}")
    
    # Wait for both
    for idx, proc, out in procs:
        stdout, stderr = proc.communicate(timeout=1800)
        print(f"\n[BATCH {idx}] Finished (rc={proc.returncode})")
        if proc.returncode != 0:
            print(f"  stderr: {stderr[:400]}")
        else:
            count = sum(1 for _ in open(out)) if out.exists() else 0
            print(f"  Output: {count} jobs")
    
    # Merge
    merged = output_dir / "ollama_100_merged.jsonl"
    total = 0
    with open(merged, "w") as out_f:
        for idx, _, out in procs:
            if out.exists():
                with open(out) as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total += 1
    print(f"\n{'='*60}")
    print(f"Merged {total} jobs into {merged}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate V16 teacher v2 labels for bad jobs — OMLX only."""
import glob, json, os, subprocess, sys, time
from pathlib import Path

OUTDIR = "versions/v16/data/full_relabel_v2"
PROMPT = "versions/v16/prompts/teacher_v2.txt"
API_KEY = "fbe900938b6dedf7eb6994a03054ebbe1654b7d6788c35fdccde7568c8c70e76f300f3faebec1ff7e7066b45a07b7920d406a445682c9019ff71f9cae4d07c17ee473c670c2442d44b04d1b3f4f74f650453d0e0b7743567a525cd3c34f7df5553c64f96fb940b12ab15a8b5d094b5a53dbf7daf9a81fef711b55f7ff2b07c35"

WORKERS = [
    {"name": "omlx_regen_0", "url": "http://127.0.0.1:8000/v1", "key": API_KEY, "model": "gemma-4-31B-it"},
    {"name": "omlx_regen_1", "url": "http://127.0.0.1:8000/v1", "key": API_KEY, "model": "gemma-4-31B-it"},
    {"name": "omlx_regen_2", "url": "http://127.0.0.1:8000/v1", "key": API_KEY, "model": "gemma-4-31B-it"},
    {"name": "omlx_regen_3", "url": "http://127.0.0.1:8000/v1", "key": API_KEY, "model": "gemma-4-31B-it"},
]


def relabel_worker(worker, jobs):
    name = worker["name"]
    out_path = Path(OUTDIR) / f"batch_{name}.jsonl"
    log_path = Path(OUTDIR) / f"log_{name}.txt"
    tmp_in = f"/tmp/v16_v2_regen_{name}.jsonl"

    with open(tmp_in, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")

    existing = sum(1 for _ in open(out_path)) if out_path.exists() else 0

    cmd = [
        sys.executable, "-u",
        "finetune/relabel_for_drift_v16.py",
        "--input", tmp_in,
        "--output", str(out_path),
        "--prompt", PROMPT,
        "--base-url", worker["url"],
        "--api-key", worker["key"],
        "--model", worker["model"],
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"\n[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs (existing: {existing})\n")
    log_f.flush()

    # Tee output to both terminal and log file
    tee = subprocess.Popen(
        ["tee", "-a", str(log_path)],
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    proc = subprocess.Popen(cmd, stdout=tee.stdin, stderr=subprocess.STDOUT, text=True, env=env)
    return name, proc, tee, existing, len(jobs)


def monitor(workers_info):
    start = time.time()
    while True:
        time.sleep(30)
        elapsed = (time.time() - start) / 60
        header = f"{'Worker':<18} {'Done':>6} {'Total':>6} {'%':>6} {'Jobs/min':>10} {'Status':>10}"
        print(f"\n{header}", flush=True)
        print("-" * len(header), flush=True)
        all_done = True
        for name, _, initial, total in workers_info:
            out = Path(OUTDIR) / f"batch_{name}.jsonl"
            current = sum(1 for _ in open(out)) if out.exists() else 0
            done = max(0, current - initial)
            status = "FINISHED" if current >= initial + total else "RUNNING"
            if status == "RUNNING":
                all_done = False
            print(f"{name:<18} {done:>6} {total:>6} {(done/total*100) if total else 0:>6.1f} {(done/elapsed) if elapsed else 0:>10.2f} {status:>10}", flush=True)
        print(flush=True)
        if all_done:
            print(f"All workers finished!")
            break


def main():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    with open("/tmp/v16_v2_bad_jobs.jsonl") as f:
        bad_jobs = [json.loads(l) for l in f]

    if not bad_jobs:
        print("Nothing left.", flush=True)
        return

    print(f"Regenerating {len(bad_jobs)} bad jobs via OMLX...", flush=True)

    n = len(bad_jobs) // 4
    slices = [bad_jobs[:n], bad_jobs[n:2*n], bad_jobs[2*n:3*n], bad_jobs[3*n:]]

    workers_info = []
    tees = []
    for worker, jobs in zip(WORKERS, slices):
        name, proc, tee, existing, total = relabel_worker(worker, jobs)
        workers_info.append((name, proc, existing, total))
        tees.append(tee)
        print(f"  [{name}] {total} jobs", flush=True)

    try:
        monitor(workers_info)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        return

    print("\nWaiting for remaining workers...", flush=True)
    for name, proc, _, _ in workers_info:
        proc.wait()
        print(f"  [{name}] exit {proc.returncode}", flush=True)

    for tee in tees:
        tee.stdin.close()
        tee.wait(timeout=5)

    final_count = sum(1 for w in WORKERS for _ in open(Path(OUTDIR) / f"batch_{w['name']}.jsonl") if (Path(OUTDIR) / f"batch_{w['name']}.jsonl").exists())
    print(f"\n{'='*60}")
    print(f"Regen done. Total new jobs: {final_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

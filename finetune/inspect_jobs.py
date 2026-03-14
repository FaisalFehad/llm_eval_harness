#!/usr/bin/env python3
"""Inspect specific jobs from the test set for Phase 0B audit."""
import json
import re
import sys

jobs = []
with open("data/v7/test_labeled.jsonl") as f:
    for line in f:
        if line.strip():
            jobs.append(json.loads(line))

indices = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [36, 141, 156, 196]

for idx in indices:
    job = jobs[idx - 1]
    print(f"=== JOB {idx}: {job['title']} ===")
    print(f"  Location: {job.get('job_location', '?')}")
    print(f"  Label: {job['label']} Score: {job.get('score', '?')}")
    print(f"  loc={job['loc']} arr={job['arr']} sen={job['sen']} tech={job['tech']} comp={job['comp']}")
    print(f"  comp_raw: {job.get('comp_raw', '?')}")
    print(f"  tech_raw: {job.get('tech_raw', '?')}")

    jd = job.get("jd_text", "")

    # Find salary/comp context
    for m in re.finditer(r'.{0,80}(£|\btc\b|\btotal\s+comp|\bote\b|\bper\s+day|\bdaily\s+rate)', jd, re.IGNORECASE):
        ctx = jd[m.start():min(m.end()+80, len(jd))]
        print(f"  COMP CONTEXT: ...{ctx[:160]}...")

    # For job 196 and others: find tech mentions
    for keyword in ["javascript", "typescript", "node", "react", "ai ", "machine learning", "llm"]:
        positions = [m.start() for m in re.finditer(re.escape(keyword), jd.lower())]
        for pos in positions[:2]:
            start = max(0, pos - 40)
            end = min(len(jd), pos + len(keyword) + 60)
            print(f"  TECH [{keyword.strip()}]: ...{jd[start:end]}...")

    print()

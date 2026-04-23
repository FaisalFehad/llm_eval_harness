"""
Experiment: Does chain-of-thought reasoning improve GPT-4o-mini's token accuracy?

Labels the same 200 jobs twice:
  Run A: V6 teacher prompt (8-field, with reasoning)
  Run B: V6 no-reasoning prompt (4-field, tokens only)

Compares token-by-token (loc, role, tech, comp) and reports match rates.

Usage:
  OPENAI_API_KEY=sk-... python3 scripts/experiment_reasoning_comparison.py

Cost estimate: ~200 × 2 calls × ~1500 tokens = ~600K tokens ≈ $0.10
"""

import json
import os
import sys
import asyncio
import time
from pathlib import Path

# ── Check API key ────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: Set OPENAI_API_KEY environment variable")
    sys.exit(1)

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────
SAMPLE_FILE = "data/v6/experiment/sample_100.jsonl"
PROMPT_WITH_REASONING = "prompts/teacher_v6.txt"
PROMPT_NO_REASONING = "prompts/teacher_v6_no_reasoning.txt"
OUTPUT_DIR = "data/v6/experiment"
MODEL = "gpt-4o-mini"
CONCURRENCY = 10
TEMPERATURE = 0

# ── Load data ────────────────────────────────────────────────────────
def load_jsonl(path):
    jobs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))
    return jobs

def load_prompt(path):
    with open(path) as f:
        return f.read()

# ── API calls ────────────────────────────────────────────────────────
async def label_job(client, semaphore, prompt_template, job, index, total):
    """Label a single job with the given prompt template."""
    prompt_text = (
        prompt_template
        .replace("{{job_title}}", job.get("title", ""))
        .replace("{{job_location}}", job.get("location", ""))
        .replace("{{jd_text}}", job.get("jd_text", ""))
    )

    async with semaphore:
        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "Respond with JSON only."},
                        {"role": "user", "content": prompt_text},
                    ],
                    max_tokens=500,
                    temperature=TEMPERATURE,
                )
                content = response.choices[0].message.content or ""
                usage = response.usage

                # Parse JSON
                content = content.strip()
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Try extracting from markdown
                    import re
                    match = re.search(r'\{[\s\S]*\}', content)
                    if match:
                        parsed = json.loads(match.group(0))
                    else:
                        return {
                            "job_id": job["job_id"],
                            "error": "JSON parse failure",
                            "raw": content[:300],
                            "prompt_tokens": usage.prompt_tokens if usage else 0,
                            "completion_tokens": usage.completion_tokens if usage else 0,
                        }

                return {
                    "job_id": job["job_id"],
                    "loc": parsed.get("loc", ""),
                    "role": parsed.get("role", ""),
                    "tech": parsed.get("tech", ""),
                    "comp": parsed.get("comp", ""),
                    "loc_reason": parsed.get("loc_reason", ""),
                    "role_reason": parsed.get("role_reason", ""),
                    "tech_reason": parsed.get("tech_reason", ""),
                    "comp_reason": parsed.get("comp_reason", ""),
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }

            except Exception as e:
                msg = str(e)
                if "429" in msg and attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                return {
                    "job_id": job["job_id"],
                    "error": f"API error: {msg}",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

    return {"job_id": job["job_id"], "error": "exhausted retries"}


async def label_batch(client, prompt_template, jobs, label):
    """Label all jobs with the given prompt, with progress tracking."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    total = len(jobs)
    completed = 0
    start = time.time()

    async def tracked_label(job, i):
        nonlocal completed
        result = await label_job(client, semaphore, prompt_template, job, i, total)
        completed += 1
        if completed % 50 == 0 or completed == total:
            elapsed = time.time() - start
            print(f"  [{label}] {completed}/{total} ({elapsed:.1f}s)")
        return result

    tasks = [tracked_label(job, i) for i, job in enumerate(jobs)]
    return await asyncio.gather(*tasks)


# ── Comparison ───────────────────────────────────────────────────────
def compare_results(results_a, results_b, jobs):
    """Compare token-by-token between Run A (reasoning) and Run B (no reasoning)."""
    # Index by job_id
    a_by_id = {r["job_id"]: r for r in results_a}
    b_by_id = {r["job_id"]: r for r in results_b}

    fields = ["loc", "role", "tech", "comp"]
    field_match = {f: 0 for f in fields}
    field_total = {f: 0 for f in fields}
    all_match = 0
    total = 0
    errors_a = 0
    errors_b = 0
    mismatches = []

    for job in jobs:
        jid = job["job_id"]
        a = a_by_id.get(jid)
        b = b_by_id.get(jid)

        if not a or not b:
            continue

        if "error" in a:
            errors_a += 1
            continue
        if "error" in b:
            errors_b += 1
            continue

        total += 1
        job_matches = True

        for f in fields:
            field_total[f] += 1
            if a[f] == b[f]:
                field_match[f] += 1
            else:
                job_matches = False
                mismatches.append({
                    "job_id": jid,
                    "title": job["title"],
                    "field": f,
                    "with_reasoning": a[f],
                    "no_reasoning": b[f],
                    "reason_a": a.get(f"{f}_reason", ""),
                })

        if job_matches:
            all_match += 1

    return {
        "total_compared": total,
        "all_fields_match": all_match,
        "all_fields_match_pct": round(100 * all_match / total, 1) if total else 0,
        "field_match": {f: round(100 * field_match[f] / field_total[f], 1) if field_total[f] else 0 for f in fields},
        "field_match_counts": {f: f"{field_match[f]}/{field_total[f]}" for f in fields},
        "errors_a": errors_a,
        "errors_b": errors_b,
        "mismatches": mismatches,
        "total_prompt_tokens_a": sum(r.get("prompt_tokens", 0) for r in results_a),
        "total_completion_tokens_a": sum(r.get("completion_tokens", 0) for r in results_a),
        "total_prompt_tokens_b": sum(r.get("prompt_tokens", 0) for r in results_b),
        "total_completion_tokens_b": sum(r.get("completion_tokens", 0) for r in results_b),
    }


# ── Main ─────────────────────────────────────────────────────────────
async def main():
    jobs = load_jsonl(SAMPLE_FILE)
    prompt_a = load_prompt(PROMPT_WITH_REASONING)
    prompt_b = load_prompt(PROMPT_NO_REASONING)

    print(f"Experiment: reasoning vs no-reasoning")
    print(f"Sample: {len(jobs)} jobs")
    print(f"Model: {MODEL}, temperature: {TEMPERATURE}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Prompt A: {PROMPT_WITH_REASONING} ({len(prompt_a)} chars)")
    print(f"Prompt B: {PROMPT_NO_REASONING} ({len(prompt_b)} chars)")
    print("=" * 60)

    client = AsyncOpenAI()

    # Run A: with reasoning
    print(f"\nRun A: WITH reasoning ({len(jobs)} jobs)...")
    start_a = time.time()
    results_a = await label_batch(client, prompt_a, jobs, "A")
    time_a = time.time() - start_a
    print(f"  Done in {time_a:.1f}s")

    # Run B: no reasoning
    print(f"\nRun B: NO reasoning ({len(jobs)} jobs)...")
    start_b = time.time()
    results_b = await label_batch(client, prompt_b, jobs, "B")
    time_b = time.time() - start_b
    print(f"  Done in {time_b:.1f}s")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    comparison = compare_results(results_a, results_b, jobs)

    print(f"\nJobs compared: {comparison['total_compared']}")
    print(f"Errors — Run A: {comparison['errors_a']}, Run B: {comparison['errors_b']}")
    print(f"\nAll 4 fields match: {comparison['all_fields_match']}/{comparison['total_compared']} = {comparison['all_fields_match_pct']}%")
    print(f"\nPer-field match rates:")
    for f in ["loc", "role", "tech", "comp"]:
        print(f"  {f}: {comparison['field_match_counts'][f]} = {comparison['field_match'][f]}%")

    print(f"\nToken usage:")
    print(f"  Run A (reasoning):    {comparison['total_prompt_tokens_a']:,} prompt + {comparison['total_completion_tokens_a']:,} completion")
    print(f"  Run B (no reasoning): {comparison['total_prompt_tokens_b']:,} prompt + {comparison['total_completion_tokens_b']:,} completion")
    savings = comparison['total_completion_tokens_a'] - comparison['total_completion_tokens_b']
    if comparison['total_completion_tokens_a'] > 0:
        savings_pct = round(100 * savings / comparison['total_completion_tokens_a'])
        print(f"  Completion token savings: {savings:,} ({savings_pct}%)")

    print(f"\nTime:")
    print(f"  Run A: {time_a:.1f}s")
    print(f"  Run B: {time_b:.1f}s")

    # Show mismatches
    if comparison['mismatches']:
        print(f"\n{'─' * 60}")
        print(f"MISMATCHES ({len(comparison['mismatches'])} disagreements)")
        print(f"{'─' * 60}")
        for m in comparison['mismatches'][:30]:
            print(f"\n  [{m['job_id']}] {m['title']}")
            print(f"    {m['field']}: reasoning={m['with_reasoning']}  vs  no_reasoning={m['no_reasoning']}")
            if m['reason_a']:
                print(f"    reason: {m['reason_a']}")

    # Save full results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/run_a_with_reasoning.jsonl", "w") as f:
        for r in results_a:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUTPUT_DIR}/run_b_no_reasoning.jsonl", "w") as f:
        for r in results_b:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUTPUT_DIR}/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())

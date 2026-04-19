#!/usr/bin/env python3
"""
Generate synthetic job descriptions for V15 training data augmentation.

Targets specific distribution gaps identified in V15 error analysis:
1. "Anywhere" location → REMOTE (20 jobs)
2. NODE-alone tech (30 jobs)
3. AI_ML+NODE combos (25 jobs)
4. JS_TS+NODE combos (15 jobs)
5. Next.js contrastive — JS_TS not NODE (15 jobs)
6. Secondary AI_ML examples (10 jobs)
7. Comp boundary examples (15 jobs)
8. Non-GBP currency examples (15 jobs)
9. UP_TO_ONLY examples (10 jobs)

Usage:
    OPENAI_API_KEY=... python3 finetune/generate_v15_data.py \
        --output data/v15/synthetic_raw.jsonl \
        --model gpt-4.1-mini
"""

import json
import os
import sys
import time
import argparse
import random
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    sys.exit(1)


# ── Generation Specifications ────────────────────────────────────────────────

# Each spec defines a batch of jobs to generate.
# The system prompt tells gpt-4.1-mini to generate realistic JDs.
# The user prompt gives specific constraints for each batch.

GENERATION_SPECS = [
    # ── LOC: "Anywhere" → REMOTE ──────────────────────────────────────────
    {
        "id": "anywhere_remote",
        "count": 15,
        "description": "Jobs with job_location='Anywhere', should be classified as loc=REMOTE",
        "constraints": """Generate realistic software engineering job descriptions where:
- job_location MUST be exactly "Anywhere" (no other text)
- The job is genuinely remote, open to global applicants
- Do NOT mention any specific country in the JD text
- Include a mix of: backend, frontend, fullstack, DevOps, data roles
- Vary company sizes (startup to enterprise)
- Some should mention Node.js, React, TypeScript, AI/ML in their tech stacks
- Some should be OOS (Python, Go, Java only)
- Include realistic salary ranges in GBP (£) for about half, no salary for the rest
- Each JD should be 200-400 words""",
    },
    {
        "id": "anywhere_outside_uk",
        "count": 5,
        "description": "Jobs with job_location='Anywhere' but clearly non-UK (OUTSIDE_UK)",
        "constraints": """Generate realistic job descriptions where:
- job_location MUST be exactly "Anywhere"
- BUT the JD text clearly states the company is based in a non-UK country (US, Germany, Singapore, etc.)
- The JD should mention "our office in [non-UK city]" or "based in [non-UK location]"
- These should still be classified as OUTSIDE_UK despite the 'Anywhere' location
- Mix of tech stacks
- Each JD should be 200-400 words""",
    },

    # ── TECH: NODE-alone ──────────────────────────────────────────────────
    {
        "id": "node_alone",
        "count": 30,
        "description": "Jobs where Node.js is the only tracked tech (no React, no JS_TS explicit mention, no AI/ML)",
        "constraints": """Generate realistic job descriptions where:
- The tech stack includes Node.js (use variations: "Node.js", "nodejs", "node.js", "Node")
- Do NOT mention React, Angular, Vue or any frontend framework
- Do NOT mention JavaScript or TypeScript by name (Node.js implies it but don't state it)
- Do NOT mention AI, ML, machine learning, LLM, or deep learning
- Other tech is fine: Express, MongoDB, PostgreSQL, Redis, Docker, AWS, Python, Go, etc.
- Mix of roles: Backend Engineer, API Developer, Backend Developer, Node.js Developer
- Vary seniority: junior, mid, senior, lead
- UK-based locations for most (London, Manchester, Bristol, etc.)
- Include GBP salaries for about half
- Each JD should be 200-400 words""",
    },

    # ── TECH: AI_ML + NODE combos ─────────────────────────────────────────
    {
        "id": "ai_ml_node",
        "count": 25,
        "description": "Jobs with both AI/ML AND Node.js in the tech stack",
        "constraints": """Generate realistic job descriptions where:
- The tech stack includes BOTH Node.js AND AI/ML technologies
- Use "Node.js" or "nodejs" explicitly
- Use AI keywords: "AI", "machine learning", "LLM", "deep learning", "generative AI"
- Example roles: ML Engineer with Node.js API, Full Stack AI Developer, AI Platform Engineer
- Some should also include React or TypeScript (multiple tracked tokens)
- UK-based locations for most
- Include GBP salaries for about half
- Each JD should be 200-400 words""",
    },

    # ── TECH: JS_TS + NODE combos ─────────────────────────────────────────
    {
        "id": "js_ts_node",
        "count": 15,
        "description": "Jobs with JavaScript/TypeScript AND Node.js explicitly mentioned",
        "constraints": """Generate realistic job descriptions where:
- The tech stack includes BOTH Node.js AND TypeScript/JavaScript explicitly
- Use "Node.js" and "TypeScript" or "JavaScript" by name
- Do NOT include React
- Do NOT include AI/ML keywords
- Example roles: Backend TypeScript Developer, Node.js API Engineer
- Other tech: Express, NestJS (Note: NestJS is a TypeScript framework, NOT Node.js), PostgreSQL, etc.
- UK-based locations for most
- Each JD should be 200-400 words""",
    },

    # ── TECH: Next.js contrastive (JS_TS not NODE) ────────────────────────
    {
        "id": "nextjs_not_node",
        "count": 15,
        "description": "Jobs using Next.js (which is JS_TS/REACT, NOT NODE)",
        "constraints": """Generate realistic job descriptions where:
- The tech stack includes Next.js (a React framework built on TypeScript)
- Do NOT mention Node.js, nodejs, or "node" as a standalone technology
- Next.js is a frontend/fullstack React framework — it is NOT Node.js
- Use "Next.js", "TypeScript", "React" explicitly
- Example roles: Frontend Engineer, Next.js Developer, Full Stack React Developer
- Some should mention AI/ML as secondary tech
- UK-based locations for most
- Include GBP salaries for about half
- Each JD should be 200-400 words""",
    },

    # ── TECH: Secondary AI_ML ─────────────────────────────────────────────
    {
        "id": "secondary_ai_ml",
        "count": 10,
        "description": "Non-AI roles where AI/ML is mentioned as secondary skill",
        "constraints": """Generate realistic job descriptions where:
- The PRIMARY role is NOT AI/ML (e.g., DevOps, QA, Data Analyst, Technical Writer, Product Manager)
- BUT AI/ML is mentioned as a secondary skill or tool: "AI-assisted development", "uses machine learning", "works with LLM APIs", "familiar with AI tools"
- The golden tech should include AI_ML because ANY mention counts
- Some should also have other tracked tech (Node.js, React, JavaScript)
- UK-based locations for most
- Each JD should be 200-400 words""",
    },

    # ── COMP: Boundary examples ───────────────────────────────────────────
    {
        "id": "comp_boundary",
        "count": 15,
        "description": "Jobs with salaries on comp bucket boundaries",
        "constraints": """Generate realistic UK-based software engineering job descriptions with these SPECIFIC salary ranges (use GBP £):
- 5 jobs with "£60,000 - £80,000" (midpoint £70k → RANGE_55_74K)
- 3 jobs with "£70,000 - £75,000" (midpoint £72.5k → RANGE_55_74K)
- 3 jobs with "£75,000 - £85,000" (midpoint £80k → RANGE_75_99K)
- 2 jobs with "£40,000 - £75,000" (midpoint £57.5k → RANGE_55_74K)
- 2 jobs with "£43,000 - £51,000" (midpoint £47k → RANGE_45_54K)
- State the salary EXACTLY as shown above (the formatting matters for training)
- Mix of roles and seniority levels
- UK locations (London, Manchester, etc.)
- Each JD should be 200-400 words""",
    },

    # ── COMP: Non-GBP currency ────────────────────────────────────────────
    {
        "id": "comp_non_gbp",
        "count": 15,
        "description": "Jobs with non-GBP salaries (USD, CAD, EUR, OTE, daily rates)",
        "constraints": """Generate realistic job descriptions with these specific salary patterns:
- 5 jobs with USD salaries: "$90,000", "$150,000 - $200,000", etc.
- 3 jobs with CAD salaries: "$105,000 CAD", "CAD $80,000 - $100,000"
- 2 jobs with EUR salaries: "€80,000", "€60,000 - €75,000"
- 3 jobs with OTE amounts in GBP: "OTE £92,000", "On-target earnings: £80k", "OTE of £75,000"
- 2 jobs with daily rates in GBP: "£600 per day", "day rate of £500-£700"
- All of these should be classified as NO_GBP
- Mix of locations (US, Canada, Europe, UK for OTE/daily)
- Mix of tech stacks
- Each JD should be 200-400 words""",
    },

    # ── COMP: UP_TO_ONLY ──────────────────────────────────────────────────
    {
        "id": "comp_upto",
        "count": 10,
        "description": "Jobs with 'up to' or 'to' salary patterns (UP_TO_ONLY)",
        "constraints": """Generate realistic UK-based job descriptions with these salary patterns:
- 3 jobs with "Up to £80,000" format
- 3 jobs with "to £85k" format (no "up" prefix, just "to £X")
- 2 jobs with "Up to £70,000" format
- 2 CONTRASTIVE jobs with "£60,000 to £80,000" format — this is a RANGE, not UP_TO_ONLY
  (having a lower bound "£60,000 to" makes it a range, not "up to")
- UK locations
- Mix of tech stacks
- Each JD should be 200-400 words""",
    },
]


SYSTEM_PROMPT = """You are a job description generator for a training dataset. Generate realistic, diverse job descriptions that a human recruiter might post on LinkedIn or Indeed.

Output format: JSON array of objects, each with these fields:
- "job_id": unique string (use format "v15_gen_NNNN")
- "title": job title
- "company": realistic company name (make up a name)
- "job_location": location string (EXACTLY as specified in constraints)
- "jd_text": full job description (200-400 words, realistic structure with sections)

Make each JD unique — vary company names, industries, benefits, requirements, and writing style. Use realistic UK/international company patterns. Include sections like About Us, The Role, Requirements, Salary/Benefits, How to Apply."""


def generate_batch(client, spec, model, temperature=0.7):
    """Generate a batch of synthetic JDs using OpenAI."""
    user_prompt = f"""Generate exactly {spec['count']} job descriptions.

CRITICAL CONSTRAINTS:
{spec['constraints']}

Output a JSON array of {spec['count']} objects. Start job_id numbering from "v15_gen_{spec['id']}_001"."""

    print(f"\n  Generating {spec['count']} jobs for: {spec['id']}...")

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=min(16384, 800 * spec["count"]),  # ~800 tokens per job, cap at 16k
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        # Handle both {"jobs": [...]} and [...] formats
        if isinstance(result, dict):
            jobs = result.get("jobs", result.get("data", result.get("results", [])))
            if not isinstance(jobs, list):
                jobs = [v for v in result.values() if isinstance(v, list)]
                jobs = jobs[0] if jobs else []
        elif isinstance(result, list):
            jobs = result
        else:
            jobs = []

        print(f"  Generated {len(jobs)} jobs (target: {spec['count']})")
        return jobs, response.usage
    except json.JSONDecodeError as e:
        print(f"  ERROR: Failed to parse JSON response: {e}")
        print(f"  Response (first 500 chars): {content[:500]}")
        return [], response.usage


def main():
    parser = argparse.ArgumentParser(description="Generate V15 synthetic training data")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    parser.add_argument("--batch", default=None, help="Only generate specific batch ID")
    args = parser.parse_args()

    # Plan summary
    total = sum(s["count"] for s in GENERATION_SPECS)
    print(f"V15 Synthetic Data Generation Plan")
    print(f"{'='*60}")
    print(f"Total jobs to generate: {total}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output}")
    print(f"\nBatches:")
    for spec in GENERATION_SPECS:
        marker = " ← selected" if args.batch and spec["id"] == args.batch else ""
        print(f"  {spec['id']:30s} {spec['count']:3d} jobs{marker}")

    if args.dry_run:
        print("\n[DRY RUN — no generation performed]")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable is required")
        print("Set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    client = OpenAI(timeout=120.0, max_retries=3)

    # Filter to specific batch if requested
    specs = GENERATION_SPECS
    if args.batch:
        specs = [s for s in specs if s["id"] == args.batch]
        if not specs:
            print(f"\nERROR: Batch '{args.batch}' not found")
            sys.exit(1)

    # Generate all batches
    all_jobs = []
    total_tokens = 0

    for spec in specs:
        jobs, usage = generate_batch(client, spec, args.model, args.temperature)
        if usage:
            total_tokens += usage.total_tokens

        # Validate and tag each job
        for job in jobs:
            # Ensure required fields exist
            if not all(k in job for k in ["title", "company", "jd_text"]):
                print(f"  WARNING: Skipping job missing required fields: {job.get('title', '?')}")
                continue

            # Add metadata
            job["source_file"] = f"v15_gen_{spec['id']}"
            job["augmentation_type"] = spec["id"]
            if "job_id" not in job:
                job["job_id"] = f"v15_gen_{spec['id']}_{len(all_jobs):04d}"
            if "job_location" not in job:
                job["job_location"] = ""

            all_jobs.append(job)

        # Rate limit
        time.sleep(1)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for job in all_jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Generated {len(all_jobs)} jobs total")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Output saved to: {args.output}")

    # Verify distribution
    print(f"\nPer-batch counts:")
    from collections import Counter
    batch_counts = Counter(j["augmentation_type"] for j in all_jobs)
    for spec in GENERATION_SPECS:
        actual = batch_counts.get(spec["id"], 0)
        target = spec["count"]
        status = "✓" if actual >= target else f"⚠️ {target - actual} short"
        print(f"  {spec['id']:30s} {actual:3d}/{target:3d} {status}")


if __name__ == "__main__":
    main()

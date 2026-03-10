#!/usr/bin/env python3
"""
Generate V7 Gap-Filling Recipes
================================
Creates recipes (6-field specifications) for generating synthetic jobs
to fill distribution gaps. Each recipe combines multiple underrepresented
tokens to maximize efficiency.

Output: JSONL file where each line is a recipe with all 6 V7 fields specified,
plus a donor_strategy field for the OpenAI generation step.

Usage:
    python3 scripts/generate_v7_recipes.py \
        --gap-plan data/v7/generation_gap_plan.json \
        --selected data/v7/curated_dataset.jsonl \
        --csv data/v7/distribution_raw.csv \
        --output data/v7/generation_recipes.jsonl
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

# V7 token definitions for validation
VALID_TOKENS = {
    'location': ['IN_LONDON', 'FULLY_REMOTE', 'UK_OTHER', 'OUTSIDE_UK', 'UNKNOWN'],
    'work_arrangement': ['REMOTE', 'HYBRID', 'IN_OFFICE', 'UNKNOWN'],
    'scope': ['IN_SCOPE', 'OUT_OF_SCOPE'],
    'seniority': ['LEVEL_1', 'LEVEL_2', 'LEVEL_3'],
    'tech': ['NONE', 'JS_TS', 'NODE', 'NODE_JS_TS', 'AI_ML', 'JS_TS_AI_ML', 'NODE_AI_ML', 'NODE_JS_TS_AI_ML'],
    'comp': ['NO_GBP', 'UP_TO_ONLY', 'BELOW_45K', 'RANGE_55_74K', 'RANGE_75_99K', 'ABOVE_100K'],
}

# Location examples for generation prompts
LOCATION_EXAMPLES = {
    'IN_LONDON': ['London, England, United Kingdom', 'London, UK', 'Central London'],
    'FULLY_REMOTE': ['Remote', 'Fully Remote', 'Remote (UK)', 'Remote - United Kingdom', 'UK Remote', 'Anywhere'],
    'UK_OTHER': ['Manchester, England, United Kingdom', 'Birmingham, UK', 'Edinburgh, Scotland', 'Bristol, UK', 'Leeds, UK', 'Cambridge, UK'],
    'OUTSIDE_UK': ['New York, NY', 'San Francisco, CA', 'Berlin, Germany', 'Amsterdam, Netherlands', 'Dublin, Ireland', 'Toronto, Canada'],
    'UNKNOWN': ['', 'Not specified', 'N/A'],
}

# Tech stack descriptions for generation
TECH_DESCRIPTIONS = {
    'NONE': 'No JavaScript/TypeScript/Node.js/AI-ML technologies mentioned',
    'JS_TS': 'JavaScript or TypeScript (React, Angular, Vue, frontend) but NOT Node.js',
    'NODE': 'Node.js backend but NOT JavaScript/TypeScript frontend frameworks',
    'NODE_JS_TS': 'Node.js backend WITH JavaScript/TypeScript (fullstack)',
    'AI_ML': 'AI/ML technologies (TensorFlow, PyTorch, LLMs, NLP) but NOT JS/TS/Node',
    'JS_TS_AI_ML': 'JavaScript/TypeScript AND AI/ML but NOT Node.js',
    'NODE_AI_ML': 'Node.js AND AI/ML but NOT frontend JS/TS',
    'NODE_JS_TS_AI_ML': 'Node.js + JavaScript/TypeScript + AI/ML (fullstack + AI)',
}

# Comp ranges for generation
COMP_RANGES = {
    'NO_GBP': 'No salary in GBP (USD, EUR, or no salary mentioned)',
    'UP_TO_ONLY': 'Only "up to" amount (e.g., "up to £80,000")',
    'BELOW_45K': 'GBP salary below £45,000 (e.g., £25,000-£40,000)',
    'RANGE_55_74K': 'GBP salary £55,000-£74,999 (e.g., £55,000-£70,000)',
    'RANGE_75_99K': 'GBP salary £75,000-£99,999 (e.g., £75,000-£95,000)',
    'ABOVE_100K': 'GBP salary £100,000+ (e.g., £100,000-£130,000)',
}

# Scope descriptions
SCOPE_DESCRIPTIONS = {
    'IN_SCOPE': 'Software engineering role (developer, engineer, architect)',
    'OUT_OF_SCOPE': 'Non-engineering role (PM, designer, DevOps, data analyst, QA, support, sales)',
}

# Seniority descriptions
SENIORITY_TITLES = {
    'LEVEL_3': ['Senior Software Engineer', 'Lead Engineer', 'Principal Engineer', 'Staff Engineer', 'Senior Developer', 'Engineering Manager'],
    'LEVEL_2': ['Software Engineer', 'Developer', 'Full Stack Engineer', 'Backend Engineer', 'Frontend Developer'],
    'LEVEL_1': ['Junior Software Engineer', 'Junior Developer', 'Graduate Engineer', 'Associate Developer', 'Trainee Engineer'],
}


def load_distribution_targets(csv_path: str) -> dict:
    """Load exact distribution targets from CSV."""
    targets = defaultdict(dict)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            field = row['Field'].lower()
            token = row['Token']
            targets[field][token] = int(row['Total'])
    return dict(targets)


def compute_selected_counts(selected_path: str) -> dict:
    """Compute V7 estimated token counts from selected dataset."""
    counts = defaultdict(Counter)
    with open(selected_path) as f:
        for line in f:
            job = json.loads(line)
            v7_est = job.get('_v7_est', {})
            for field, token in v7_est.items():
                if token != 'UNSURE':
                    counts[field][token] += 1
    return dict(counts)


def compute_gaps(selected_counts: dict, targets: dict, total_selected: int) -> dict:
    """Compute remaining gaps per field/token."""
    total_target = sum(targets.get('location', {}).values())
    n_to_generate = total_target - total_selected

    gaps = {}
    for field, token_targets in targets.items():
        gaps[field] = {}
        for token, target in token_targets.items():
            current = selected_counts.get(field, {}).get(token, 0)
            gap = max(0, target - current)
            gaps[field][token] = gap

    return gaps, n_to_generate


def build_recipes(gaps: dict, n_recipes: int) -> list:
    """Build n_recipes generation recipes that fill all gaps.

    Each recipe specifies all 6 fields. The marginal distributions
    of the recipes across all fields should match the gaps.

    Strategy: assign tokens to recipes in proportion to their gaps,
    distributing across recipes to maximize diversity.
    """
    random.seed(42)

    # Build token pools for each field
    # Each pool has tokens repeated according to their gap count
    pools = {}
    for field in ['location', 'work_arrangement', 'scope', 'seniority', 'tech', 'comp']:
        pool = []
        field_gaps = gaps.get(field, {})

        if field == 'seniority':
            # Seniority not in distribution CSV — distribute evenly
            per = n_recipes // 3
            pool = ['LEVEL_3'] * per + ['LEVEL_2'] * per + ['LEVEL_1'] * (n_recipes - 2 * per)
        elif field == 'work_arrangement':
            # Work arrangement: for FULLY_REMOTE location, pair with REMOTE
            # For others, distribute based on gaps
            # We'll handle this separately after location assignment
            continue
        elif field == 'scope':
            # Scope: IN_SCOPE should be 70%, OUT_OF_SCOPE 30%
            in_scope_gap = field_gaps.get('IN_SCOPE', 0)
            out_scope_gap = field_gaps.get('OUT_OF_SCOPE', 0)
            total_gap = in_scope_gap + out_scope_gap
            if total_gap > n_recipes:
                # Scale down proportionally
                in_scope_n = round(n_recipes * in_scope_gap / total_gap)
                out_scope_n = n_recipes - in_scope_n
            else:
                in_scope_n = in_scope_gap
                out_scope_n = out_scope_gap
                # Fill remaining with proportional split
                remaining = n_recipes - in_scope_n - out_scope_n
                in_scope_n += round(remaining * 0.7)
                out_scope_n = n_recipes - in_scope_n
            pool = ['IN_SCOPE'] * in_scope_n + ['OUT_OF_SCOPE'] * out_scope_n
        else:
            for token, gap in field_gaps.items():
                pool.extend([token] * gap)

            # If pool is shorter than n_recipes, pad with most-needed tokens
            while len(pool) < n_recipes:
                # Add tokens proportionally to remaining gaps
                for token, gap in sorted(field_gaps.items(), key=lambda x: -x[1]):
                    if gap > 0 and len(pool) < n_recipes:
                        pool.append(token)

            # If pool is longer, truncate (shouldn't happen often)
            pool = pool[:n_recipes]

        random.shuffle(pool)
        pools[field] = pool

    # Now build recipes by assigning from pools
    recipes = []
    for i in range(n_recipes):
        recipe = {}
        for field in ['location', 'scope', 'seniority', 'tech', 'comp']:
            if field in pools and i < len(pools[field]):
                recipe[field] = pools[field][i]
            else:
                # Fallback: pick a random valid token
                recipe[field] = random.choice(VALID_TOKENS[field])

        # Assign work_arrangement based on location
        if recipe['location'] == 'FULLY_REMOTE':
            recipe['work_arrangement'] = 'REMOTE'
        elif recipe['location'] == 'IN_LONDON':
            recipe['work_arrangement'] = random.choice(['HYBRID', 'IN_OFFICE', 'HYBRID'])
        elif recipe['location'] == 'UK_OTHER':
            recipe['work_arrangement'] = random.choice(['HYBRID', 'IN_OFFICE', 'HYBRID'])
        elif recipe['location'] == 'OUTSIDE_UK':
            recipe['work_arrangement'] = random.choice(['HYBRID', 'IN_OFFICE', 'REMOTE'])
        else:  # UNKNOWN
            recipe['work_arrangement'] = 'UNKNOWN'

        # Assign a concrete location string
        recipe['location_text'] = random.choice(LOCATION_EXAMPLES[recipe['location']])

        # Assign a title based on scope + seniority
        if recipe['scope'] == 'IN_SCOPE':
            recipe['suggested_title'] = random.choice(SENIORITY_TITLES[recipe['seniority']])
        else:
            # Out of scope roles
            out_of_scope_titles = [
                'Product Manager', 'Project Manager', 'Data Analyst',
                'QA Engineer', 'DevOps Engineer', 'UX Designer',
                'Business Analyst', 'IT Support Engineer', 'Security Engineer',
                'Database Administrator', 'Scrum Master', 'Technical Writer',
            ]
            recipe['suggested_title'] = random.choice(out_of_scope_titles)

        # SEMANTIC CONSTRAINT: OUT_OF_SCOPE roles always get tech=NONE from GPT.
        # If recipe has OUT_OF_SCOPE + non-NONE tech, convert to IN_SCOPE.
        if recipe['scope'] == 'OUT_OF_SCOPE' and recipe['tech'] != 'NONE':
            recipe['scope'] = 'IN_SCOPE'
            recipe['suggested_title'] = random.choice(SENIORITY_TITLES[recipe['seniority']])

        # Add descriptions for generation
        recipe['tech_description'] = TECH_DESCRIPTIONS[recipe['tech']]
        recipe['comp_description'] = COMP_RANGES[recipe['comp']]
        recipe['scope_description'] = SCOPE_DESCRIPTIONS[recipe['scope']]

        recipes.append(recipe)

    return recipes


def validate_recipes(recipes: list, gaps: dict, n_recipes: int):
    """Validate that recipes fill the gaps correctly."""
    counts = defaultdict(Counter)
    for r in recipes:
        for field in ['location', 'work_arrangement', 'scope', 'seniority', 'tech', 'comp']:
            counts[field][r[field]] += 1

    print("\n  Recipe distribution vs gaps:")
    for field in ['location', 'tech', 'comp', 'scope', 'work_arrangement', 'seniority']:
        print(f"\n  {field.upper()}")
        field_gaps = gaps.get(field, {})
        for token, count in sorted(counts[field].items(), key=lambda x: -x[1]):
            gap = field_gaps.get(token, 0)
            status = '✅' if count >= gap else f'⚠️ (gap={gap})'
            print(f"    {token:<25} recipes={count:>5}  gap={gap:>5}  {status}")


def output_recipes(recipes: list, output_path: str):
    """Output recipes as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i, r in enumerate(recipes):
            r['recipe_id'] = f"gen_{i:04d}"
            f.write(json.dumps(r) + '\n')
    print(f"\n  Wrote {len(recipes)} recipes to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate V7 Gap-Filling Recipes')
    parser.add_argument('--selected', type=str, default='data/v7/curated_dataset.jsonl')
    parser.add_argument('--csv', type=str, default='data/v7/distribution_raw.csv')
    parser.add_argument('--output', type=str, default='data/v7/generation_recipes.jsonl')
    parser.add_argument('--base-dir', type=str,
                        default='/Users/faisal/Code/automation/ai_eval_harness')
    args = parser.parse_args()

    # Resolve paths
    selected_path = os.path.join(args.base_dir, args.selected)
    csv_path = os.path.join(args.base_dir, args.csv)
    output_path = os.path.join(args.base_dir, args.output)

    print("=" * 70)
    print("V7 RECIPE GENERATOR")
    print("=" * 70)

    # Load targets and current counts
    targets = load_distribution_targets(csv_path)
    selected_counts = compute_selected_counts(selected_path)

    # Count selected
    total_selected = 0
    with open(selected_path) as f:
        for line in f:
            total_selected += 1
    print(f"  Selected: {total_selected} real jobs")

    total_target = sum(targets.get('location', {}).values())
    n_to_generate = total_target - total_selected
    print(f"  Target: {total_target} total")
    print(f"  Need to generate: {n_to_generate}")

    # Compute gaps
    gaps, _ = compute_gaps(selected_counts, targets, total_selected)
    print("\n  Gaps to fill:")
    for field in ['location', 'tech', 'comp', 'scope', 'work_arrangement']:
        field_gaps = {k: v for k, v in gaps.get(field, {}).items() if v > 0}
        if field_gaps:
            print(f"    {field}: {dict(field_gaps)}")

    # Build recipes
    print(f"\n  Building {n_to_generate} recipes...")
    recipes = build_recipes(gaps, n_to_generate)

    # Validate
    validate_recipes(recipes, gaps, n_to_generate)

    # Output
    output_recipes(recipes, output_path)

    print("\n" + "=" * 70)
    print("DONE — Next step: run generate_v7_jobs.py to create JDs from recipes")
    print("=" * 70)


if __name__ == '__main__':
    main()

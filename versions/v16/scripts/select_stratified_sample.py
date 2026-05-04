#!/usr/bin/env python3
"""
Select stratified sample of 50 high-risk jobs from V15 data for V16 prompt validation.
Risk categories based on XGBoost audit findings.
"""
import json
import sys
from pathlib import Path

def select_stratified_sample(input_path: str, output_path: str, n=50):
    jobs = []
    with open(input_path) as f:
        for line in f:
            jobs.append(json.loads(line))

    categories = {
        'sen_plain_title': [],      # Plain "Software Engineer" etc - sen mismatch risk
        'sen_ambiguous': [],        # Title without clear L1/L3 keywords
        'tech_non_swe': [],         # Non-SWE role with tech mentions
        'tech_company_desc': [],    # Company descriptions likely have AI/ML boilerplate
        'comp_qualifier': [],       # "competitive", "DOE", "OTE" in comp
        'comp_range_edge': [],      # Ranges near bucket boundaries
        'loc_ambiguous': [],        # Ambiguous locations
        'multi_field_risk': [],     # Multiple fields look suspicious
    }

    for job in jobs:
        title = job.get('title', '').lower()
        jd = job.get('jd_text', '').lower()
        labels = job.get('v15_labels', {})

        # SEN risks
        if any(t in title for t in ['software engineer', 'developer', 'engineer']):
            if not any(k in title for k in ['senior', 'lead', 'staff', 'principal', 'head', 'director', 'vp', 'cto', 'founding', 'distinguished', 'sr', 'iii', 'manager', 'junior', 'jr', 'graduate', 'intern', 'trainee', 'apprentice', 'associate']):
                categories['sen_plain_title'].append(job)
            else:
                categories['sen_ambiguous'].append(job)

        # TECH risks
        tech_raw = labels.get('tech_raw', '') or ''
        tech = labels.get('tech', [])
        if 'AI' in tech_raw or 'machine learning' in jd or 'LLM' in tech_raw:
            if not any(swe in title for swe in ['software', 'engineer', 'developer', 'full stack', 'frontend', 'backend', 'web']):
                categories['tech_non_swe'].append(job)
            else:
                categories['tech_company_desc'].append(job)

        # COMP risks
        comp_raw = labels.get('comp_raw', '') or ''
        if any(q in comp_raw.lower() for q in ['competitive', 'doe', 'ote', 'dependent on experience']):
            categories['comp_qualifier'].append(job)
        elif any(q in comp_raw.lower() for q in ['+', 'and above', 'up to']):
            categories['comp_range_edge'].append(job)

        # LOC risks
        loc = labels.get('loc', '')
        if loc == 'OUTSIDE_UK':
            categories['loc_ambiguous'].append(job)

    # Select from each category proportionally
    selected = []
    target_per_cat = max(1, n // len(categories))

    for cat_name, cat_jobs in categories.items():
        # Sort by some proxy for risk (e.g., plain titles first, longer JDs for tech)
        if cat_name.startswith('sen'):
            cat_jobs.sort(key=lambda j: len(j.get('title', '')), reverse=False)  # shorter = more ambiguous
        elif cat_name.startswith('tech'):
            cat_jobs.sort(key=lambda j: len(j.get('jd_text', '')), reverse=True)  # longer = more boilerplate risk
        else:
            cat_jobs.sort(key=lambda j: j.get('title', ''))

        take = min(target_per_cat, len(cat_jobs))
        selected.extend(cat_jobs[:take])
        print(f"  {cat_name}: {len(cat_jobs)} available, took {take}")

    # If we don't have enough, fill from multi-field risk
    if len(selected) < n:
        # Find jobs that appear in multiple categories
        job_ids = set(j.get('job_id') for j in selected)
        for job in jobs:
            if job.get('job_id') not in job_ids:
                categories['multi_field_risk'].append(job)
        categories['multi_field_risk'].sort(key=lambda j: len(j.get('jd_text', '')), reverse=True)
        needed = n - len(selected)
        selected.extend(categories['multi_field_risk'][:needed])
        print(f"  multi_field_risk: filled {needed} from general pool")

    # Deduplicate
    seen = set()
    deduped = []
    for job in selected:
        jid = job.get('job_id', job.get('index'))
        if jid not in seen:
            seen.add(jid)
            deduped.append(job)

    selected = deduped[:n]

    print(f"\nTotal selected: {len(selected)} jobs")

    with open(output_path, 'w') as f:
        for job in selected:
            f.write(json.dumps(job, ensure_ascii=False) + '\n')

    print(f"Wrote to {output_path}")

if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'versions/v16/data/v15_jobs_extracted.jsonl'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'versions/v16/data/sample_50_high_risk.jsonl'
    select_stratified_sample(input_path, output_path)

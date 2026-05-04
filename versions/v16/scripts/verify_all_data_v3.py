#!/usr/bin/env python3
"""Comprehensive V16 teacher v3 data verification - ALL jobs, only real bugs."""
import json
import re
from pathlib import Path

# Load all jobs
with open('versions/v16/data/v16_teacher_labels_v3.jsonl') as f:
    jobs = [json.loads(l) for l in f]

print(f"Total jobs: {len(jobs)}")

# Bug counters
bugs = {
    'loc': [],
    'arr': [],
    'sen': [],
    'tech': [],
    'comp': [],
    'schema': [],
}

# Location validation
LOC_VALID = {'IN_LONDON', 'REMOTE', 'UK_OTHER', 'OUTSIDE_UK', 'UNK'}
LOC_MAP = {'IN_LONDON':25,'REMOTE':25,'UK_OTHER':10,'OUTSIDE_UK':-50,'UNK':0}

# Arr validation  
ARR_VALID = {'REMOTE', 'HYBRID', 'IN_OFFICE', 'UNK'}

# Sen validation
SEN_VALID = {'LEVEL_3', 'LEVEL_2', 'LEVEL_1'}
SEN_MAP = {'LEVEL_3':25,'LEVEL_2':15,'LEVEL_1':0}

# Tech validation
TECH_VALID = {'NODE', 'REACT', 'JS_TS', 'AI_ML', 'OOS'}

# Comp validation
COMP_VALID = {'NO_GBP', 'UP_TO_ONLY', 'BELOW_30K', '30_40K', '40_50K', '50_60K', 
              '60_70K', '70_80K', '80_90K', '90_100K', '100_120K', '120_140K',
              '140_160K', '160_180K', '180_200K', 'ABOVE_200K', 'BELOW_45K',
              'RANGE_45_54K', 'RANGE_55_74K', 'RANGE_75_99K', 'ABOVE_100K'}
COMP_MAP = {'NO_GBP':0,'UP_TO_ONLY':0,'BELOW_30K':-30,'30_40K':-20,'40_50K':-10,
            '50_60K':0,'60_70K':5,'70_80K':10,'80_90K':15,'90_100K':20,'100_120K':25,
            '120_140K':25,'140_160K':25,'160_180K':25,'180_200K':25,'ABOVE_200K':25,
            'BELOW_45K':-30,'RANGE_45_54K':0,'RANGE_55_74K':5,'RANGE_75_99K':15,'ABOVE_100K':25}

def validate_loc(job):
    """Validate location."""
    pred = job.get('v16_loc')
    raw = job.get('v16_loc_raw')
    
    if pred not in LOC_VALID:
        return f"Invalid loc token: {pred}"
    if raw is None:
        return "loc_raw is null"
    
    # Check London presence
    if pred == 'IN_LONDON':
        if 'london' not in raw.lower() and 'london' not in job.get('job_location','').lower():
            return "loc=IN_LONDON but no London in location/text"
    elif pred == 'OUTSIDE_UK':
        if any(x in raw.lower() for x in ['london', 'uk', 'england', 'scotland', 'wales']):
            return "loc=OUTSIDE_UK but UK location found"
    return None

def validate_arr(job):
    """Validate work arrangement."""
    pred = job.get('v16_arr')
    desc = job.get('jd_text', '')
    
    if pred not in ARR_VALID:
        return f"Invalid arr token: {pred}"
    # No more checks - arr is purely from description
    return None

def validate_sen(job):
    """Validate seniority - check consistency with title/description."""
    pred = job.get('v16_sen')
    title = job.get('title', '').lower()
    desc = job.get('jd_text', '').lower()
    title_raw = job.get('v16_sen_raw', '')
    
    if pred not in SEN_VALID:
        return f"Invalid sen token: {pred}"
    
    # Check title consistency
    sen_tokens = []
    if 'senior' in title or 'staff' in title or 'lead' in title or 'principal' in title or 'head' in title or 'director' in title or 'vp' in title:
        sen_tokens.append('L3')
    if 'junior' in title or 'graduate' in title or 'intern' in title or 'trainee' in title or 'apprentice' in title or 'associate' in title:
        sen_tokens.append('L1')
    
    # Senior before roman numeral is LEVEL_1 (e.g., Senior Data Engineer I)
    if re.search(r'senior\s+.*\s+[ivxl]+$', title, re.I):
        if 'senior' in title:
            sen_tokens = ['L1']  # Override
    
    # Check L3 indicators in title
    has_l3_in_title = any(x in title for x in ['senior', 'staff', 'lead', 'principal', 'head', 'director', 'vp', 'cto', 'cfo', 'ceo'])
    # Remove "Senior X I/II/III" cases
    if has_l3_in_title and not re.search(r'^senior\s+.*\s+[ivxl]+$', title, re.I):
        if pred == 'LEVEL_1':
            return f"sen=LEVEL_1 but title has L3 indicator: {job.get('title')}"
    
    # Check L1 indicators
    has_l1 = any(x in title for x in ['junior', 'graduate', 'intern', 'trainee', 'apprentice', 'associate'])
    if has_l1 and pred == 'LEVEL_3':
        return f"sen=LEVEL_3 but title has L1 indicator: {job.get('title')}"
    
    # Senior with roman numeral at end is LEVEL_1
    if re.search(r'\b(senior| jr)\s+.*\s+([ivxl]+)$', title, re.I):
        if pred == 'LEVEL_3':
            return f"sen=LEVEL_3 but 'Senior X I/II/III' pattern: {job.get('title')}"
    
    return None

def validate_tech(job):
    """Validate tech stack."""
    pred = job.get('v16_tech', [])
    
    if not isinstance(pred, list):
        return f"tech is not a list: {type(pred)}"
    if len(pred) == 0:
        return "tech is empty array"
    if 'OOS' in pred and len(pred) > 1:
        return "tech has OOS mixed with other tokens"
    if 'OOS' in pred and len(pred) == 1:
        # OOS only valid if not SWE
        title = job.get('title', '').lower()
        if any(x in title for x in ['engineer', 'developer', 'software', 'devops', 'sre']):
            return f"tech=['OOS'] but title suggests SWE: {job.get('title')}"
        return None
    
    # Check for invalid tokens
    for t in pred:
        if t not in TECH_VALID:
            return f"Invalid tech token: {t}"
    
    return None

def validate_comp(job):
    """Validate compensation."""
    pred = job.get('v16_comp')
    raw = job.get('v16_comp_raw', '')
    
    if pred not in COMP_VALID:
        return f"Invalid comp token: {pred}"
    if raw is None or raw == '':
        return "comp_raw is empty/missing"
    
    # Check if raw contains GBP indicator
    has_gbp = '£' in raw or 'gbp' in raw.lower() or 'pound' in raw.lower()
    if pred == 'NO_GBP' and has_gbp:
        return f"comp=NO_GBP but raw contains GBP: {raw}"
    
    # Check specific patterns
    if 'OTE' in raw.upper() or 'DOE' in raw.upper() or 'depending' in raw.lower():
        return None  # These are valid NO_GBP cases
    
    return None

def check_schema(job):
    """Check required schema fields with v16_ prefix."""
    required = ['v16_loc_raw', 'v16_loc', 'v16_arr_raw', 'v16_arr', 'v16_sen_raw', 'v16_sen', 
                'v16_tech_raw', 'v16_tech', 'v16_comp_raw', 'v16_comp']
    missing = [f for f in required if job.get(f) is None]
    if missing:
        return f"Missing fields: {missing}"
    return None

# Process ALL jobs
print("\n=== VALIDATING ALL JOBS ===\n")
all_bugs = []
job_ids_with_bugs = set()

for job in jobs:
    idx = job.get('index', 'UNKNOWN')
    title = job.get('title', 'NO TITLE')
    
    # Check schema first
    schema_err = check_schema(job)
    if schema_err:
        all_bugs.append({'index': idx, 'title': title, 'field': 'schema', 'error': schema_err})
        job_ids_with_bugs.add(idx)
    
    # Validate each field
    loc_err = validate_loc(job)
    if loc_err:
        all_bugs.append({'index': idx, 'title': title, 'field': 'loc', 'error': loc_err})
        job_ids_with_bugs.add(idx)
    
    arr_err = validate_arr(job)
    if arr_err:
        all_bugs.append({'index': idx, 'title': title, 'field': 'arr', 'error': arr_err})
        job_ids_with_bugs.add(idx)
    
    sen_err = validate_sen(job)
    if sen_err:
        all_bugs.append({'index': idx, 'title': title, 'field': 'sen', 'error': sen_err})
        job_ids_with_bugs.add(idx)
    
    tech_err = validate_tech(job)
    if tech_err:
        all_bugs.append({'index': idx, 'title': title, 'field': 'tech', 'error': tech_err})
        job_ids_with_bugs.add(idx)
    
    comp_err = validate_comp(job)
    if comp_err:
        all_bugs.append({'index': idx, 'title': title, 'field': 'comp', 'error': comp_err})
        job_ids_with_bugs.add(idx)

# Report results
print(f"Total bugs found: {len(all_bugs)}")
print(f"Jobs with bugs: {len(job_ids_with_bugs)} / {len(jobs)} ({100*len(job_ids_with_bugs)/len(jobs):.1f}%)\n")

# Group by field
by_field = {}
for bug in all_bugs:
    by_field.setdefault(bug['field'], []).append(bug)

print("Bugs by field:")
for field in ['schema', 'loc', 'arr', 'sen', 'tech', 'comp']:
    count = len(by_field.get(field, []))
    print(f"  {field}: {count}")

print(f"\n=== DETAILED BUG REPORT (ALL {len(all_bugs)} BUGS) ===\n")

for i, bug in enumerate(all_bugs, 1):
    print(f"[{i}] Index {bug['index']} | {bug['field']}")
    print(f"    Title: {bug['title'][:80]}")
    print(f"    Error: {bug['error']}")
    print()

# Save bugs to file
with open('versions/v16/data/real_bugs_v3.json', 'w') as f:
    json.dump({
        'total_jobs': len(jobs),
        'jobs_with_bugs': len(job_ids_with_bugs),
        'total_bugs': len(all_bugs),
        'by_field': {k: len(v) for k, v in by_field.items()},
        'bugs': all_bugs
    }, f, indent=2)

print(f"\n=== SAVED: versions/v16/data/real_bugs_v3.json ===")

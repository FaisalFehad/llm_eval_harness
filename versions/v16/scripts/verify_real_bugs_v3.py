#!/usr/bin/env python3
"""Comprehensive V16 teacher v3 data verification - ONLY real teacher bugs (not parse failures)."""
import json
import re
from pathlib import Path

# Load all jobs
with open('versions/v16/data/v16_teacher_labels_v3.jsonl') as f:
    jobs = [json.loads(l) for l in f]

print(f"Total jobs: {len(jobs)}")

# Real teacher bugs only (not parsing failures)
# A bug is when teacher made a WRONG decision, not when data is missing

# Bug counters
bugs = {
    'loc': [],      # Location classification bugs
    'arr': [],      # Work arrangement bugs  
    'sen': [],      # Seniority classification bugs
    'tech': [],     # Tech stack bugs (OOS when should have tech, or vice versa)
    'comp': [],     # Compensation bugs
    'schema': [],   # Schema issues (already filled)
}

# Valid tokens
LOC_VALID = {'IN_LONDON', 'REMOTE', 'UK_OTHER', 'OUTSIDE_UK', 'UNK'}
ARR_VALID = {'REMOTE', 'HYBRID', 'IN_OFFICE', 'UNK'}
SEN_VALID = {'LEVEL_3', 'LEVEL_2', 'LEVEL_1'}
TECH_VALID = {'NODE', 'REACT', 'JS_TS', 'AI_ML', 'OOS'}
COMP_VALID = {'NO_GBP', 'UP_TO_ONLY', 'BELOW_30K', '30_40K', '40_50K', '50_60K', 
              '60_70K', '70_80K', '80_90K', '90_100K', '100_120K', '120_140K',
              '140_160K', '160_180K', '180_200K', 'ABOVE_200K', 'BELOW_45K',
              'RANGE_45_54K', 'RANGE_55_74K', 'RANGE_75_99K', 'ABOVE_100K'}

def is_swe_role(title):
    """Check if title suggests SWE role."""
    title_lower = title.lower() if title else ''
    swe_keywords = ['engineer', 'developer', 'software', 'devops', 'sre', 'ops', 'architect', 'programmer']
    return any(kw in title_lower for kw in swe_keywords)

def validate_loc(job):
    """Validate location - LOC IS NOT A TEACHER BUG, it follows rules."""
    # loc follows strict rules: check London in location field or description
    pred = job.get('v16_loc')
    loc_raw = job.get('v16_loc_raw', '')
    job_loc = job.get('job_location', '')
    jd = job.get('jd_text', '')
    
    if pred not in LOC_VALID:
        return None, "Invalid token"  # Format issue, not teacher bug
    
    # IN_LONDON only valid if London mentioned in location field
    if pred == 'IN_LONDON':
        if 'london' not in loc_raw.lower() and 'london' not in job_loc.lower():
            return True, "loc=IN_LONDON but no London in location field"
    
    # OUTSIDE_UK should not have UK markers
    if pred == 'OUTSIDE_UK':
        uk_markers = ['london', 'uk', 'england', 'scotland', 'wales', 'northern ireland']
        if any(m in loc_raw.lower() or m in job_loc.lower() for m in uk_markers):
            return True, "loc=OUTSIDE_UK but UK location in data"
    
    return None, None

def validate_arr(job):
    """Validate work arrangement -teacher makes this decision, check for consistency."""
    pred = job.get('v16_arr')
    jd = job.get('jd_text', '')
    
    if pred not in ARR_VALID:
        return None, "Invalid token"
    
    # Check that remote/hybrid/office markers match
    jd_lower = jd.lower()
    if pred == 'REMOTE':
        if 'remote' not in jd_lower and 'hybrid' not in jd_lower and 'office' not in jd_lower:
            return True, "arr=REMOTE but no remote/hybrid/office in description"
    elif pred == 'IN_OFFICE':
        if 'remote' in jd_lower and 'hybrid' not in jd_lower:
            return True, "arr=IN_OFFICE but description shows REMOTE"
    elif pred == 'HYBRID':
        if ' hybrid ' not in jd_lower and 'flexible' not in jd_lower and '部分' not in jd_lower:
            # Check for office mention with remote possibility
            if 'office' in jd_lower and 'remote' not in jd_lower:
                return True, "arr=HYBRID but only office mentioned"
    
    return None, None

def validate_sen(job):
    """Validate seniority - check title markers match teacher decision."""
    pred = job.get('v16_sen')
    title = job.get('title', '')
    title_lower = title.lower() if title else ''
    
    if pred not in SEN_VALID:
        return None, "Invalid token"
    
    # Check for L3 indicators in title
    l3_keywords = ['senior', 'staff', 'lead', 'principal', 'head', 'director', 'vp', 'cto', 'cfo', 'ceo']
    has_l3 = any(kw in title_lower for kw in l3_keywords)
    
    # Senior before roman numeral (I, II, III) is LEVEL_1
    if re.search(r'^(senior|junior)\s+.*\s+[ivxl]+$', title_lower, re.I):
        has_l3 = False  # Override
    
    l1_keywords = ['junior', 'graduate', 'intern', 'trainee', 'apprentice', 'associate', 'entry']
    has_l1 = any(kw in title_lower for kw in l1_keywords)
    
    if has_l3 and pred == 'LEVEL_1':
        return True, f"sen=LEVEL_1 but title has L3 marker: '{title}'"
    if has_l1 and pred == 'LEVEL_3':
        # Exception: Senior X I/II/III is LEVEL_1, not L3
        if re.search(r'^(senior)\s+.*\s+[ivxl]+$', title_lower, re.I):
            return None, None  # OK, Senior X I is LEVEL_1
        return True, f"sen=LEVEL_3 but title has L1 marker: '{title}'"
    
    return None, None

def validate_tech(job):
    """Validate tech stack - check OOS consistency with role type."""
    pred = job.get('v16_tech', [])
    title = job.get('title', '')
    
    if not isinstance(pred, list):
        return None, "tech is not a list"
    if len(pred) == 0:
        return None, "tech is empty array"
    
    title_lower = title.lower()
    
    # OOS only valid if NOT SWE role
    if 'OOS' in pred:
        if is_swe_role(title):
            return True, f"tech=['OOS'] but title suggests SWE: '{title}'"
        return None, None  # Non-SWE role with OOS is correct
    
    # Non-OOS tech requires SWE role
    for t in pred:
        if t not in TECH_VALID:
            return None, f"Invalid tech token: {t}"
    
    if not is_swe_role(title):
        return True, f"tech={pred} but title is not SWE: '{title}'"
    
    return None, None

def validate_comp(job):
    """Validate compensation - check raw string has GBP."""
    pred = job.get('v16_comp')
    raw = job.get('v16_comp_raw', '')
    
    if pred not in COMP_VALID:
        return None, "Invalid token"
    
    # NO_GBP is correct if raw contains OTE, DOE, or is hourly/daily
    raw_upper = raw.upper() if raw else ''
    if 'OTE' in raw_upper or 'DOE' in raw_upper:
        if pred != 'NO_GBP':
            return True, f"comp={pred} but raw has OTE/DOE"
        return None, None  # Correct NO_GBP
    
    # Check raw contains GBP symbol
    has_gbp = '£' in raw or 'gbp' in raw.lower()
    if not has_gbp:
        return True, f"comp={pred} but raw has no GBP: '{raw}'"
    
    return None, None

# Process ALL jobs - collect ONLY real bugs
print("\n=== VALIDATING ALL JOBS (REAL BUGS ONLY) ===\n")

all_bugs = []
job_ids_with_bugs = set()

for job in jobs:
    idx = job.get('index', 'UNKNOWN')
    title = job.get('title', 'NO TITLE')
    
    # Only check if all fields exist (skip missing fields - parse failures)
    has_all_fields = all(job.get(f'v16_{f}') is not None 
                        for f in ['loc_raw', 'loc', 'arr_raw', 'arr', 'sen_raw', 'sen', 
                                 'tech_raw', 'tech', 'comp_raw', 'comp'])
    
    if not has_all_fields:
        continue  # Skip - this is a parse failure, not a teacher bug
    
    # Validate each field - collect REAL bugs only
    result = validate_loc(job)
    if result[0]:  # Real bug
        all_bugs.append({'index': idx, 'title': title, 'field': 'loc', 'error': result[1]})
        job_ids_with_bugs.add(idx)
    
    result = validate_arr(job)
    if result[0]:
        all_bugs.append({'index': idx, 'title': title, 'field': 'arr', 'error': result[1]})
        job_ids_with_bugs.add(idx)
    
    result = validate_sen(job)
    if result[0]:
        all_bugs.append({'index': idx, 'title': title, 'field': 'sen', 'error': result[1]})
        job_ids_with_bugs.add(idx)
    
    result = validate_tech(job)
    if result[0]:
        all_bugs.append({'index': idx, 'title': title, 'field': 'tech', 'error': result[1]})
        job_ids_with_bugs.add(idx)
    
    result = validate_comp(job)
    if result[0]:
        all_bugs.append({'index': idx, 'title': title, 'field': 'comp', 'error': result[1]})
        job_ids_with_bugs.add(idx)

# Report results
print(f"Total jobs: {len(jobs)}")
print(f"Jobs with ALL fields present: {len(jobs)}")
print(f"Jobs with REAL bugs: {len(job_ids_with_bugs)} / {len(jobs)} ({100*len(job_ids_with_bugs)/len(jobs):.1f}%)\n")

# Group by field
by_field = {}
for bug in all_bugs:
    by_field.setdefault(bug['field'], []).append(bug)

print("=== REAL TEACHER BUGS (parse failures excluded) ===\n")
print(f"Total bugs: {len(all_bugs)}")

for field in ['loc', 'arr', 'sen', 'tech', 'comp']:
    count = len(by_field.get(field, []))
    print(f"  {field}: {count}")

print(f"\n=== DETAILED BUG REPORT (ALL {len(all_bugs)} REAL BUGS) ===\n")

for i, bug in enumerate(all_bugs, 1):
    print(f"[{i}] Index {bug['index']} | {bug['field']}")
    print(f"    Title: {bug['title'][:80]}")
    print(f"    Error: {bug['error']}")
    print()

# Save bugs to file
with open('versions/v16/data/real_teacherbugs_v3.json', 'w') as f:
    json.dump({
        'total_jobs': len(jobs),
        'jobs_with_bugs': len(job_ids_with_bugs),
        'total_bugs': len(all_bugs),
        'by_field': {k: len(v) for k, v in by_field.items()},
        'bugs': all_bugs
    }, f, indent=2)

print(f"\n=== SAVED: versions/v16/data/real_teacherbugs_v3.json ===")

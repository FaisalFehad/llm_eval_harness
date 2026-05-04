#!/usr/bin/env python3
"""Classify 83 critical XGBoost-only issues and add to verification report."""
import json
from collections import defaultdict

# Load data
with open('versions/v16/data/xgboost_audit_issues.json') as f:
    xgb_data = json.load(f)
with open('versions/v16/data/teacher_verification_report.json') as f:
    report = json.load(f)
with open('versions/v16/data/v16_teacher_labels.jsonl') as f:
    labels = [json.loads(l) for l in f]

# Get indices already in verification report
verified_indices = set()
for field, items in report.get('by_field', {}).items():
    for item in items:
        verified_indices.add(item['index'])

label_map = {j['index']: j for j in labels}

# Filter critical XGBoost-only issues (gap >= 0.9, not in verification)
critical = [i for i in xgb_data['issues'] if i['index'] not in verified_indices and i['gap'] >= 0.9]

# ── Classification helpers ──
L1_KW = ['junior', 'graduate', 'intern', 'trainee', 'apprentice', 'associate']
L3_KW = ['senior', 'lead', 'principal', 'head', 'director', 'vp', 'cto']
FLAG_KW = ['engineer i', 'staff', 'full stack']

def has_word(word, text):
    """Check if word appears as whole word in text (case-insensitive)."""
    return bool(re.search(rf'\b{word}\b', text.lower()))

import re

def classify_sen(job):
    t = job.get('title', '').lower()
    if not t:
        return 'no_title'
    # Senior before Roman numeral (I, II, III) is LEVEL_1, not L3 (e.g., Senior Data Engineer I)
    if re.search(r'^(senior\s+.*\s+)?[ivxl]+$', t):
        return 'senior_with_roman'
    # Check L3 keywords as whole words
    if any(has_word(k, t) for k in L3_KW):
        return 'has_l3_kw'
    # Check L1 keywords as whole words
    if any(has_word(k, t) for k in L1_KW):
        return 'has_l1_kw'
    # Check flag keywords
    for k in FLAG_KW:
        if re.search(r'\b' + k.replace(' ', r'\s+') + r'\b', t):
            return 'has_flag_kw'
    return 'plain_title'

def classify_comp(job, issue):
    raw = job.get('v16_comp_raw', '') or ''
    jd = job.get('jd_text', '')
    jd_lower = jd.lower()
    if 'ote' in jd_lower or 'on-target earn' in jd_lower:
        return 'ote_disqualifier'
    if 'hourly' in jd_lower or 'daily rates' in jd_lower or 'day rate' in jd_lower:
        return 'rate_disqualifier'
    if 'competitive' in jd_lower:
        return 'competitive_disqualifier'
    if '£' in raw or 'gbp' in jd_lower:
        if '+' in raw and '£' in raw:
            return 'plus_indicator'
        if 'per month' in jd_lower or 'pcm' in jd_lower:
            return 'monthly_rate'
        if 'up to' in jd_lower or 'up to ' in raw.lower():
            return 'up_to_only'
        return 'has_gbp'
    return 'no_gbp_signal'

def classify_tech(job, issue):
    t = job.get('title', '').lower()
    swe_kw = ['engineer', 'developer', 'architect', 'devops', 'software', 'programmer']
    non_swe = ['analyst', 'manager', 'support', 'admin', 'consultant', 'sales', 'market']
    is_swe = any(k in t for k in swe_kw)
    is_non_swe = any(k in t for k in non_swe)
    if is_non_swe and not is_swe:
        return 'non_swe_title'
    if is_swe:
        return 'swe_title'
    return 'ambiguous_title'

def classify_loc(job):
    loc = job.get('job_location', '') or ''
    jd = job.get('jd_text', '')
    loc_lower = (loc + ' ' + jd).lower()
    if 'london' in loc_lower:
        return 'london_signal'
    if 'ireland' in loc_lower or 'dublin' in loc_lower or 'us ' in loc_lower or 'usa' in loc_lower or 'germany' in loc_lower or 'france' in loc_lower:
        return 'outside_uk_signal'
    return 'uk_default'

def classify_arr(job):
    jd = job.get('jd_text', '').lower()
    if 'fully remote' in jd or 'work from home' in jd or '100% remote' in jd:
        return 'fully_remote_jd'
    if 'hybrid' in jd or 'some days' in jd or 'flexible' in jd:
        return 'hybrid_jd'
    if 'office-based' in jd or 'on-site' in jd or 'office based' in jd:
        return 'in_office_jd'
    return 'generic_jd'

# ── Classify each critical issue ──
new_items = []
counts = defaultdict(int)
field_counts = defaultdict(lambda: defaultdict(int))

for issue in critical:
    idx = issue['index']
    job = label_map.get(idx, {})
    field = issue['field']
    pred = issue['predicted']
    true_val = issue['true']
    gap = issue['gap']
    conf = issue['confidence']
    title = job.get('title', '')[:60]

    if field == 'sen':
        kw_cls = classify_sen(job)
        if pred in ['LEVEL_3'] and true_val == 'LEVEL_2':
            if 'has_l3_kw' in kw_cls:
                classification = 'by_design'
                reason = f'Teacher LEVEL_2 but title has L3 keyword (XGB caught it). class={kw_cls}'
            elif 'has_flag_kw' in kw_cls and pred == 'LEVEL_3':
                classification = 'ambiguous'
                reason = f'Staff/Engineer I/L2 ambiguous. XGB LEVEL_3, teacher LEVEL_2. class={kw_cls}'
            else:
                classification = 'ambiguous'
                reason = f'Plain title LEVEL_2 vs LEVEL_3. class={kw_cls}'
        elif pred in ['LEVEL_1'] and true_val == 'LEVEL_2':
            if 'has_l1_kw' in kw_cls:
                classification = 'teacher_bug'
                reason = f'Teacher LEVEL_2 but title has L1 keyword (XGB caught it). class={kw_cls}'
            else:
                classification = 'ambiguous'
                reason = f'L1 keyword mismatch. class={kw_cls}'
        elif pred in ['LEVEL_2'] and true_val == 'LEVEL_3':
            if 'has_l3_kw' in kw_cls:
                classification = 'teacher_bug'
                reason = f'Title has L3 keyword but teacher wrong. XGB=LEVEL_3, teacher=LEVEL_2. class={kw_cls}'
            else:
                classification = 'ambiguous'
                reason = f'Seniority boundary plain title. class={kw_cls}'
        elif pred in ['LEVEL_2'] and true_val == 'LEVEL_1':
            if 'has_l1_kw' in kw_cls:
                classification = 'teacher_bug'
                reason = f'Title has L1 keyword but teacher wrong. XGB=LEVEL_2, teacher=LEVEL_1. class={kw_cls}'
            else:
                classification = 'ambiguous'
                reason = f'Junior boundary plain title. class={kw_cls}'
        else:
            classification = 'ambiguous'
            reason = f'SEN mismatch: pred={pred} true={true_val} class={kw_cls}'

    elif field == 'comp':
        comp_cls = classify_comp(job, issue)
        if 'plus_indicator' in comp_cls:
            classification = 'teacher_bug'
            reason = f'£Xk+ not understood by XGB. raw={repr(job.get("v16_comp_raw","")[:50])}. class={comp_cls}'
        elif 'monthly_rate' in comp_cls:
            classification = 'teacher_bug'
            reason = f'Monthly rate not annualized. XGB likely NO_GBP, teacher bucket. class={comp_cls}'
        elif 'ote_disqualifier' in comp_cls:
            if true_val == 'NO_GBP':
                classification = 'teacher_bug'
                reason = f'Teacher missed OTE disqualifier. XGB=NO_GBP, teacher=OTHER. class={comp_cls}'
            else:
                classification = 'ambiguous'
                reason = f'OTE context with unusual bucket. class={comp_cls}'
        elif 'competitive_disqualifier' in comp_cls:
            classification = 'teacher_bug'
            reason = f'Competitive = NO_GBP per V16 rules. Teacher used bucket. class={comp_cls}'
        elif 'rate_disqualifier' in comp_cls:
            classification = 'teacher_bug'
            reason = f'Hourly/daily rate = NO_GBP per V16 rules. Teacher used bucket. class={comp_cls}'
        elif pred == 'NO_GBP' and true_val not in ['NO_GBP']:
            if 'no_gbp_signal' in comp_cls:
                classification = 'ambiguous'
                reason = f'No GBP signal found. Teacher used bucket but no £ found. class={comp_cls}'
            else:
                classification = 'ambiguous'
                reason = f'GBP GBP mismatch. class={comp_cls}'
        elif true_val == 'UP_TO_ONLY' and pred != 'UP_TO_ONLY':
            classification = 'ambiguous'
            reason = f'Up-to bucket mismatch. class={comp_cls}'
        else:
            classification = 'ambiguous'
            reason = f'Comp: pred={pred} true={true_val} class={comp_cls}'

    elif field == 'tech':
        tech_cls = classify_tech(job, issue)
        if 'non_swe_title' in tech_cls and pred == 'OOS' and 'OOS' not in (true_val if isinstance(true_val, list) else [true_val]):
            classification = 'teacher_bug'
            reason = f'Non-SWE title got AI_ML from teacher. XGB=OOS, teacher=OTHER. class={tech_cls}'
        elif 'non_swe_title' in tech_cls and pred == 'OOS':
            classification = 'by_design'
            reason = f'Non-SWE title, OOS agreed. class={tech_cls}'
        elif 'swe_title' in tech_cls and pred == 'OOS' and 'OOS' not in (true_val if isinstance(true_val, list) else [true_val]):
            classification = 'ambiguous'
            reason = f'SWE title, XGB=OOS but teacher has tech. class={tech_cls}'
        elif pred == 'OOS' and true_val == 'OOS':
            classification = 'by_design'
            reason = f'OOS agreed. class={tech_cls}'
        elif 'swe_title' in tech_cls and pred != 'OOS' and true_val == 'OOS':
            classification = 'ambiguous'
            reason = f'SWE title, XGB found tech but teacher OOS. class={tech_cls}'
        else:
            classification = 'ambiguous'
            reason = f'Tech: pred={pred} true={true_val} class={tech_cls}'

    elif field == 'loc':
        loc_cls = classify_loc(job)
        if 'london_signal' in loc_cls and pred != 'IN_LONDON':
            classification = 'ambiguous'
            reason = f'London signal but XGB not IN_LONDON. class={loc_cls}'
        elif 'outside_uk_signal' in loc_cls and pred not in ['OUTSIDE_UK']:
            classification = 'ambiguous'
            reason = f'Outside UK signal but XGB not OUTSIDE_UK. class={loc_cls}'
        else:
            classification = 'ambiguous'
            reason = f'Loc: pred={pred} true={true_val} class={loc_cls}'

    elif field == 'arr':
        arr_cls = classify_arr(job)
        if 'fully_remote_jd' in arr_cls and pred != 'REMOTE':
            classification = 'ambiguous'
            reason = f'Fully remote JD but XGB not REMOTE. class={arr_cls}'
        elif 'hybrid_jd' in arr_cls and pred != 'HYBRID':
            classification = 'ambiguous'
            reason = f'Hybrid JD signals but XGB not HYBRID. class={arr_cls}'
        elif 'in_office_jd' in arr_cls and pred != 'IN_OFFICE':
            classification = 'ambiguous'
            reason = f'In-office JD but XGB not IN_OFFICE. class={arr_cls}'
        else:
            classification = 'ambiguous'
            reason = f'Arr: pred={pred} true={true_val} class={arr_cls}'

    else:
        classification = 'unknown'
        reason = f'Unknown field: {field}'

    item = {
        'index': idx,
        'title': job.get('title', '')[:80],
        'field': field,
        'teacher': true_val,
        'rule': pred,
        'xgboost_conf': round(conf, 3),
        'xgboost_gap': round(gap, 3),
        'severity': 'CRITICAL',
        'classification': classification,
        'reason': reason,
        'job_location': job.get('job_location', '')[:100],
        'jd_snippet': job.get('jd_text', '')[:200],
    }
    new_items.append(item)
    counts[classification] += 1
    field_counts[field][classification] += 1

# ── Write to verification report ──
for item in new_items:
    field = item['field']
    if field not in report['by_field']:
        report['by_field'][field] = []
    report['by_field'][field].append(item)

# Update total
report['total'] = report.get('total', 0) + len(new_items)
# Add XGBoost-only count
report['xgb_only_critical_count'] = len(new_items)
# Merge by_classification
existing_bc = report.get('by_classification', {})
for k, v in counts.items():
    existing_bc[k] = existing_bc.get(k, 0) + v
report['by_classification'] = existing_bc

# Update by_field_summary
existing_bfs = report.get('by_field_summary', {})
for field, fc in field_counts.items():
    if field not in existing_bfs:
        existing_bfs[field] = {}
    for k, v in fc.items():
        existing_bfs[field][k] = existing_bfs[field].get(k, 0) + v
report['by_field_summary'] = existing_bfs

# Sort each field by gap descending
for field in report['by_field']:
    report['by_field'][field].sort(key=lambda x: -x['xgboost_gap'])

with open('versions/v16/data/teacher_verification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ── Print summary ──
print(f"=== CLASSIFICATION COMPLETE ===")
print(f"Added {len(new_items)} critical XGBoost-only issues to verification report")
print(f"\nBy classification:")
for k, v in sorted(counts.items()):
    print(f"  {k}: {v}")
print(f"\nBy field:")
for field, fc in sorted(field_counts.items()):
    print(f"  {field}:")
    for k, v in sorted(fc.items()):
        print(f"    {k}: {v}")

# ── Print a few examples per classification ──
for cls_type in ['teacher_bug', 'by_design']:
    examples = [i for i in new_items if i['classification'] == cls_type][:3]
    if examples:
        print(f"\n=== {cls_type.upper()} EXAMPLES ===")
        for i in examples:
            print(f"  [{i['field']}] idx={i['index']} | {i['title'][:50]}")
            print(f"    XGB={i['rule']} vs Teacher={i['teacher']} gap={i['xgboost_gap']}")
            print(f"    {i['reason']}")
            print()

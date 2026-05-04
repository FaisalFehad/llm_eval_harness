#!/usr/bin/env python3
"""
Compare V15 vs V16 labels on relabeled sample.
"""
import json
from collections import Counter

with open('versions/v16/data/sample_v16_relabeled.jsonl') as f:
    relabeled = [json.loads(l) for l in f]

print(f'Total relabeled: {len(relabeled)}')
print()

# Check for errors
errors = [r for r in relabeled if '_v16_raw_response' in r and r['_v16_raw_response'].startswith('{"_error')]
print(f'API errors: {len(errors)}')
if errors:
    for e in errors[:3]:
        print(f'  - {e.get("title", "")}: {e.get("_v16_raw_response", "")[:100]}')

# Check parse failures
parse_fails = [r for r in relabeled if '_v16_raw_response' in r and not r.get('v16_loc')]
print(f'Parse failures: {len(parse_fails)}')
print()

# Field comparison
fields = ['loc', 'arr', 'sen', 'tech', 'comp']
total_mismatches = 0
all_mismatch_jobs = set()

for field in fields:
    mismatches = []
    for r in relabeled:
        v15_labels = r.get('v15_labels', {})
        v15 = v15_labels.get(field)
        v16 = r.get(f'v16_{field}')
        
        if field == 'tech':
            v15_set = set(v15) if isinstance(v15, list) else set()
            v16_set = set(v16) if isinstance(v16, list) else set()
            if v15_set != v16_set:
                mismatches.append({
                    'title': r.get('title', '')[:60],
                    'v15': sorted(v15_set),
                    'v16': sorted(v16_set),
                    'added': sorted(v16_set - v15_set),
                    'removed': sorted(v15_set - v16_set)
                })
                all_mismatch_jobs.add(r.get('title', '')[:60])
        else:
            if v15 != v16:
                mismatches.append({
                    'title': r.get('title', '')[:60],
                    'v15': v15,
                    'v16': v16
                })
                all_mismatch_jobs.add(r.get('title', '')[:60])
    
    total_mismatches += len(mismatches)
    pct = len(mismatches)/len(relabeled)*100 if relabeled else 0
    print(f'{field.upper()}: {len(mismatches)}/{len(relabeled)} mismatches ({pct:.1f}%)')
    
    # Show first 5
    for m in mismatches[:5]:
        if field == 'tech':
            print(f'  - {m["title"]}: v15={m["v15"]} → v16={m["v16"]} (+{m["added"]}, -{m["removed"]})')
        else:
            print(f'  - {m["title"]}: v15={m["v15"]} → v16={m["v16"]}')
    if len(mismatches) > 5:
        print(f'  ... and {len(mismatches)-5} more')
    print()

print(f'='*60)
print(f'TOTAL: {total_mismatches} field-level mismatches across {len(relabeled)} jobs')
print(f'Jobs with any mismatch: {len(all_mismatch_jobs)}')
print(f'Jobs with zero mismatches: {len(relabeled) - len(all_mismatch_jobs)}')
print(f'='*60)

# Categorize tech changes
tech_changes = Counter()
tech_mismatches = []
with open('versions/v16/data/sample_v16_relabeled.jsonl') as f:
    for line in f:
        r = json.loads(line)
        v15 = r.get('v15_labels', {}).get('tech', [])
        v16 = r.get('v16_tech', [])
        v15_set = set(v15) if isinstance(v15, list) else set()
        v16_set = set(v16) if isinstance(v16, list) else set()
        if v15_set != v16_set:
            tech_mismatches.append({
                'title': r.get('title', '')[:60],
                'v15': sorted(v15_set),
                'v16': sorted(v16_set),
                'added': sorted(v16_set - v15_set),
                'removed': sorted(v15_set - v16_set)
            })

if tech_mismatches:
    print()
    print('TECH CHANGE BREAKDOWN:')
    added_counts = Counter()
    removed_counts = Counter()
    for m in tech_mismatches:
        for a in m['added']:
            added_counts[a] += 1
        for rm in m['removed']:
            removed_counts[rm] += 1
    
    print('  Added by V16:')
    for token, cnt in added_counts.most_common():
        print(f'    {token}: {cnt} jobs')
    print('  Removed by V16:')
    for token, cnt in removed_counts.most_common():
        print(f'    {token}: {cnt} jobs')

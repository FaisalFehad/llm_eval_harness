#!/usr/bin/env python3
import json

with open('versions/v16/data/xgboost_audit_v3.json') as f:
    data = json.load(f)

# Sample high-confidence teacher bugs
print('=== HIGH CONFIDENCE TEACHER BUGS (XGB confident, teacher wrong) ===')
for issue in data['issues'][:30]:
    if issue['gap'] > 0.95:
        print(f"Index {issue['index']}: {issue['field']} - {issue['title'][:50]}...")
        print(f"  XGB: {issue['predicted']} (conf={issue['confidence']}) vs Teacher: {issue['true']} (prob={issue['true_prob']})")
        if issue.get('features'):
            print(f"  Features: {issue['features']}")

print('\n=== TEACHER BUG ANALYSIS ===')
sen_issues = [i for i in data['issues'] if i['field'] == 'sen']
print(f'Seniority issues: {len(sen_issues)}')
for i in sen_issues[:5]:
    print(f"  {i['index']}: {i['title'][:40]} - {i['true']} vs {i['predicted']}")

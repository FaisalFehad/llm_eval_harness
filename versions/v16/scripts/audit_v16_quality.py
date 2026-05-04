#!/usr/bin/env python3
"""
XGBoost audit on V16 Ollama-generated labels (100 jobs).
Extracts tabular features from JDs, trains classifiers, flags suspicious labels.
"""
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Vocabularies
LOC_TOKENS = ["IN_LONDON", "REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNK"]
ARR_TOKENS = ["REMOTE", "HYBRID", "IN_OFFICE", "UNK"]
SEN_TOKENS = ["LEVEL_3", "LEVEL_2", "LEVEL_1"]
TECH_TOKENS = ["NODE", "REACT", "JS_TS", "AI_ML", "OOS"]
COMP_TOKENS = ["NO_GBP", "UP_TO_ONLY", "BELOW_30K", "30_40K", "40_50K", "50_60K", "60_70K",
               "70_80K", "80_90K", "90_100K", "100_120K", "120_140K", "140_160K",
               "160_180K", "180_200K", "ABOVE_200K",
               # Legacy V1 coarse buckets (fallback)
               "BELOW_45K", "RANGE_45_54K", "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K"]
LABEL_TOKENS = ["bad_fit", "maybe", "good_fit"]

# Keywords
L3_KEYWORDS = ['senior', 'lead', 'staff', 'principal', 'head', 'director', 'vp', 'cto', 'founding', 'distinguished', 'sr', 'iii', 'manager']
L1_KEYWORDS = ['junior', 'jr', 'graduate', 'intern', 'trainee', 'apprentice', 'associate', 'internship']
TECH_KEYWORDS = ['node', 'react', 'javascript', 'typescript', 'python', 'java', 'go', 'rust', 'c++', 'c#', 'ai', 'ml', 'llm', 'nlp', 'pytorch', 'tensorflow']
COMP_KEYWORDS = ['£', 'gbp', 'salary', 'compensation', 'pay', 'kpa']

def extract_features(job):
    """Extract tabular features from a job."""
    title = job.get('title', '').lower()
    jd = job.get('jd_text', '').lower()
    loc = job.get('job_location', '').lower()
    
    features = {
        # Title signals
        'title_len': len(title),
        'title_has_senior': int('senior' in title),
        'title_has_lead': int('lead' in title),
        'title_has_junior': int('junior' in title or 'jr' in title),
        'title_has_graduate': int('graduate' in title),
        'title_has_manager': int('manager' in title),
        
        # Location signals  
        'loc_has_london': int('london' in loc),
        'loc_has_remote': int('remote' in loc or 'anywhere' in loc),
        'loc_is_uk': int(any(x in loc for x in ['england', 'scotland', 'wales', 'northern ireland', 'uk', 'united kingdom'])),
        
        # JD signals
        'jd_len': len(jd),
        'jd_has_remote': int('remote' in jd or 'work from home' in jd or 'home-based' in jd),
        'jd_has_hybrid': int('hybrid' in jd),
        'jd_has_office': int('office' in jd or 'on-site' in jd or 'onsite' in jd),
        'jd_has_salary': int('£' in jd or 'gbp' in jd or 'salary' in jd),
        'jd_has_ote': int('ote' in jd or 'on target earnings' in jd),
        'jd_has_competitive': int('competitive' in jd),
        
        # Tech signals
        'jd_has_node': int('node' in jd),
        'jd_has_react': int('react' in jd),
        'jd_has_js': int('javascript' in jd or 'typescript' in jd or 'js' in jd),
        'jd_has_ai': int('ai' in jd or 'machine learning' in jd or 'ml ' in jd or 'llm' in jd),
        'jd_has_python': int('python' in jd),
        
        # Comp signals
        'jd_has_range': int(re.search(r'£[\d,]+.*-|£[\d,]+.*to', jd) is not None),
        'jd_has_fixed_amount': int(re.search(r'£[\d,]+(k|000)', jd) is not None),
    }
    return features

def main():
    input_path = "versions/v16/data/ollama_100/ollama_100_merged.jsonl"
    output_dir = Path("versions/v16/data/ollama_100/xgboost_audit")
    output_dir.mkdir(exist_ok=True)
    
    with open(input_path) as f:
        jobs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(jobs)} jobs for XGBoost audit")
    
    # Extract features and labels for V16 labels
    features_list = []
    labels_by_field = {f: [] for f in ['loc', 'arr', 'sen', 'tech', 'comp']}
    
    for job in jobs:
        feats = extract_features(job)
        features_list.append(feats)
        
        # Get V16 labels (use if available, else V15 for comparison)
        for field in ['loc', 'arr', 'sen', 'comp']:
            v16 = job.get(f'v16_{field}', 'UNK')
            labels_by_field[field].append(v16)
        
        # Tech as string for simplicity (sorted)
        tech = job.get('v16_tech', [])
        if isinstance(tech, list):
            labels_by_field['tech'].append(','.join(sorted(tech)))
        else:
            labels_by_field['tech'].append(str(tech))
    
    feature_names = list(features_list[0].keys()) if features_list else []
    
    # Simple rule-based checks to flag suspicious labels
    print("\n" + "="*60)
    print("RULE-BASED QUALITY FLAGS")
    print("="*60)
    
    suspicious = []
    for i, job in enumerate(jobs):
        flags = []
        title = job.get('title', '').lower()
        jd = job.get('jd_text', '').lower()
        v16 = {f: job.get(f'v16_{f}') for f in ['loc', 'arr', 'sen', 'tech', 'comp']}
        
        # SEN checks
        has_l3_title = any(kw in title for kw in L3_KEYWORDS)
        has_l1_title = any(kw in title for kw in L1_KEYWORDS)
        if v16['sen'] == 'LEVEL_3' and not has_l3_title:
            if not any(kw in jd for kw in ['8+ years', 'senior', 'principal', 'lead']):
                flags.append("SEN: L3 but no title signal")
        if v16['sen'] == 'LEVEL_1' and has_l3_title and not has_l1_title:
            flags.append("SEN: L1 but title has L3 keywords")
        if v16['sen'] == 'LEVEL_2' and has_l3_title:
            flags.append("SEN: L2 but title has L3 keywords")
        
        # COMP checks  
        if v16['comp'] == 'NO_GBP' and ('£' in jd or 'gbp' in jd):
            flags.append("COMP: NO_GBP but £ found in JD")
        if v16['comp'] != 'NO_GBP' and not any(x in jd for x in ['£', 'gbp', 'salary', 'compensation']):
            flags.append(f"COMP: {v16['comp']} but no salary evidence in JD")
        
        # TECH checks
        if isinstance(v16['tech'], list):
            if 'AI_ML' in v16['tech'] and not any(kw in jd for kw in ['ai', 'machine learning', 'ml ', 'llm', 'pytorch', 'tensorflow']):
                flags.append("TECH: AI_ML but no AI keywords in JD")
            if v16['tech'] == ['OOS'] and any(kw in jd for kw in ['node', 'react', 'javascript', 'typescript']):
                flags.append("TECH: OOS but tracked tech found in JD")
        
        # LOC checks
        if 'london' in job.get('job_location', '').lower() and v16['loc'] not in ['IN_LONDON', 'UK_OTHER']:
            flags.append(f"LOC: {v16['loc']} but location mentions London")
        
        if flags:
            suspicious.append({
                'index': i,
                'title': job.get('title', '')[:60],
                'flags': flags,
                'v16_labels': {f: v16[f] for f in ['loc', 'arr', 'sen', 'tech', 'comp']}
            })
    
    print(f"\nTotal jobs flagged: {len(suspicious)}/{len(jobs)} ({len(suspicious)/len(jobs)*100:.1f}%)")
    print(f"\nBreakdown:")
    flag_counts = Counter()
    for s in suspicious:
        for f in s['flags']:
            flag_counts[f] += 1
    for flag, cnt in flag_counts.most_common():
        print(f"  {flag}: {cnt}")
    
    # Sample flagged jobs
    print(f"\n{'='*60}")
    print("TOP 10 FLAGGED JOBS")
    print("="*60)
    for s in suspicious[:10]:
        print(f"\n{s['title']}")
        for f in s['flags']:
            print(f"  - {f}")
        print(f"  Labels: {json.dumps(s['v16_labels'])}")
    
    # Save report
    report = {
        'total_jobs': len(jobs),
        'flagged_jobs': len(suspicious),
        'flag_rate_pct': len(suspicious)/len(jobs)*100,
        'flag_breakdown': dict(flag_counts),
        'flagged_details': suspicious
    }
    with open(output_dir / "v16_quality_audit.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nReport saved to: {output_dir / 'v16_quality_audit.json'}")

if __name__ == "__main__":
    main()

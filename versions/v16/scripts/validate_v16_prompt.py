#!/usr/bin/env python3
"""
Validate V16 teacher prompt on stratified sample: compare V15 labels vs V16 labels.
Uses local OMLX server with gemma-4-31B-it-oQ4.
"""
import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from finetune.relabel_for_drift_v16 import relabel_jobs

def validate_sample(sample_path: str, prompt_path: str, output_path: str, api_key: str, model: str = "gemma-4-31B-it-oQ4"):
    """Re-label sample jobs and compare V15 vs V16 predictions."""

    with open(sample_path) as f:
        jobs = [json.loads(line) for line in f]

    print(f"Validating {len(jobs)} jobs with V16 teacher prompt...")

    # Relabel with V16 prompt
    v16_labels = relabel_jobs(
        jobs=jobs,
        prompt_path=prompt_path,
        api_key=api_key,
        model=model,
        batch_size=5,
        output_path=output_path.replace('.jsonl', '_raw.jsonl')
    )

    # Compare field by field
    fields = ['loc', 'arr', 'sen', 'tech', 'comp']
    mismatches = {f: [] for f in fields}
    v15_only = {'tech': []}  # Track V15 tech for comparison

    for i, job in enumerate(jobs):
        v15 = job.get('v15_labels', {})
        v16 = v16_labels[i] if i < len(v16_labels) else {}

        if not v16:
            print(f"  Job {i}: FAILED to get V16 label")
            continue

        for field in fields:
            v15_val = v15.get(field)
            v16_val = v16.get(field)

            if field == 'tech':
                # Compare as sorted lists
                v15_set = set(v15_val) if isinstance(v15_val, list) else set()
                v16_set = set(v16_val) if isinstance(v16_val, list) else set()
                if v15_set != v16_set:
                    mismatches[field].append({
                        'job_id': job.get('job_id', job.get('index')),
                        'title': job.get('title', ''),
                        'v15': sorted(v15_set),
                        'v16': sorted(v16_set),
                        'added': sorted(v16_set - v15_set),
                        'removed': sorted(v15_set - v16_set)
                    })
            else:
                if v15_val != v16_val:
                    mismatches[field].append({
                        'job_id': job.get('job_id', job.get('index')),
                        'title': job.get('title', ''),
                        'v15': v15_val,
                        'v16': v16_val
                    })

    # Print summary
    print("\n" + "="*60)
    print("V16 vs V15 LABEL COMPARISON")
    print("="*60)
    total_mismatches = 0
    for field in fields:
        cnt = len(mismatches[field])
        total_mismatches += cnt
        print(f"\n{field.upper()}: {cnt} mismatches / {len(jobs)} jobs ({cnt/len(jobs)*100:.1f}%)")
        for m in mismatches[field][:5]:  # Show first 5
            if field == 'tech':
                print(f"  - {m['title'][:50]}: v15={m['v15']} → v16={m['v16']} (added={m['added']}, removed={m['removed']})")
            else:
                print(f"  - {m['title'][:50]}: v15={m['v15']} → v16={m['v16']}")
        if len(mismatches[field]) > 5:
            print(f"  ... and {len(mismatches[field]) - 5} more")

    print(f"\n{'='*60}")
    print(f"TOTAL MISMATCHES: {total_mismatches} field-level changes across {len(jobs)} jobs")
    print(f"Jobs with any change: {len(set(m['job_id'] for fld in mismatches.values() for m in fld))}")

    # Save full report
    report = {
        'total_jobs': len(jobs),
        'total_mismatches': total_mismatches,
        'field_breakdown': {f: len(mismatches[f]) for f in fields},
        'details': mismatches
    }
    report_path = output_path.replace('.jsonl', '_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    return report

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default='versions/v16/data/sample_50_high_risk.jsonl')
    parser.add_argument('--prompt', default='versions/v16/prompts/teacher.txt')
    parser.add_argument('--output', default='versions/v16/data/v16_validation.jsonl')
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--model', default='gemma-4-31B-it-oQ4')
    args = parser.parse_args()

    validate_sample(args.sample, args.prompt, args.output, args.api_key, args.model)

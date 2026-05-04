#!/usr/bin/env python3
"""
Compare V1 vs V2 teacher labels using XGBoost audit.
Loads both v1 and v2 labels, flags jobs where v2 diverges from v1
and XGBoost confidence is high on v2's prediction.
Also flags v2-only jobs with high-confidence model disagreement.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "versions/v16/scripts")
from xgboost_audit_v16 import extract_features, encode_tech, compute_label, XGBoostAuditor, FIELD_ORDER, FIELDS

VALID_COMP_V2 = {
    "NO_GBP", "BELOW_30K", "30_40K", "40_50K", "50_60K", "60_70K",
    "70_80K", "80_90K", "90_100K", "100_120K", "120_140K",
    "140_160K", "160_180K", "180_200K", "ABOVE_200K",
}


def load_merged(v1_dir="versions/v16/data/full_relabel",
                  v2_dir="versions/v16/data/full_relabel_v2"):
    """Merge v1 and v2 labels keyed by index."""
    v1_by_idx = {}
    for path in glob.glob(f"{v1_dir}/batch_*.jsonl"):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                idx = d.get("index")
                if idx is not None:
                    v1_by_idx[idx] = d

    v2_by_idx = {}
    for path in glob.glob(f"{v2_dir}/batch_*.jsonl"):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                idx = d.get("index")
                if idx is not None:
                    v2_by_idx[idx] = d

    # Build unified list: start with all v2 jobs, overlay v1 when present
    merged = []
    for idx, v2j in sorted(v2_by_idx.items()):
        row = dict(v2j)
        if idx in v1_by_idx:
            v1j = v1_by_idx[idx]
            for field in ["loc", "arr", "sen", "tech", "comp"]:
                row[f"v1_{field}"] = v1j.get(f"v16_{field}")
                row[f"v1_{field}_raw"] = v1j.get(f"v16_{field}_raw")
            row["v1_exists"] = True
        else:
            row["v1_exists"] = False
        merged.append(row)

    return merged


def count_diffs(jobs):
    """Count field-by-field diffs between v1 and v2."""
    diffs = {f: 0 for f in FIELDS}
    total_overlap = 0
    for j in jobs:
        if not j.get("v1_exists"):
            continue
        total_overlap += 1
        for f in FIELDS:
            v1 = j.get(f"v1_{f}")
            v2 = j.get(f"v16_{f}")
            if f == "tech":
                v1 = tuple(sorted(v1)) if isinstance(v1, list) else ()
                v2 = tuple(sorted(v2)) if isinstance(v2, list) else ()
            if v1 != v2:
                diffs[f] += 1

    print(f"\nV1/V2 overlap: {total_overlap} jobs")
    for f in FIELDS:
        print(f"  {f}: {diffs[f]} diffs  ({diffs[f]/total_overlap*100:.1f}%)")
    return total_overlap, diffs


def flag_v2_invalidation(jobs, auditor, confidence_threshold=0.7, gap_threshold=0.2):
    """Flag v2 labels that XGBoost strongly disagrees with (possible bugs)."""
    issues = []
    for i, job in enumerate(jobs):
        for field in FIELDS:
            true_val = job.get(f"v16_{field}")
            if true_val is None:
                continue
            if field == "tech" and isinstance(true_val, list):
                true_val = encode_tech(true_val)

            probs = auditor.models[field][i]
            pred_idx = np.argmax(probs)
            pred_val = auditor.encoders[field].inverse_transform([pred_idx])[0]
            if pred_val == true_val:
                continue

            true_idx = None
            try:
                true_idx = list(auditor.encoders[field].classes_).index(true_val)
            except ValueError:
                continue

            confidence = probs[pred_idx]
            true_prob = probs[true_idx]
            gap = confidence - true_prob
            if confidence >= confidence_threshold and gap >= gap_threshold:
                issue = {
                    "index": job.get("index"),
                    "title": job.get("title")[:60],
                    "field": field,
                    "v2_label": true_val,
                    "model_pred": pred_val,
                    "confidence": round(float(confidence), 3),
                    "gap": round(float(gap), 3),
                }
                v1_label = job.get(f"v1_{field}")
                if field == "tech" and isinstance(v1_label, list):
                    v1_label = encode_tech(v1_label)
                issue["v1_label"] = v1_label if v1_label is not None else "N/A"
                issue["v1_agrees_model"] = (v1_label == pred_val)
                issue["v1_agrees_v2"] = (v1_label == true_val) if v1_label else None
                issues.append(issue)

    issues.sort(key=lambda x: (-x["gap"], -x["confidence"]))
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="versions/v16/data/v2_xgboost_audit.json")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--gap", type=float, default=0.2)
    args = parser.parse_args()

    print("Loading merged v1+v2 data...")
    jobs = load_merged()
    print(f"  Total v2 jobs: {len(jobs)}")

    total_overlap, diffs = count_diffs(jobs)

    print("\nExtracting features...")
    X, _ = extract_features(jobs)
    print(f"  Feature matrix: {X.shape}")

    print("\nTraining XGBoost on V2 labels...")
    auditor = XGBoostAuditor(cv_folds=5)
    auditor.train(X, jobs, prefix="v16_")

    print(f"\nFlagging suspicious V2 labels (conf≥{args.confidence}, gap≥{args.gap})...")
    issues = flag_v2_invalidation(jobs, auditor,
                                  confidence_threshold=args.confidence,
                                  gap_threshold=args.gap)

    by_field = {}
    for issue in issues:
        by_field.setdefault(issue["field"], []).append(issue)

    print(f"\nFound {len(issues)} suspicious V2 labels:")
    for field in FIELD_ORDER:
        if field in by_field:
            n_v1_agrees = sum(1 for x in by_field[field] if x.get("v1_agrees_model"))
            print(f"  {field}: {len(by_field[field])}  (v1 agrees w/ model: {n_v1_agrees})")

    report = {
        "total_v2_jobs": len(jobs),
        "v1_overlap": total_overlap,
        "v1_v2_diffs": diffs,
        "total_issues": len(issues),
        "by_field": {k: len(v) for k, v in by_field.items()},
        "issues": issues,
        "confidence_threshold": args.confidence,
        "gap_threshold": args.gap,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {args.output}")

    # Print top 15
    print("\nTop 15 most suspicious V2 labels:")
    for issue in issues[:15]:
        v1_hint = f" [v1={issue['v1_label']} {'✅' if issue['v1_agrees_model'] else ''}]"
        print(f"  [{issue['field']}] #{issue['index']} {issue['title'][:50]}")
        print(f"    V2: {issue['v2_label']}  |  MODEL: {issue['model_pred']} (conf={issue['confidence']}, gap={issue['gap']}){v1_hint}")


if __name__ == "__main__":
    main()

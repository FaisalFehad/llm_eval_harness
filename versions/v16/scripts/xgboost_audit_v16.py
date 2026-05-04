#!/usr/bin/env python3
"""XGBoost quality audit for V16 teacher labels.

Trains CV models on 50+ features extracted from job descriptions.
Flags teacher labels where model confidence is high but disagrees.
Cross-validates against deterministic prompt rules for validation.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from classify_xgb_critical import classify_sen

FIELD_ORDER = ["loc","arr","sen","tech","comp","label"]
FIELDS = ["loc","arr","sen","tech","comp"]
TECH_TOKENS = {"NODE","REACT","JS_TS","AI_ML","OOS"}
SALARY_PATTERNS = [
    re.compile(r'£(\d[\d,]+)(?:\.\d+)?\s*(k?)', re.I),
    re.compile(r'(\d[\d,]+)\s*(?:GBP|£)?\s*(k?)', re.I),
]

def extract_features(jobs):
    """Build feature matrix from job descriptions."""
    rows = []
    texts = []
    for j in jobs:
        text = (j.get("title","") + " " + j.get("jd_text","")).lower()
        texts.append(text)
        title_lower = j.get("title","").lower()

        # Title keywords
        has_senior = "senior" in title_lower
        # Senior before Roman numeral (I, II, III) is LEVEL_1 (e.g., Senior Data Engineer I)
        # Pattern: "Senior" anywhere, followed by optional words, then "I/II/III/IV" at end
        senior_before_roman = bool(re.search(r'^senior\s+.*\s+[ivxl]+$', title_lower))
        senior_general = has_senior and not senior_before_roman  # "Senior" elsewhere = L3
        manager = any(re.search(rf'\b{k}\b', title_lower) for k in ['manager', 'engineering manager', 'tech lead', 'engineering lead'])
        new_grad = any(re.search(rf'\b{k}\b', title_lower) for k in ['new grad', 'new graduate', 'grad'])
        it_support = any(re.search(rf'\b{k}\b', title_lower) for k in ['it support', 'support engineer', 'service desk', 'tier 1', '1st line', 'first line'])
        lead = any(re.search(rf'\b{k}\b', title_lower) for k in ['lead', 'principal', 'head', 'director'])
        junior = any(re.search(rf'\b{k}\b', title_lower) for k in ['junior', 'entry', 'trainee', 'apprentice', 'associate'])
        has_eng = any(re.search(rf'\b{k}\b', title_lower) for k in ["engineer","developer","architect","devops","software"])
        market = any(t in title_lower for t in ["market","sale","admin","support","exec","manage","consult"])

        # Tech keywords in JD
        node = "node" in text or "nodejs" in text or "node.js" in text
        react = "react" in text or "react.js" in text
        typescript = "typescript" in text or "type script" in text
        pytorch = "pytorch" in text or "torch" in text
        nlp = "nlp" in text or "natural language" in text or "hugging" in text
        ml = "machine learning" in text or "ml" in text.split() or "deep learning" in text
        ai = "ai" in text.split()  # standalone
        rest = "rest" in text or "api" in text or "aws" in text or "html" in text or "css" in text

        # Seniority signals in JD
        yrs = re.findall(r'(\d+)\s*[-+]?\s*(?:years?|yr)', text)
        max_yrs = max(int(y) for y in yrs) if yrs else 0
        exp = "experience" in text
        senior_in_jd = "senior" in j.get("jd_text","").lower()
        lead_in_jd = "lead" in j.get("jd_text","").lower()

        # Location
        london = "london" in text
        remote = "remote" in title_lower or "hybrid" in title_lower or "home" in text
        usa = any(t in text for t in ["united states", "usa", " us ", " california", " new york", " texas", " florida"])
        uk = any(t in text for t in ["united kingdom", "uk", " england", " scotland", " wales", "northern ireland"])
        outside_uk = any(t in text for t in ["germany", "france", "spain", "netherlands", "poland", "portugal", "ireland", "india", "dubai", "uae"])

        # Comp signals
        has_gbp = "£" in j.get("comp_raw","") or "gbp" in text
        has_ote = "ote" in text
        doe = "doe" in text or "depending" in text or "experience" in text
        salary_matches = []
        for pattern in SALARY_PATTERNS:
            for m in pattern.finditer(j.get("jd_text","")):
                num = m.group(1).replace(",","")
                k = m.group(2).lower() == "k"
                val = int(num) * 1000 if k or len(num) <= 3 else int(num)
                salary_matches.append(val)

        min_sal = min(salary_matches) if salary_matches else 0
        max_sal = max(salary_matches) if salary_matches else 0
        n_salaries = len(salary_matches)
        midpoint = (min_sal + max_sal) / 2 if salary_matches else 0

        # Company size/age signals
        startup = any(t in text for t in ["startup", "seed", "series a", "series b", "pre-seed"])
        enterprise = any(t in text for t in ["enterprise", "fortune 500", "global"])

        rows.append({
            "has_senior": int(has_senior),
            "senior_before_roman": int(senior_before_roman),
            "has_manager": int(manager),  # "Manager" = L3
            "has_new_grad": int(new_grad),  # "New Grad" = L1
            "has_it_support": int(it_support),  # "IT Support" = L1
            "has_lead": int(lead),
            "has_junior": int(junior),
            "has_eng": int(has_eng),
            "market_role": int(market),
            "node": int(node), "react": int(react), "typescript": int(typescript),
            "pytorch": int(pytorch), "nlp": int(nlp), "ml": int(ml), "ai": int(ai), "rest": int(rest),
            "max_yrs": max_yrs, "exp_mentioned": int(exp),
            "senior_in_jd": int(senior_in_jd), "lead_in_jd": int(lead_in_jd),
            "london": int(london), "remote": int(remote), "usa": int(usa), "uk": int(uk), "outside_uk": int(outside_uk),
            "has_gbp": int(has_gbp), "has_ote": int(has_ote), "doe": int(doe),
            "min_sal": min_sal, "max_sal": max_sal, "n_salaries": n_salaries, "midpoint": midpoint,
            "startup": int(startup), "enterprise": int(enterprise),
            "title_len": len(j.get("title","")),
            "jd_len": len(j.get("jd_text","")),
        })

    # TF-IDF on title + JD (top 50 features)
    vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1,2), stop_words="english")
    tfidf = vectorizer.fit_transform(texts).toarray()
    tfidf_cols = [f"tfidf_{i}" for i in range(tfidf.shape[1])]

    df = pd.DataFrame(rows)
    for i, col in enumerate(tfidf_cols):
        df[col] = tfidf[:,i]

    return df, vectorizer

def encode_tech(tech_arr):
    """Convert tech array to sorted comma string for classification."""
    if not tech_arr:
        return "OOS"
    return ",".join(sorted(set(tech_arr)))

def decode_tech(s):
    """Decode comma string back to list."""
    if s == "OOS":
        return ["OOS"]
    return s.split(",") if s else ["OOS"]

def compute_label(row):
    """Compute score/label from token fields."""
    loc_map = {"IN_LONDON":25,"REMOTE":25,"UK_OTHER":10,"OUTSIDE_UK":-50,"UNK":0}
    sen_map = {"LEVEL_3":25,"LEVEL_2":15,"LEVEL_1":0}
    tech_map = {"NODE":10,"REACT":5,"JS_TS":5,"AI_ML":10,"OOS":0}
    comp_map = {"UP_TO_ONLY":0,"NO_GBP":0,"BELOW_30K":-30,"30_40K":-20,"40_50K":-10,"50_60K":0,"60_70K":5,"70_80K":10,"80_90K":15,"90_100K":20,"100_120K":25,"120_140K":25,"140_160K":25,"160_180K":25,"180_200K":25,"ABOVE_200K":25,"BELOW_45K":-30,"RANGE_45_54K":0,"RANGE_55_74K":5,"RANGE_75_99K":15,"ABOVE_100K":25}

    loc = loc_map.get(row.get("loc","UNK"),0)
    is_oos = row.get("tech") == ["OOS"] or row.get("tech") == [] or (isinstance(row.get("tech"),list) and "OOS" in row.get("tech",[]))
    role = 0 if is_oos else sen_map.get(row.get("sen","LEVEL_1"),0)
    tech_score = 0 if is_oos else sum(tech_map.get(t,0) for t in (row.get("tech") or []))
    comp = comp_map.get(row.get("comp","NO_GBP"),0)

    score = max(0, min(100, loc + role + tech_score + comp))
    if score >= 70: return "good_fit"
    if score >= 50: return "maybe"
    return "bad_fit"

class XGBoostAuditor:
    def __init__(self, cv_folds=5):
        self.cv_folds = cv_folds
        self.models = {}
        self.encoders = {}

    def train(self, X, jobs, prefix="v16_"):
        """Train CV models for each field."""
        for field in FIELDS:
            labels = []
            for j in jobs:
                val = j.get(f"{prefix}{field}")
                if field == "tech" and isinstance(val, list):
                    labels.append(encode_tech(val))
                else:
                    labels.append(val if val is not None else "UNK")

            le = LabelEncoder()
            y = le.fit_transform(labels)
            self.encoders[field] = le

            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            oof_probs = np.zeros((len(y), len(le.classes_)))

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y[train_idx]

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val)

                params = {
                    "objective": "multi:softprob",
                    "num_class": len(le.classes_),
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "eval_metric": "mlogloss",
                }
                model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
                oof_probs[val_idx] = model.predict(dval).reshape(len(val_idx), -1)

            self.models[field] = oof_probs
            acc = np.mean(np.argmax(oof_probs, axis=1) == y)
            print(f"  {field}: CV accuracy = {acc:.3f} ({len(le.classes_)} classes)")

        # label is computed, so we train on computed labels
        labels = [compute_label({f: j.get(f"{prefix}{f}") for f in FIELDS}) for j in jobs]
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.encoders["label"] = le

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        oof_probs = np.zeros((len(y), len(le.classes_)))

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y[train_idx]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val)
            params = {
                "objective": "multi:softprob",
                "num_class": len(le.classes_),
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }
            model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
            oof_probs[val_idx] = model.predict(dval).reshape(len(val_idx), -1)

        self.models["label"] = oof_probs
        acc = np.mean(np.argmax(oof_probs, axis=1) == y)
        print(f"  label: CV accuracy = {acc:.3f} ({len(le.classes_)} classes)")

    def flag_issues(self, jobs, prefix="v16_", confidence_threshold=0.7, gap_threshold=0.3):
        """Flag suspicious labels where model is confident but disagrees."""
        issues = []
        fields = [f for f in FIELD_ORDER if f in self.models]

        for i, job in enumerate(jobs):
            for field in fields:
                true_val = job.get(f"{prefix}{field}")
                if field == "tech" and isinstance(true_val, list):
                    true_val = encode_tech(true_val)
                if true_val is None:
                    continue

                probs = self.models[field][i]
                pred_idx = np.argmax(probs)
                pred_val = self.encoders[field].inverse_transform([pred_idx])[0]

                # Skip if predicted matches
                if pred_val == true_val:
                    continue

                confidence = probs[pred_idx]
                # Confidence gap between predicted and true
                true_idx = None
                try:
                    true_idx = list(self.encoders[field].classes_).index(true_val)
                except ValueError:
                    continue  # True value not in training distribution

                true_prob = probs[true_idx]
                gap = confidence - true_prob

                # Flag: high confidence on prediction + large gap over true value
                if confidence >= confidence_threshold and gap >= gap_threshold:
                    # Get classification features for seniority
                    kw_cls = classify_sen(job)
                    features = {"classify_sen": kw_cls} if field == "sen" else {}
                    
                    issues.append({
                        "index": job.get("index"),
                        "title": job.get("title"),
                        "field": field,
                        "predicted": pred_val,
                        "true": true_val,
                    "confidence": round(float(confidence), 3),
                    "true_prob": round(float(true_prob), 3),
                    "gap": round(float(gap), 3),
                    "job_id": job.get("job_id"),
                    "features": features,
                })

        # Sort by gap descending (most suspicious first)
        issues.sort(key=lambda x: (-x["gap"], -x["confidence"]))
        return issues


def main():
    parser = argparse.ArgumentParser(description="XGBoost audit for V16 teacher labels")
    parser.add_argument("--input", default="versions/v16/data/v16_teacher_labels.jsonl")
    parser.add_argument("--output", default="versions/v16/data/xgboost_audit_issues.json")
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--gap", type=float, default=0.3)
    parser.add_argument("--prefix", default="v16_")
    args = parser.parse_args()

    with open(args.input) as f:
        jobs = [json.loads(l) for l in f]

    print(f"Loaded {len(jobs)} jobs. Extracting features...")
    X, vectorizer = extract_features(jobs)
    print(f"  Feature matrix: {X.shape}")

    print("Training XGBoost models (CV)...")
    auditor = XGBoostAuditor(cv_folds=5)
    auditor.train(X, jobs, prefix=args.prefix)

    print(f"\nFlagging suspicious labels (confidence≥{args.confidence}, gap≥{args.gap})...")
    issues = auditor.flag_issues(jobs, prefix=args.prefix, confidence_threshold=args.confidence, gap_threshold=args.gap)

    # Group by field
    by_field = {}
    for issue in issues:
        field = issue["field"]
        by_field.setdefault(field, []).append(issue)

    print(f"\nFound {len(issues)} suspicious labels:")
    for field in FIELD_ORDER:
        if field in by_field:
            print(f"  {field}: {len(by_field[field])}")

    report = {
        "total_jobs": len(jobs),
        "total_issues": len(issues),
        "by_field": {k: len(v) for k, v in by_field.items()},
        "issues": issues,
        "confidence_threshold": args.confidence,
        "gap_threshold": args.gap,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")

    # Print top 10 most suspicious
    print("\nTop 10 most suspicious labels:")
    for issue in issues[:10]:
        print(f"  [{issue['field']}] {issue['title'][:40]}")
        print(f"    Predicted: {issue['predicted']} (conf={issue['confidence']})")
        print(f"    True:      {issue['true']} (prob={issue['true_prob']}, gap={issue['gap']})")

if __name__ == "__main__":
    main()

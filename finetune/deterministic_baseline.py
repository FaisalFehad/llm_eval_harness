#!/usr/bin/env python3
"""
Deterministic regex/rule-based baseline classifier for V7 job-fit scoring.

Classifies jobs across all 5 fields (loc, arr, sen, tech, comp) using
pattern matching on job_location, title, and jd_text. Computes scores
and labels using the V7 scoring rules, then measures accuracy against
golden labels.

Usage:
    python3 finetune/deterministic_baseline.py --test-file data/v7/test_labeled.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ── Score Maps (from semantic_tokens_v7.py) ──────────────────────────────────

LOCATION_MAP = {
    "IN_LONDON": 25,
    "REMOTE": 25,
    "UK_OTHER": 10,
    "OUTSIDE_UK": -50,
    "UNK": 0,
}

SENIORITY_MAP = {
    "LEVEL_3": 25,
    "LEVEL_2": 15,
    "LEVEL_1": 0,
}

TECH_INDIVIDUAL_MAP = {
    "OOS": 0,
    "NODE": 10,
    "REACT": 5,
    "JS_TS": 5,
    "AI_ML": 10,
}

COMP_MAP = {
    "NO_GBP": 0,
    "UP_TO_ONLY": 0,
    "BELOW_45K": -30,
    "RANGE_45_54K": 0,
    "RANGE_55_74K": 5,
    "RANGE_75_99K": 15,
    "ABOVE_100K": 25,
}

# ── Classification Rules ─────────────────────────────────────────────────────

# UK cities for UK_OTHER detection (lowercase)
UK_CITIES = [
    "manchester", "birmingham", "bristol", "edinburgh", "glasgow",
    "leeds", "cambridge", "oxford", "liverpool", "newcastle",
    "sheffield", "nottingham", "cardiff", "belfast", "southampton",
    "brighton", "leicester", "coventry", "bath", "york",
    "aberdeen", "dundee", "swansea", "exeter", "norwich",
    "reading", "portsmouth", "warwick", "milton keynes",
    "chester", "derby", "stoke", "sunderland", "hull",
    "wolverhampton", "plymouth", "luton", "slough", "guildford",
    "basingstoke", "swindon", "cheltenham", "harrogate",
    "bournemouth", "ipswich", "peterborough", "northampton",
    "worcester", "stevenage", "watford", "crawley", "maidenhead",
]

# Non-UK indicators (lowercase)
NON_UK_INDICATORS = [
    # Countries
    "united states", "usa", "india", "germany", "france", "australia",
    "canada", "netherlands", "ireland", "spain", "italy", "japan",
    "china", "brazil", "mexico", "israel", "switzerland", "austria",
    "sweden", "norway", "denmark", "finland", "poland", "portugal",
    "belgium", "czech", "romania", "hungary", "south korea", "korea",
    "singapore", "hong kong", "dubai", "uae", "united arab emirates",
    "qatar", "saudi", "egypt", "south africa", "nigeria", "kenya",
    "new zealand", "philippines", "vietnam", "thailand", "indonesia",
    "malaysia", "taiwan", "pakistan",
    # US cities / regions
    "new york", "san francisco", "los angeles", "seattle", "chicago",
    "austin", "boston", "denver", "atlanta", "miami", "dallas",
    "houston", "phoenix", "san diego", "san jose", "portland",
    "minneapolis", "philadelphia", "detroit", "washington dc",
    "washington, dc", "virginia", "california", "texas", "florida",
    "colorado", "massachusetts", "georgia", "north carolina",
    "bay area", "silicon valley",
    # European cities
    "berlin", "munich", "hamburg", "paris", "amsterdam", "barcelona",
    "madrid", "rome", "milan", "zurich", "stockholm", "oslo",
    "copenhagen", "helsinki", "dublin", "lisbon", "prague", "warsaw",
    "budapest", "vienna", "brussels",
    # Asian/other cities
    "bangalore", "hyderabad", "mumbai", "delhi", "pune", "chennai",
    "kolkata", "noida", "gurgaon", "toronto", "vancouver", "montreal",
    "sydney", "melbourne", "tel aviv", "tokyo",
]


def classify_loc(job_location: str) -> str:
    """Classify location from the job_location field."""
    loc = (job_location or "").strip().lower()

    if not loc:
        return "UNK"

    # London check — match "london" but not "londonderry" etc.
    if re.search(r'\blondon\b', loc):
        return "IN_LONDON"

    # Remote check
    if re.search(r'\bremote\b', loc):
        return "REMOTE"

    # Non-UK check (before UK check to handle e.g., "Dublin, Ireland")
    for indicator in NON_UK_INDICATORS:
        if indicator in loc:
            return "OUTSIDE_UK"

    # US state abbreviations: ", XX" at end (2-letter uppercase)
    if re.search(r',\s*[A-Z]{2}\s*$', job_location or ""):
        # Could be US state — but could also be "Cardiff, UK"
        # Check if it's UK first
        if re.search(r'\buk\b|\bunited kingdom\b|\bengland\b|\bscotland\b|\bwales\b|\bnorthern ireland\b', loc):
            return "UK_OTHER"
        return "OUTSIDE_UK"

    # UK cities
    for city in UK_CITIES:
        if city in loc:
            return "UK_OTHER"

    # Explicit UK indicators
    if re.search(r'\buk\b|\bunited kingdom\b|\bengland\b|\bscotland\b|\bwales\b|\bnorthern ireland\b|\bbritain\b', loc):
        return "UK_OTHER"

    return "UNK"


def classify_arr(jd_text: str) -> str:
    """Classify work arrangement from JD text."""
    text = (jd_text or "").lower()

    has_hybrid = bool(re.search(r'\bhybrid\b', text))
    has_remote = bool(re.search(r'\bremote\b', text))
    has_onsite = bool(re.search(r'\bon[- ]?site\b|\bin[- ]office\b', text))

    # Hybrid is the most specific signal — if mentioned, it's hybrid
    if has_hybrid:
        return "HYBRID"

    # On-site / in-office
    if has_onsite:
        return "IN_OFFICE"

    # Remote without hybrid
    if has_remote:
        return "REMOTE"

    return "UNK"


def classify_sen(title: str) -> str:
    """Classify seniority from job title."""
    t = (title or "").lower()

    # LEVEL_3 keywords
    if re.search(r'\b(?:lead|principal|staff|director|head\s+of|vp|cto|chief|architect|manager|founding)\b', t):
        return "LEVEL_3"

    # LEVEL_2 keywords
    if re.search(r'\bsenior\b|\bsr\.|\bsr\s', t):
        return "LEVEL_2"

    return "LEVEL_1"


def classify_tech(jd_text: str) -> list[str]:
    """Classify tech stack from JD text. Returns array of tech tokens."""
    text = (jd_text or "").lower()
    tokens = []

    # NODE detection
    if re.search(r'\bnode\.?js\b|\bnode\s+js\b|\bnodejs\b|\bexpress\.?js\b|\bexpressjs\b|\bnest\.?js\b|\bnestjs\b|\bkoa\.?js\b|\bfastify\b', text):
        tokens.append("NODE")

    # REACT detection
    if re.search(r'\breact\b|\breact\.?js\b|\breactjs\b|\bnext\.?js\b|\bnextjs\b|\breact\s*native\b', text):
        tokens.append("REACT")

    # JS_TS detection
    if re.search(r'\btypescript\b|\bjavascript\b|\bjs/ts\b|\bts/js\b|\bangular\b|\bvue\.?js\b|\bvuejs\b|\bsvelte\b', text):
        tokens.append("JS_TS")

    # AI_ML detection
    if re.search(
        r'\bmachine\s+learning\b|\bdeep\s+learning\b|\bartificial\s+intelligence\b'
        r'|\bnatural\s+language\s+processing\b|\bcomputer\s+vision\b'
        r'|\bneural\s+network\b|\bllm\b|\blarge\s+language\s+model\b'
        r'|\btensorflow\b|\bpytorch\b|\bml\s+engineer\b|\bml\s+platform\b'
        r'|\b(?<!em)ai\b(?!\s*(?:r|m|ds))',  # "ai" but not "email", "aim", "aids"
        text
    ):
        tokens.append("AI_ML")

    if not tokens:
        return ["OOS"]

    return tokens


def _parse_salary_value(s: str) -> float | None:
    """
    Parse a salary string like '£75,000', '£75k', '75000', '75K' into a number.
    Returns None if unparseable.
    """
    s = s.strip().replace(",", "").replace(" ", "")
    # Remove currency symbols
    s = re.sub(r'[£$€]', '', s)

    m = re.match(r'^(\d+(?:\.\d+)?)\s*[kK]$', s)
    if m:
        return float(m.group(1)) * 1000

    m = re.match(r'^(\d+(?:\.\d+)?)$', s)
    if m:
        val = float(m.group(1))
        # If it looks like a "per hour" or too small for annual, skip
        if val < 1000:
            return val * 1000  # Likely in thousands (e.g., "75" meaning 75k)
        return val

    return None


def classify_comp(jd_text: str) -> str:
    """Classify compensation from JD text."""
    text = jd_text or ""
    text_lower = text.lower()

    # Check for GBP presence (£ sign or "gbp")
    has_gbp = bool(re.search(r'£|gbp', text_lower))

    if not has_gbp:
        # Check for non-GBP currencies
        if re.search(r'\$|usd|eur|€', text_lower):
            return "NO_GBP"
        # No salary info at all
        return "NO_GBP"

    # We have GBP — look for salary patterns

    # Pattern: "up to £XXX" without a range
    up_to_match = re.search(r'up\s+to\s+£\s*([\d,]+(?:\.\d+)?)\s*[kK]?', text, re.IGNORECASE)
    # Pattern: salary range "£XX,XXX - £YY,YYY" or "£XXk - £YYk"
    range_pattern = r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?\s*(?:[-–—to]+)\s*£?\s*([\d,]+(?:\.\d+)?)\s*[kK]?'
    range_match = re.search(range_pattern, text, re.IGNORECASE)

    if range_match:
        low = _parse_salary_value(range_match.group(1))
        high = _parse_salary_value(range_match.group(2))
        if low is not None and high is not None:
            # Ensure low < high
            if low > high:
                low, high = high, low
            midpoint = (low + high) / 2
            return _midpoint_to_comp(midpoint)
        elif high is not None:
            return _midpoint_to_comp(high)
        elif low is not None:
            return _midpoint_to_comp(low)

    if up_to_match:
        val = _parse_salary_value(up_to_match.group(1))
        if val is not None:
            return "UP_TO_ONLY"

    # Single salary value: "£XX,XXX" or "£XXk" (not in a range)
    single_match = re.search(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?', text)
    if single_match:
        val = _parse_salary_value(single_match.group(1))
        if val is not None:
            # Check if it's "up to" phrasing even if we didn't catch it above
            context_start = max(0, single_match.start() - 30)
            context = text[context_start:single_match.start()].lower()
            if "up to" in context:
                return "UP_TO_ONLY"
            return _midpoint_to_comp(val)

    return "NO_GBP"


def _midpoint_to_comp(midpoint: float) -> str:
    """Convert a salary midpoint to a comp token."""
    if midpoint < 45000:
        return "BELOW_45K"
    elif midpoint < 55000:
        return "RANGE_45_54K"
    elif midpoint < 75000:
        return "RANGE_55_74K"
    elif midpoint < 100000:
        return "RANGE_75_99K"
    else:
        return "ABOVE_100K"


# ── Score Computation (mirrors semantic_tokens_v7.py) ────────────────────────

def compute_score_and_label(loc: str, sen: str, tech: list[str], comp: str) -> dict:
    """Compute numeric score and label from classified tokens."""
    loc_score = LOCATION_MAP.get(loc, 0)
    comp_score = COMP_MAP.get(comp, 0)

    is_oos = "OOS" in tech or len(tech) == 0
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech)
    role_score = 0 if is_oos else SENIORITY_MAP.get(sen, 0)

    raw = loc_score + role_score + tech_score + comp_score
    score = max(0, min(100, raw))

    if score >= 70:
        label = "good_fit"
    elif score >= 50:
        label = "maybe"
    else:
        label = "bad_fit"

    return {
        "loc_score": loc_score,
        "role_score": role_score,
        "tech_score": tech_score,
        "comp_score": comp_score,
        "score": score,
        "label": label,
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def classify_job(job: dict) -> dict:
    """Classify a single job using deterministic rules."""
    loc = classify_loc(job.get("job_location", ""))
    arr = classify_arr(job.get("jd_text", ""))
    sen = classify_sen(job.get("title", ""))
    tech = classify_tech(job.get("jd_text", ""))
    comp = classify_comp(job.get("jd_text", ""))

    result = compute_score_and_label(loc, sen, tech, comp)
    result["loc"] = loc
    result["arr"] = arr
    result["sen"] = sen
    result["tech"] = tech
    result["comp"] = comp

    return result


def evaluate(test_file: str, verbose: bool = False) -> None:
    """Run the baseline classifier on the test set and report accuracy."""
    jobs = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    print(f"Loaded {len(jobs)} jobs from {test_file}\n")

    # Classify all jobs
    predictions = []
    for job in jobs:
        pred = classify_job(job)
        predictions.append(pred)

    # ── Per-field accuracy ────────────────────────────────────────────────
    fields = ["loc", "arr", "sen", "tech", "comp"]
    field_correct = {f: 0 for f in fields}
    field_total = len(jobs)

    # Per-field confusion tracking
    field_errors = {f: [] for f in fields}

    for i, (job, pred) in enumerate(zip(jobs, predictions)):
        for f in fields:
            golden = job[f]
            predicted = pred[f]
            if f == "tech":
                # Tech is an array — compare as sorted lists
                golden_sorted = sorted(golden) if isinstance(golden, list) else [golden]
                pred_sorted = sorted(predicted) if isinstance(predicted, list) else [predicted]
                if golden_sorted == pred_sorted:
                    field_correct[f] += 1
                else:
                    field_errors[f].append({
                        "idx": i,
                        "job_id": job.get("job_id", "?"),
                        "golden": golden,
                        "predicted": predicted,
                        "title": job.get("title", ""),
                    })
            else:
                if golden == predicted:
                    field_correct[f] += 1
                else:
                    field_errors[f].append({
                        "idx": i,
                        "job_id": job.get("job_id", "?"),
                        "golden": golden,
                        "predicted": predicted,
                        "title": job.get("title", ""),
                    })

    # ── Label accuracy ────────────────────────────────────────────────────
    label_correct = 0
    label_counts = Counter()
    label_correct_counts = Counter()
    confusion = Counter()  # (golden, predicted)

    for job, pred in zip(jobs, predictions):
        golden_label = job["label"]
        pred_label = pred["label"]
        label_counts[golden_label] += 1
        confusion[(golden_label, pred_label)] += 1
        if golden_label == pred_label:
            label_correct += 1
            label_correct_counts[golden_label] += 1

    # ── Print Results ─────────────────────────────────────────────────────
    print("=" * 70)
    print("DETERMINISTIC BASELINE RESULTS")
    print("=" * 70)

    print(f"\nOverall Label Accuracy: {label_correct}/{field_total} "
          f"= {100 * label_correct / field_total:.1f}%\n")

    # Per-field accuracy table
    print("Per-field Accuracy:")
    print(f"  {'Field':<8} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Errors':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for f in fields:
        acc = 100 * field_correct[f] / field_total
        errors = field_total - field_correct[f]
        print(f"  {f:<8} {field_correct[f]:>8} {field_total:>8} {acc:>9.1f}% {errors:>8}")

    # Per-label accuracy
    print(f"\nPer-label Accuracy:")
    label_order = ["good_fit", "maybe", "bad_fit"]
    print(f"  {'Label':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for lbl in label_order:
        total = label_counts[lbl]
        correct = label_correct_counts[lbl]
        acc = 100 * correct / total if total > 0 else 0
        print(f"  {lbl:<12} {correct:>8} {total:>8} {acc:>9.1f}%")

    # Confusion matrix
    print(f"\nLabel Confusion Matrix (rows=golden, cols=predicted):")
    print(f"  {'':>12} {'good_fit':>10} {'maybe':>10} {'bad_fit':>10} {'total':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for g in label_order:
        row = [confusion[(g, p)] for p in label_order]
        total = sum(row)
        print(f"  {g:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10} {total:>8}")

    # ── Per-field error distribution ──────────────────────────────────────
    print(f"\nPer-field Error Distribution:")
    for f in fields:
        errors = field_errors[f]
        if not errors:
            print(f"\n  {f}: 0 errors")
            continue
        print(f"\n  {f}: {len(errors)} errors")
        # Show distribution of golden→predicted
        transition_counts = Counter()
        for e in errors:
            g = str(e["golden"])
            p = str(e["predicted"])
            transition_counts[f"{g} → {p}"] += 1
        for transition, count in transition_counts.most_common(10):
            print(f"    {transition}: {count}")

    # ── Verbose: show individual errors ───────────────────────────────────
    if verbose:
        print(f"\n{'=' * 70}")
        print("DETAILED ERRORS")
        print(f"{'=' * 70}")

        # Label mismatches with score details
        print(f"\nLabel Mismatches:")
        for i, (job, pred) in enumerate(zip(jobs, predictions)):
            if job["label"] != pred["label"]:
                print(f"\n  Job {i} [{job.get('job_id', '?')}]: {job.get('title', '?')}")
                print(f"    Golden: {job['label']} (score={job['score']})")
                print(f"    Predicted: {pred['label']} (score={pred['score']})")
                # Show which fields differ
                diffs = []
                for f in fields:
                    g = job[f]
                    p = pred[f]
                    if f == "tech":
                        if sorted(g) != sorted(p):
                            diffs.append(f"tech: {g}→{p}")
                    elif g != p:
                        diffs.append(f"{f}: {g}→{p}")
                if diffs:
                    print(f"    Field diffs: {', '.join(diffs)}")
                else:
                    print(f"    Field diffs: NONE (score calc difference only)")

    # ── Summary comparison with student models ────────────────────────────
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH STUDENT MODELS (from CLAUDE.md)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<30} {'Label Acc':>10} {'Tech':>8} {'Comp':>8} {'Loc':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    baseline_acc = 100 * label_correct / field_total
    tech_acc = 100 * field_correct["tech"] / field_total
    comp_acc = 100 * field_correct["comp"] / field_total
    loc_acc = 100 * field_correct["loc"] / field_total
    print(f"  {'Deterministic Baseline':<30} {baseline_acc:>9.1f}% {tech_acc:>7.1f}% {comp_acc:>7.1f}% {loc_acc:>7.1f}%")
    print(f"  {'V7 0.5B (2000 iters)':<30} {'84.9%':>10} {'70.3%':>8} {'71.7%':>8} {'95.8%':>8}")
    print(f"  {'V7 1.5B (2000 iters)':<30} {'85.4%':>10} {'70.4%':>8} {'77.9%':>8} {'97.3%':>8}")
    print(f"  {'V5.1 baseline':<30} {'83.9%':>10} {'72.5%':>8} {'78.5%':>8} {'92.6%':>8}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic regex/rule-based baseline classifier for V7 jobs"
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to test_labeled.jsonl with golden labels",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-job error information",
    )
    args = parser.parse_args()

    if not Path(args.test_file).exists():
        print(f"ERROR: File not found: {args.test_file}", file=sys.stderr)
        sys.exit(1)

    evaluate(args.test_file, verbose=args.verbose)


if __name__ == "__main__":
    main()

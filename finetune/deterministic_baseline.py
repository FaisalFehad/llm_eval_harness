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


def classify_loc(job_location: str, jd_text: str = "") -> str:
    """Classify location from the job_location field.

    Phase 1A improvements (V12):
    - "Anywhere" + remote check in JD text
    - Non-UK check before REMOTE (fixes "Paris, France (Remote)")
    - "Northern Ireland" handled before generic "ireland" non-UK match
    - "Little London" exclusion
    """
    loc = (job_location or "").strip().lower()

    if not loc:
        return "UNK"

    # London check — match "london" but not "londonderry" or "little london"
    if re.search(r'\blondon\b', loc):
        # Exclude "Little London" (a village, not London)
        if "little london" in loc:
            # Fall through to UK checks below
            pass
        else:
            return "IN_LONDON"

    # "Northern Ireland" → UK_OTHER (must check BEFORE non-UK "ireland" match)
    if "northern ireland" in loc:
        return "UK_OTHER"

    # Non-UK check — BEFORE remote check to fix "Paris, France (Remote)"
    for indicator in NON_UK_INDICATORS:
        if indicator in loc:
            return "OUTSIDE_UK"

    # US state abbreviations: ", XX" at end (2-letter uppercase)
    if re.search(r',\s*[A-Z]{2}\s*$', job_location or ""):
        if re.search(r'\buk\b|\bunited kingdom\b|\bengland\b|\bscotland\b|\bwales\b|\bbritain\b', loc):
            return "UK_OTHER"
        return "OUTSIDE_UK"

    # Remote check (after non-UK, so "Singapore (Remote)" → OUTSIDE_UK)
    if re.search(r'\bremote\b', loc):
        return "REMOTE"

    # "Anywhere" → check JD text for remote signal, default to REMOTE
    if "anywhere" in loc:
        return "REMOTE"

    # UK cities
    for city in UK_CITIES:
        if city in loc:
            return "UK_OTHER"

    # Explicit UK indicators
    if re.search(r'\buk\b|\bunited kingdom\b|\bengland\b|\bscotland\b|\bwales\b|\bbritain\b', loc):
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


def classify_tech(jd_text: str, title: str = "") -> list[str]:
    """Classify tech stack from JD text. Returns array of tech tokens.

    Phase 1B improvements (V12):
    - Added bare \\bnode\\b to NODE detection (verified zero false positives)
    - Added bare \\bjs\\b to JS_TS detection
    - AI_ML boilerplate filter: bare \\bai\\b is ignored when only found in
      hiring process boilerplate ("we use AI in our hiring", "AI-powered screening")

    Phase 3 improvements (V12):
    - Search title + JD body (title has tech keywords not in body for some jobs)
    - Detect concatenated "NodeJavaScript"/"NodeJavascript" patterns
    - Expanded AI_ML boilerplate filter for company descriptions
    """
    # Combine title and JD for tech detection (Phase 3: fixes Job 172 etc.)
    text = ((title or "") + " " + (jd_text or "")).lower()
    tokens = []

    # NODE detection — includes bare "node" (V12: +7 fixes)
    if re.search(
        r'\bnode\.?js\b|\bnode\s+js\b|\bnodejs\b'
        r'|\bexpress\.?js\b|\bexpressjs\b'
        r'|\bnest\.?js\b|\bnestjs\b'
        r'|\bkoa\.?js\b|\bfastify\b'
        r'|\bnode\b'  # bare "node" — nearly always Node.js in SWE job context
        r'|nodejavascri',  # Phase 3: "NodeJavaScript" etc. (concatenated, no word boundary)
        text
    ):
        tokens.append("NODE")

    # REACT detection
    if re.search(r'\breact\b|\breact\.?js\b|\breactjs\b|\bnext\.?js\b|\bnextjs\b|\breact\s*native\b', text):
        tokens.append("REACT")

    # JS_TS detection (V12: refined bare JS matching)
    if re.search(
        r'\btypescript\b|\bjavascript\b'
        r'|\bjs/ts\b|\bts/js\b'
        r'|\bangular\b|\bvue\.?js\b|\bvuejs\b|\bsvelte\b'
        r'|(?<!\.)(?<!\w)js(?:/|\s*,|\s+and\b)'  # "JS/" "JS," "JS and" but NOT ".js"
        r'|nodejavascri',  # Phase 3: "NodeJavaScript" etc. — also triggers JS_TS
        text
    ):
        tokens.append("JS_TS")

    # AI_ML detection — with boilerplate filter (V12: -2 false positives)
    # First check for strong AI_ML signals (always count)
    has_strong_ai = bool(re.search(
        r'\bmachine\s+learning\b|\bdeep\s+learning\b|\bartificial\s+intelligence\b'
        r'|\bnatural\s+language\s+processing\b|\bcomputer\s+vision\b'
        r'|\bneural\s+network\b|\bllm\b|\blarge\s+language\s+model\b'
        r'|\btensorflow\b|\bpytorch\b|\bml\s+engineer\b|\bml\s+platform\b'
        r'|\bfine[- ]?tun(?:e|ing)\b|\bprompt\s+engineering\b|\bnlp\b',
        text
    ))

    if has_strong_ai:
        tokens.append("AI_ML")
    else:
        # Weak signal: bare "ai" — only count if NOT in hiring boilerplate
        ai_matches = list(re.finditer(r'\b(?<!em)ai\b(?!\s*(?:r|m|ds))', text))
        if ai_matches:
            # Check if ALL "ai" mentions are in boilerplate context
            boilerplate_patterns = [
                r'(?:use|using|leverage|embrace)\s+(?:of\s+)?(?:artificial\s+intelligence|ai)\s+(?:in|to|for|tools|powered)',
                r'ai[- ](?:powered|driven|generated)\s+(?:tool|screen|hiring|recruit|assess|version|due\s+diligence|risk)',
                r'(?:note|policy)\s+(?:on|about|regarding)\s+(?:using\s+)?ai',
                r'ai\s+(?:usage|tools?)\s+(?:polic|to\s+support)',
                r'(?:hiring|recruitment|screening|application)\s+(?:process|pipeline).*?ai',
                r'ai.*?(?:hiring|recruitment|screening|application)\s+(?:process|pipeline)',
                # Phase 3: Company/product descriptions (not job requirements)
                r'(?:cutting[- ]edge|global\s+player\s+in)\s+ai',  # "cutting-edge AI", "global player in AI"
                r'ai[- ]driven\s+(?:due|risk|insight|solution|platform|product|intelligence)',  # "AI-driven due diligence"
                r'ai\s+start[- ]?up',  # "AI start up" (company name)
                r'(?:embrace|advantages?\s+of)\s+(?:the\s+)?ai',  # "embrace the advantages of AI"
                r'\bin\s+ai[- ]driven\b',  # "in AI-driven [noun]"
                r'ai\s+(?:talent|agent|partner|recrui)',  # "AI talent partner", "AI agent"
            ]
            all_boilerplate = True
            for m in ai_matches:
                # Get 100-char context around each "ai" match
                ctx_start = max(0, m.start() - 80)
                ctx_end = min(len(text), m.end() + 80)
                context = text[ctx_start:ctx_end]
                is_boilerplate = any(re.search(bp, context) for bp in boilerplate_patterns)
                if not is_boilerplate:
                    all_boilerplate = False
                    break

            if not all_boilerplate:
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
    # Track whether input had explicit currency symbol (£500 is exact, not 500k)
    has_currency = bool(re.search(r'[£$€]', s))
    # Remove currency symbols
    s = re.sub(r'[£$€]', '', s)

    m = re.match(r'^(\d+(?:\.\d+)?)\s*[kK]$', s)
    if m:
        return float(m.group(1)) * 1000

    m = re.match(r'^(\d+(?:\.\d+)?)$', s)
    if m:
        val = float(m.group(1))
        # If it looks like shorthand (e.g., "75" meaning 75k) — but only when
        # there's no explicit £ sign. "£500" is exactly £500 (a budget),
        # while bare "75" in salary context likely means 75k.
        if val < 1000 and not has_currency:
            return val * 1000  # Likely in thousands
        return val

    return None


def classify_comp(jd_text: str, title: str = "") -> str:
    """Classify compensation from JD text.

    Phase 1C improvements (V12):
    - "between £X and £Y" range pattern
    - "£Xk+" pattern (adjacent +)
    - "to £X" / "Salary to £X" → UP_TO_ONLY
    - Per-day rate → NO_GBP filter
    - TC / total compensation disqualifier
    - Salary in job title as fallback
    - Filter non-salary £ amounts (£1.4 trillion, £500 budget, £2.7bn)
    """
    text = jd_text or ""
    text_lower = text.lower()

    # Check for GBP presence (£ sign or "gbp") in JD or title
    has_gbp_jd = bool(re.search(r'£|gbp', text_lower))
    has_gbp_title = bool(re.search(r'£|gbp', (title or "").lower()))

    if not has_gbp_jd and not has_gbp_title:
        return "NO_GBP"

    # ── Disqualifiers (per teacher_v7.txt rules) ─────────────────────────
    # TC / total compensation / total package → NO_GBP (case-insensitive, Phase 3 fix)
    if re.search(r'\btotal\s+comp(?:ensation)?\b|\btotal\s+package\b|\b(?:TC)\b', text, re.IGNORECASE):
        return "NO_GBP"

    # ── Find all salary-like £ patterns, pick the first valid one ────────
    # This implements "find the first salary mentioned in reading order"

    # Collect all candidate salary mentions with their positions
    candidates = []

    # Pattern: "between £X and £Y"
    for m in re.finditer(r'between\s+£\s*([\d,]+(?:\.\d+)?)\s*[kK]?\s+and\s+£?\s*([\d,]+(?:\.\d+)?)\s*[kK]?', text, re.IGNORECASE):
        candidates.append(("range", m.start(), m))

    # Pattern: salary range "£XX - £YY" / "£XXk-£YYk"
    for m in re.finditer(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?\s*[-–—]\s*£?\s*([\d,]+(?:\.\d+)?)\s*[kK]?', text, re.IGNORECASE):
        candidates.append(("range", m.start(), m))

    # Pattern: "£XX to £YY" / "£XXk to £YYk" (Phase 3: fixes Job 172 "£35k to £60k")
    for m in re.finditer(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?\s+to\s+£?\s*([\d,]+(?:\.\d+)?)\s*[kK]?', text, re.IGNORECASE):
        candidates.append(("range", m.start(), m))

    # Pattern: "up to £XXX" / "to £XXX" / "(to £XXX)" (no lower bound)
    for m in re.finditer(r'(?:up\s+to|salary\s+(?:up\s+)?to|(?:salary|paying)[:\s]+to|\(to)\s+£\s*([\d,]+(?:\.\d+)?)\s*[kK]?', text, re.IGNORECASE):
        candidates.append(("up_to", m.start(), m))

    # Pattern: single "£XXk" or "£XX,XXX"
    for m in re.finditer(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?\+?', text):
        candidates.append(("single", m.start(), m))

    # Sort by position (first salary in reading order)
    candidates.sort(key=lambda x: x[1])

    for ctype, pos, m in candidates:
        if ctype == "range":
            low = _parse_salary_value(m.group(1))
            high = _parse_salary_value(m.group(2))
            if low is not None and high is not None:
                if low > high:
                    low, high = high, low
                # Filter non-salary amounts: both values should be plausible salaries
                if not _is_plausible_salary(low) or not _is_plausible_salary(high):
                    continue
                # Check for per-day rate context
                ctx_after = text[m.end():m.end()+30].lower()
                if re.search(r'per\s+day|/day|\bday\s+rate\b|\bp\.?d\.?\b', ctx_after):
                    return "NO_GBP"
                midpoint = (low + high) / 2
                return _midpoint_to_comp(midpoint)

        elif ctype == "up_to":
            val = _parse_salary_value(m.group(1))
            if val is not None and _is_plausible_salary(val):
                # Check for per-day rate context
                ctx_after = text[m.end():m.end()+30].lower()
                if re.search(r'per\s+day|/day|\bday\s+rate\b|\bp\.?d\.?\b', ctx_after):
                    return "NO_GBP"
                return "UP_TO_ONLY"

        elif ctype == "single":
            # Pass full match (with £) so _parse_salary_value can detect explicit currency
            val = _parse_salary_value(m.group(0))
            if val is not None and _is_plausible_salary(val):
                # Check context before for "up to" or "to" phrasing
                ctx_start = max(0, m.start() - 40)
                context_before = text[ctx_start:m.start()].lower()
                if re.search(r'up\s+to\s*$', context_before):
                    return "UP_TO_ONLY"
                if re.search(r'(?:salary|paying|package)[:\s]*to\s*$', context_before):
                    return "UP_TO_ONLY"
                # Phase 3: "WFH to £X" / "(to £X)" — clear "up to" patterns
                if re.search(r'(?:wfh|remote|hybrid|home)\s+to\s*$', context_before):
                    return "UP_TO_ONLY"
                if re.search(r'\(to\s*$', context_before):
                    return "UP_TO_ONLY"
                # Check for per-day rate context
                ctx_after = text[m.end():m.end()+30].lower()
                if re.search(r'per\s+day|/day|\bday\s+rate\b|\bp\.?d\.?\b', ctx_after):
                    return "NO_GBP"
                # Check for "£Xk+" pattern (adjacent +)
                if m.group(0).rstrip().endswith("+"):
                    return _midpoint_to_comp(val)
                return _midpoint_to_comp(val)

    # Fallback: check job title for £ amounts
    if title:
        title_match = re.search(r'£\s*([\d,]+(?:\.\d+)?)\s*[kK]?', title)
        if title_match:
            val = _parse_salary_value(title_match.group(1))
            if val is not None and _is_plausible_salary(val):
                return _midpoint_to_comp(val)

    return "NO_GBP"


def _is_plausible_salary(val: float) -> bool:
    """Filter out non-salary amounts (£1.4 trillion, £500 budget, £2.7bn)."""
    # Plausible annual salary range: £15k - £500k
    return 15000 <= val <= 500000


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
    loc = classify_loc(job.get("job_location", ""), job.get("jd_text", ""))
    arr = classify_arr(job.get("jd_text", ""))
    sen = classify_sen(job.get("title", ""))
    tech = classify_tech(job.get("jd_text", ""), job.get("title", ""))
    comp = classify_comp(job.get("jd_text", ""), job.get("title", ""))

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

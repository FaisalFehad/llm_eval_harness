#!/usr/bin/env python3
"""Cross-validate V16 teacher labels against deterministic prompt rules.
Finds genuine bugs (not model prediction errors) by applying the exact rules from teacher.txt."""
import json, re, sys
from pathlib import Path

UK_CITIES = {"london", "manchester", "birmingham", "leeds", "glasgow", "sheffield", "bradford", "liverpool", "edinburgh", "bristol", "cardiff", "belfast", "leicester", "coventry", "nottingham", "newcastle", "southampton", "portsmouth", "brighton", "cambridge", "oxford", "reading", "aberdeen", "dundee", "bath", "york", "canterbury", "exeter", "frome", "tipton", "derby", "northampton", "swindon", "watford", "st albans", "luton", "guildford", " maidenhead"}
USA_TERMS = {"united states", "usa", "us.", " texas", " california", " florida", " new york", " florida", "illinois", "us remote", "us.", "washington", "boston", "chicago", "seattle", "san francisco", "austin", "atlanta", "denver", " philadelphia"}
REMOTE_TERMS = {"fully remote", "work from home", "remote first", "remote uk"}
US_INDICATORS = {"us remote", "remote us", "remote usa", "$", "usd", "$", "remote first"}

L3_KEYWORDS = {"senior", "lead", "principal", "head of", "director", "vp ", " vp", "architect", "staff ", "manager"}
L1_KEYWORDS = {"associate", "junior", "jr", "graduate", "grad", "entry level", "entry-level", "trainee", "apprentice", "intern", "placement", "i ", " i"}

TECH_KEYWORDS = {
    "node": ["NODE"], "nodejs": ["NODE"], "node.js": ["NODE"],
    "react": ["REACT"], "react.js": ["REACT"],
    "typescript": ["JS_TS"], "type script": ["JS_TS"],
    "js": [],  # too ambiguous
    "pytorch": ["AI_ML"], "torch": ["AI_ML"], "hugging": ["AI_ML"],
    "nlp": ["AI_ML"], "natural language": ["AI_ML"],
    "machine learning": ["AI_ML"], "ml": [],  # standalone ML too broad
    "deep learning": ["AI_ML"],
    "ai product": [],  # NOT AI_ML
}
CONTEXT_DISQ = {
    "node": ["non-technical", "admin", "support", "marketing", "sales", "not technical", "non technical"],
    "react": ["non-technical", "admin", "support", "marketing", "sales", "not technical"],
    "typescript": ["non-technical", "admin", "marketing", "sales", "not technical"],
}

def classify_loc(title, job_location, jd):
    """Posted Location field first; JD description as fallback ONLY if location field is unclear or missing."""
    t = title.lower()
    loc = job_location.lower() if job_location else ""
    jd = jd.lower()

    is_vague = not loc or loc.strip() == "" or loc in {"uk", "united kingdom", "flexible", "anywhere", "remote - united kingdom", "remote uk", "multiple locations"}

    if not is_vague:
        # Location field is specific - trust it.
        loc_words = set(re.findall(r'\b[a-z]+\b', loc))
        if loc_words & UK_CITIES:
            if "london" in loc:
                return "IN_LONDON"
            return "UK_OTHER"
        if "remote" in loc and any(r in loc for r in REMOTE_TERMS):
            return "REMOTE"
        if any(u in loc for u in US_INDICATORS) or loc in {"usa", "us", "united states"}:
            return "OUTSIDE_UK"
        # Specific location we can't resolve - UNK, but do NOT fall back to JD
        return "UNK"

    # Location field is vague/missing - scan Title next
    title_words = set(re.findall(r'\b[a-z]+\b', t))
    # --- FIX: "UK" in title = still vague, keep scanning ---
    if "uk" in title_words or "united" in t or "kingdom" in t:
        # Title says UK (country-level). Still vague! Look for city in title.
        city_in_title = title_words & UK_CITIES
        if city_in_title:
            if "london" in t:
                return "IN_LONDON"
            return "UK_OTHER"
        # No city in title. Move to JD scan for city.
        pass
    elif title_words & UK_CITIES:
        if "london" in t:
            return "IN_LONDON"
        return "UK_OTHER"
    if "remote" in t or "work from home" in t:
        return "REMOTE"
    if any(u in t for u in US_INDICATORS):
        return "OUTSIDE_UK"

    # Title unclear or has "UK", scan Description as last fallback
    jd_words = set(re.findall(r'\b[a-z]+\b', jd))
    if jd_words & UK_CITIES:
        if "london" in jd:
            return "IN_LONDON"
        return "UK_OTHER"
    if any(r in jd for r in REMOTE_TERMS):
        return "REMOTE"
    if any(u in jd for u in US_INDICATORS):
        return "OUTSIDE_UK"
    if "remote" in jd or "hybrid" in jd:
        return "REMOTE"

    return "UNK"

    # Location field is vague/missing - scan Title next
    title_words = set(re.findall(r'\b[a-z]+\b', t))
    if title_words & UK_CITIES:
        if "london" in t:
            return "IN_LONDON"
        return "UK_OTHER"
    if "remote" in t or "work from home" in t:
        return "REMOTE"
    if any(u in t for u in US_INDICATORS):
        return "OUTSIDE_UK"

    # Title unclear, scan Description as last fallback
    jd_words = set(re.findall(r'\b[a-z]+\b', jd))
    if jd_words & UK_CITIES:
        if "london" in jd:
            return "IN_LONDON"
        return "UK_OTHER"
    if any(r in jd for r in REMOTE_TERMS):
        return "REMOTE"
    if any(u in jd for u in US_INDICATORS):
        return "OUTSIDE_UK"
    if "remote" in jd or "hybrid" in jd:
        return "REMOTE"

    return "UNK"

def classify_sen(title, jd=""):
    """Title keywords first; plain titles → JD years-of-experience fallback."""
    t = title.lower()
    if any(k in t for k in L1_KEYWORDS):
        return "LEVEL_1"
    if any(k in t for k in L3_KEYWORDS):
        return "LEVEL_3"
    jd_lower = jd.lower()
    # Check for L1 signals in JD before years
    l1_signals = {"graduate scheme", "graduate programme", "for graduates", "suitable for graduates",
                   "intern", "internship", "placement year", "industrial placement", "student placement",
                   "trainee", "apprentice", "early career", "entry-level", "entry level",
                   "newly qualified", "nq "}
    if any(s in jd_lower for s in l1_signals):
        return "LEVEL_1"
    # Plain title → JD fallback
    jd_lower = jd.lower()
    # Check for L1 signals in JD before years
    l1_signals = {"graduate scheme", "graduate programme", "for graduates", "suitable for graduates",
                   "intern", "internship", "placement year", "industrial placement", "student placement",
                   "trainee", "apprentice", "early career", "entry-level", "entry level",
                   "newly qualified", "nq "}
    if any(s in jd_lower for s in l1_signals):
        return "LEVEL_1"
    # Do NOT use years-of-experience to bump to L3 for plain titles — default to L2
    return "LEVEL_2"

def classify_tech(title, jd):
    """Apply SWE filter + keyword matching + contextual disqualification."""
    text = (title + " " + jd).lower()
    t = title.lower()

    # SWE filter - only classify tech for software engineering roles
    swe_terms = {"engineer", "developer", "software", "devops", "technical lead", "architect", "cto", "vp engineering", "engineering manager"}
    is_swe = any(term in t or term in text[:200] for term in swe_terms)
    if not is_swe:
        return ["OOS"]

    # Check for contextual disqualification (e.g., "Node" in non-technical role)
    tokens = set()
    for keyword, techs in TECH_KEYWORDS.items():
        if keyword in text:
            # Check contextual disqualifiers
            disq_found = False
            for disq in CONTEXT_DISQ.get(keyword, []):
                if disq in text:
                    disq_found = True
                    break
            if not disq_found and techs:
                tokens.update(techs)

    if not tokens:
        return ["OOS"]
    return sorted(tokens)

##############################################################################
#  COMP  – 14 granular salary buckets (midpoint logic for ranges)          #
##############################################################################

COMP_BUCKETS = [
    (0,          30000,  "BELOW_30K"),
    (30000,      40000,  "30_40K"),
    (40000,      50000,  "40_50K"),
    (50000,      60000,  "50_60K"),
    (60000,      70000,  "60_70K"),
    (70000,      80000,  "70_80K"),
    (80000,      90000,  "80_90K"),
    (90000,      100000, "90_100K"),
    (100000,     120000, "100_120K"),
    (120000,     140000, "120_140K"),
    (140000,     160000, "140_160K"),
    (160000,     180000, "160_180K"),
    (180000,     200000, "180_200K"),
    (200000,     float('inf'), "ABOVE_200K"),
]

def midpoint_to_bucket(midpoint):
    for lo, hi, label in COMP_BUCKETS:
        if midpoint >= lo and midpoint < hi:
            return label
    return "ABOVE_200K"

def parse_salary(jd_text, title=""):
    """
    Extract all salary hints from text and compute a *single midpoint*.
    Annual, monthly (×12), and weekly (×52) values are supported.
    Returns (midpoint, count_of_hints, list_of_ranges) or (None, 0, []).
    """
    text = (title + " " + jd_text).replace(",", "")
    results = []

    # ── 1. Annual range e.g. £45k - £80k ──
    for m in re.finditer(r'£\s*(\d+)[Kk]?\s*[-–]\s*£?\s*(\d+)[Kk]?', text):
        lo = int(m.group(1)) * 1000 if int(m.group(1)) < 1000 else int(m.group(1))
        hi = int(m.group(2)) * 1000 if int(m.group(2)) < 1000 else int(m.group(2))
        results.append((lo, hi))

    # ── 2. Monthly rate e.g. £700/month ── (annualise: ×12)
    for m in re.finditer(r'£\s*(\d+(?:,\d{3})*\.?\d*)\s*(?:per\s*/?\s*month|/\s*month|monthly)', text, re.I):
        annual = float(m.group(1).replace(",", "")) * 12
        results.append((annual, annual))

    # ── 3. Weekly rate e.g. £1,200/week ── (annualise: ×52)
    for m in re.finditer(r'£\s*(\d+(?:,\d{3})*\.?\d*)\s*(?:per\s*/?\s*week|/\s*week|weekly)', text, re.I):
        annual = float(m.group(1).replace(",", "")) * 52
        results.append((annual, annual))

    # ── 4. "£45k+" / "£45K and above" ──
    for m in re.finditer(r'£\s*(\d+)[Kk]?\s*(?:\+|and\s+above|plus)', text):
        v = int(m.group(1)) * 1000 if int(m.group(1)) < 1000 else int(m.group(1))
        results.append((v, v * 2))

    # ── 5. "up to £80k" ── midpoint = ceil(£X/2) when no other values found
    if not results:
        for m in re.finditer(r'up\s+to\s+£\s*(\d+)[Kk]?', text, re.I):
            v = int(m.group(1)) * 1000 if int(m.group(1)) < 1000 else int(m.group(1))
            results.append((0, v))

    if not results:
        return None, 0, []

    all_vals = [v for pair in results for v in pair]
    midpoint = (min(all_vals) + max(all_vals)) / 2 if all_vals else 0
    return midpoint, len(results), results

def classify_comp(jd_text, title=""):
    """
    Bucket by midpoint.  Returns (label, midpoint).
    """
    text = (title + " " + jd_text).lower()
    midpoint, n_salaries, raw_ranges = parse_salary(jd_text, title=title)

    if midpoint is None:
        return "NO_GBP", 0

    return midpoint_to_bucket(midpoint), midpoint

FIELDS = ["loc", "sen", "tech", "comp"]

def audit_v16_labels(input_path, output_path):
    with open(input_path) as f:
        jobs = [json.loads(l) for l in f]

    mismatches = {f: [] for f in FIELDS}

    for job in jobs:
        jd = job.get("jd_text", "")
        title = job.get("title", "")
        location = job.get("job_location", "")

        # LOC
        posted_loc = job.get("job_location", "")
        rule_loc = classify_loc(title, posted_loc, jd)
        true_loc = job.get("v16_loc")
        if true_loc and rule_loc != true_loc and true_loc != "UNK":
            mismatches["loc"].append({
                "index": job.get("index"),
                "title": title,
                "location": location,
                "rule": rule_loc,
                "true": true_loc,
                "jd_snippet": jd[:200],
            })

        # SEN
        rule_sen = classify_sen(title, jd=jd)
        true_sen = job.get("v16_sen")
        if true_sen and rule_sen != true_sen:
            mismatches["sen"].append({
                "index": job.get("index"),
                "title": title,
                "rule": rule_sen,
                "true": true_sen,
            })

        # TECH
        rule_tech = classify_tech(title, jd)
        true_tech = job.get("v16_tech", [])
        if true_tech and sorted(rule_tech) != sorted(true_tech if isinstance(true_tech, list) else [true_tech]):
            mismatches["tech"].append({
                "index": job.get("index"),
                "title": title,
                "rule": rule_tech,
                "true": true_tech,
                "jd_snippet": jd[:200],
            })

        # COMP
        rule_comp, _ = classify_comp(jd, title=title)
        true_comp = job.get("v16_comp")
        if true_comp and rule_comp != true_comp:
            mismatches["comp"].append({
                "index": job.get("index"),
                "title": title,
                "rule": rule_comp,
                "true": true_comp,
                "jd_snippet": jd[:200],
            })

    total = sum(len(v) for v in mismatches.values())
    print(f"Found {total} rule-based mismatches:")
    for field in FIELDS:
        if mismatches[field]:
            print(f"  {field}: {len(mismatches[field])}")

    # Cross-reference with XGBoost to find genuine issues
    xgboost_issues = {}
    try:
        import json as j
        with open("versions/v16/data/xgboost_audit_issues.json") as f:
            xgboost_data = j.load(f)
        for issue in xgboost_data["issues"]:
            key = (issue["index"], issue["field"])
            xgboost_issues[key] = issue
    except FileNotFoundError:
        print("XGBoost audit not found. Run xgboost_audit_v16.py first.")

    # Find genuine issues: both XGBoost AND rules disagree with teacher
    genuine_issues = []
    for field in FIELDS:
        for mismatch in mismatches[field]:
            key = (mismatch["index"], field)
            xgboost_issue = xgboost_issues.get(key)
            if xgboost_issue:
                # If XGBoost predicts the SAME as the RULE, it's a genuine bug
                if xgboost_issue["predicted"] == mismatch["rule"]:
                    genuine_issues.append({
                        **mismatch,
                        "field": field,
                        "xgboost_confidence": xgboost_issue["confidence"],
                        "xgboost_gap": xgboost_issue["gap"],
                        "severity": "HIGH" if xgboost_issue["gap"] > 0.8 else "MEDIUM",
                    })

    genuine_issues.sort(key=lambda x: -x["xgboost_gap"])

    report = {
        "total_jobs": len(jobs),
        "rule_mismatches": {k: len(v) for k, v in mismatches.items()},
        "genuine_issues_count": len(genuine_issues),
        "genuine_issues": genuine_issues[:100],  # Top 100
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nGenuine issues (XGBoost + Rules both agree against teacher): {len(genuine_issues)}")
    print(f"Report saved to: {output_path}")

    # Top 10 sample
    print("\nTop 10 genuine issues:")
    for issue in genuine_issues[:10]:
        print(f"  [{issue['field']}] [{issue['severity']}] {issue['title'][:50]}")
        print(f"    Rule: {issue['rule']} | Teacher: {issue['true']} | Conf: {issue['xgboost_confidence']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 rule_audit_v16.py <input.jsonl> [output.json]")
        print("Example: python3 versions/v16/scripts/rule_audit_v16.py versions/v16/data/v16_teacher_labels.jsonl versions/v16/data/rule_audit_genuine.json")
        sys.exit(1)
    audit_v16_labels(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "versions/v16/data/rule_audit_genuine.json")

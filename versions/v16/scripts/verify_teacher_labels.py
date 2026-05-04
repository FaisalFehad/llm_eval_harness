#!/usr/bin/env python3
"""Verify V16 genuine issues against actual V16 teacher.txt rules.

Classifies each issue as:
- by_design     : Difference expected due to V16 rule changes
- teacher_bug   : Teacher label is clearly wrong per V16 rules
- ambiguous     : Edge case / unclear
- rule_bug      : Our rule_audit script misapplied the rules
"""

import json, re
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────
with open("versions/v16/data/v16_teacher_labels.jsonl") as f:
    jobs = {json.loads(l)["index"]: json.loads(l) for l in f}

with open("versions/v16/data/rule_audit_latest.json") as f:
    report = json.load(f)

# ── V16 Rule definitions (from teacher.txt + rule_audit_v16.py) ──────────────

UK_CITIES = {"london","manchester","birmingham","leeds","glasgow","sheffield",
             "bradford","liverpool","edinburgh","bristol","cardiff","belfast",
             "leicester","coventry","nottingham","newcastle","southampton",
             "portsmouth","brighton","cambridge","oxford","reading","aberdeen",
             "dundee","bath","york","canterbury","exeter","frome","tipton",
             "derby","northampton","swindon","watford","st albans","luton",
             "guildford","maidenhead"}

L3_KEYWORDS = {"senior","lead","principal","head of","director","vp "," vp","architect","staff ","manager"}
L1_KEYWORDS = {"associate","junior","jr","graduate","grad","entry level","entry-level","trainee","apprentice","intern","placement","i "," i", "i)","i "}

# Senior I/II/III exception - these are LEVEL_1, not LEVEL_3
# Senior I/II/III exception - these are LEVEL_1, not LEVEL_3
# Check if title ends with " I", " II", " III" or "Senior I", "Senior II", "Senior III"
L1_SENIOR_EXCEPTION = {" i"," ii"," iii"," senior i"," senior ii"," senior iii"}

TECH_KEYWORDS = {
    "node": ["NODE"], "nodejs": ["NODE"], "node.js": ["NODE"],
    "react": ["REACT"], "react.js": ["REACT"],
    "typescript": ["JS_TS"], "type script": ["JS_TS"],
    "pytorch": ["AI_ML"], "torch": ["AI_ML"], "hugging": ["AI_ML"],
    "nlp": ["AI_ML"], "natural language": ["AI_ML"],
    "machine learning": ["AI_ML"], "deep learning": ["AI_ML"],
}
CONTEXT_DISQ = {
    "node": ["non-technical","admin","support","marketing","sales","not technical","non technical"],
    "react": ["non-technical","admin","support","marketing","sales","not technical"],
    "typescript": ["non-technical","admin","marketing","sales","not technical"],
}

COMP_BUCKETS = [
    (0,30000,"BELOW_30K"),(30000,40000,"30_40K"),(40000,50000,"40_50K"),
    (50000,60000,"50_60K"),(60000,70000,"60_70K"),(70000,80000,"70_80K"),
    (80000,90000,"80_90K"),(90000,100000,"90_100K"),(100000,120000,"100_120K"),
    (120000,140000,"120_140K"),(140000,160000,"140_160K"),(160000,180000,"160_180K"),
    (180000,200000,"180_200K"),(200000,float('inf'),"ABOVE_200K"),
]

# ── Rule functions (exact from rule_audit_v16.py / teacher.txt) ──────────────

def classify_loc(title, job_location, jd):
    t = title.lower()
    loc = job_location.lower() if job_location else ""
    jd_lower = jd.lower()

    # V16 teacher rule: Posted Location field first
    is_vague = not loc or loc.strip() == "" or loc in {
        "uk","united kingdom","flexible","anywhere","remote - united kingdom",
        "remote uk","multiple locations"
    }

    if not is_vague:
        loc_words = set(re.findall(r'\b[a-z]+\b', loc))
        if loc_words & UK_CITIES:
            if "london" in loc: return "IN_LONDON"
            return "UK_OTHER"
        if "remote" in loc and any(r in loc for r in {"fully remote","work from home","remote first","remote uk"}):
            return "REMOTE"
        if any(u in loc for u in {"us remote","remote us","remote usa","$","usd"}) or loc in {"usa","us","united states"}:
            return "OUTSIDE_UK"
        return "UNK"  # specific location we can't resolve

    # V16 teacher: if vague, scan Title next
    title_words = set(re.findall(r'\b[a-z]+\b', t))
    if "uk" in title_words or "united" in t or "kingdom" in t:
        city_in_title = title_words & UK_CITIES
        if city_in_title:
            if "london" in t: return "IN_LONDON"
            return "UK_OTHER"
        # No city in title → scan JD for city
        pass
    elif title_words & UK_CITIES:
        if "london" in t: return "IN_LONDON"
        return "UK_OTHER"
    if "remote" in t or "work from home" in t:
        return "REMOTE"
    if any(u in t for u in {"us remote","remote us","remote usa","$","usd"}):
        return "OUTSIDE_UK"

    # Title unclear or has "UK", scan Description as last fallback
    jd_words = set(re.findall(r'\b[a-z]+\b', jd_lower))
    if jd_words & UK_CITIES:
        if "london" in jd_lower: return "IN_LONDON"
        return "UK_OTHER"
    if any(r in jd_lower for r in {"fully remote","work from home","remote first","remote uk"}):
        return "REMOTE"
    if any(u in jd_lower for u in {"us remote","remote us","remote usa","$","usd","united states"}):
        return "OUTSIDE_UK"
    if "remote" in jd_lower or "hybrid" in jd_lower:
        return "REMOTE"
    return "UNK"


def classify_sen(title, jd=""):
    t = title.lower()
    if any(k in t for k in L1_KEYWORDS):
        return "LEVEL_1"
    if any(k in t for k in L3_KEYWORDS):
        return "LEVEL_3"
    jd_lower = jd.lower()
    l1_signals = {"graduate scheme","graduate programme","for graduates","suitable for graduates",
                  "intern","internship","placement year","industrial placement","student placement",
                  "trainee","apprentice","early career","entry-level","entry level","newly qualified","nq "}
    if any(s in jd_lower for s in l1_signals):
        return "LEVEL_1"
    # V16 rule: plain title → default LEVEL_2 (do NOT use years for L3)
    return "LEVEL_2"


def classify_tech(title, jd):
    text = (title + " " + jd).lower()
    t = title.lower()
    swe_terms = {"engineer","developer","software","devops","technical lead","architect","cto","vp engineering","engineering manager"}
    is_swe = any(term in t or term in text[:200] for term in swe_terms)
    if not is_swe:
        return ["OOS"]
    tokens = set()
    for keyword, techs in TECH_KEYWORDS.items():
        if keyword in text:
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


def parse_salary(jd_text, title=""):
    text = (title + " " + jd_text).replace(",", "")
    results = []
    # Annual range
    for m in re.finditer(r'£\s*(\d+)[Kk]?\s*[-–]\s*£?\s*(\d+)[Kk]?', text):
        lo = int(m.group(1)) * 1000 if int(m.group(1)) < 1000 else int(m.group(1))
        hi = int(m.group(2)) * 1000 if int(m.group(2)) < 1000 else int(m.group(2))
        results.append((lo, hi))
    # Monthly
    for m in re.finditer(r'£\s*(\d+(?:,\d{3})*\.?\d*)\s*(?:per\s*/?\s*month|/\s*month|monthly)', text, re.I):
        annual = float(m.group(1).replace(",","")) * 12
        results.append((annual, annual))
    # Weekly
    for m in re.finditer(r'£\s*(\d+(?:,\d{3})*\.?\d*)\s*(?:per\s*/?\s*week|/\s*week|weekly)', text, re.I):
        annual = float(m.group(1).replace(",","")) * 52
        results.append((annual, annual))
    # £X+
    for m in re.finditer(r'£\s*(\d+)[Kk]?\s*(?:\+|and\s+above|plus)', text):
        v = int(m.group(1)) * 1000 if int(m.group(1)) < 1000 else int(m.group(1))
        results.append((v, v * 2))
    # Up to
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
    text = (title + " " + jd_text).lower()
    # V16: Non-£ currency → NO_GBP
    if any(c in text for c in {"$","€","usd","eur"}):
        return "NO_GBP", 0
    # Disqualifiers
    if any(d in text for d in {"ote","commission","daily rate","hourly rate","per day","per hour"}):
        return "NO_GBP", 0
    midpoint, n_salaries, raw_ranges = parse_salary(jd_text, title=title)
    if midpoint is None:
        # Check for soft disqualifiers
        if any(s in text for s in {"competitive salary","negotiable","doe","d.o.e.","depending on experience","competitive"}):
            return "NO_GBP", 0
        return "NO_GBP", 0
    for lo, hi, label in COMP_BUCKETS:
        if midpoint >= lo and midpoint < hi:
            return label, midpoint
    return "ABOVE_200K", midpoint


# ── Classification engine ───────────────────────────────────────────────────

def classify_job(job):
    title = job.get("title", "")
    loc = job.get("job_location", "")
    jd = job.get("jd_text", "")
    return {
        "loc": classify_loc(title, loc, jd),
        "sen": classify_sen(title, jd),
        "tech": classify_tech(title, jd),
        "comp": classify_comp(jd, title=title)[0],
    }


# ── Analyzer ─────────────────────────────────────────────────────────────────

V16_DESIGN_CHANGES = {
    "loc": {
        "desc": "V16: Simplified loc — removed REMOTE/UK_OTHER distinction from JD fallback; only city names or 'remote' in title/JD count. 'Remote - United Kingdom' in location field is treated as vague (not REMOTE).",
        "expected_differences": ["REMOTE→UK_OTHER", "UK_OTHER→REMOTE", "IN_LONDON→UK_OTHER", "UK_OTHER→IN_LONDON"],
    },
    "sen": {
        "desc": "V16: No years-of-experience fallback for L3 on plain titles. Support/IT titles without L1 keywords default to L2 (not L1).",
        "expected_differences": ["LEVEL_1→LEVEL_2", "LEVEL_3→LEVEL_2"],
    },
    "tech": {
        "desc": "V16: Tech only for SWE roles. Strict SWE filter + contextual disqualification. AI_ML requires specific framework/task context. No NODE from Next.js/NestJS.",
        "expected_differences": ["OOS→AI_ML", "AI_ML→OOS", "JS_TS→OOS", "NODE→OOS"],
    },
    "comp": {
        "desc": "V16: 14 granular buckets instead of 7. Midpoint logic for ranges and 'up to'. Monthly/weekly rates annualized. NO_GBP for non-£/OTE/daily/hourly.",
        "expected_differences": ["RANGE_45_54K→40_50K", "RANGE_55_74K→50_60K/60_70K", "RANGE_75_99K→70_80K/80_90K", "ABOVE_100K→100_120K/120_140K/etc", "UP_TO_ONLY→NO_GBP"],
    },
}

def analyze_issue(issue, job, rule_out):
    field = issue["field"]
    teacher_val = issue["true"]
    rule_val = issue["rule"]
    title = job.get("title", "")
    jd = job.get("jd_text", "")[:500]
    loc = job.get("job_location", "")

    # Check if rule and XGBoost both agree
    xgboost_pred = issue.get("xgboost_confidence")

    result = {
        "index": issue["index"],
        "title": title,
        "field": field,
        "teacher": teacher_val,
        "rule": rule_val,
        "xgboost_conf": xgboost_pred,
        "xgboost_gap": issue.get("xgboost_gap"),
        "severity": issue.get("severity"),
        "classification": "unknown",
        "reason": "",
        "job_location": loc,
        "jd_snippet": jd[:300],
    }

    # ── Field-specific analysis ─────────────────────────────────────────────

    if field == "loc":
        # Check if location field has "Remote - United Kingdom" → V16 treats as vague
        loc_field = (loc or "").lower()
        if "remote - united kingdom" in loc_field or loc_field in {"uk", "united kingdom", "remote uk"}:
            if rule_val == "UK_OTHER" and teacher_val == "REMOTE":
                result["classification"] = "by_design"
                result["reason"] = "V16 treats 'Remote - UK' in location field as vague, falls back to JD. If JD has no explicit 'remote' text, defaults to UNK/UK_OTHER."
                return result
        # City mismatch
        city_in_loc = any(city in loc.lower() for city in UK_CITIES)
        if city_in_loc:
            if rule_val != teacher_val:
                result["classification"] = "teacher_bug"
                result["reason"] = f"Location field explicitly contains a city. Teacher should follow posted location field."
                return result
        # Title has city
        title_words = set(re.findall(r'\b[a-z]+\b', title.lower()))
        if title_words & UK_CITIES:
            if rule_val != teacher_val:
                result["classification"] = "by_design"
                result["reason"] = "V16 uses title-first fallback when location field is vague."
                return result
        # General
        result["classification"] = "by_design"
        result["reason"] = "V16 loc rules changed — likely by-design difference."
        return result

    elif field == "sen":
        t = title.lower()
        
        # Senior I/II/III is LEVEL_1, not LEVEL_3 - CHECK FIRST
        has_senior_i = any(k in t for k in L1_SENIOR_EXCEPTION)
        if has_senior_i:
            # Senior I/II/III is LEVEL_1
            if teacher_val != "LEVEL_1":
                result["classification"] = "teacher_bug"
                result["reason"] = "Title has 'Senior I/II/III' pattern. Should be LEVEL_1."
                return result
            # Senior I/II/III is correctly LEVEL_1
            result["classification"] = "correct"
            result["reason"] = "Senior X I/II/III correctly detected as LEVEL_1."
            return result
        
        has_l3 = any(k in t for k in L3_KEYWORDS)
        has_l1 = any(k in t for k in L1_KEYWORDS)

        if has_l3 and teacher_val != "LEVEL_3":
            result["classification"] = "teacher_bug"
            result["reason"] = f"Title has L3 keyword. Teacher should be LEVEL_3."
            return result
        if has_l1 and teacher_val != "LEVEL_1":
            result["classification"] = "teacher_bug"
            result["reason"] = f"Title has L1 keyword. Teacher should be LEVEL_1."
            return result

        # Support roles without L1 keywords default L2 in V16
        support_indicators = {"support","helpdesk","service desk","1st line","2nd line","it support"," technician"}
        is_support = any(s in t for s in support_indicators)
        if is_support and not has_l1 and teacher_val == "LEVEL_1":
            result["classification"] = "by_design"
            result["reason"] = "V16: Support/helpdesk titles without L1 keywords default to LEVEL_2 (not LEVEL_1)."
            return result

        # Senior I/II/III is LEVEL_1, not LEVEL_3
        has_senior_i = any(k in t for k in L1_SENIOR_EXCEPTION)
        if has_senior_i:
            # Senior I/II/III is LEVEL_1
            if teacher_val != "LEVEL_1":
                result["classification"] = "teacher_bug"
                result["reason"] = "Title has 'Senior I/II/III' pattern. Should be LEVEL_1."
                return result
            # Senior I/II/III is correctly LEVEL_1
            result["classification"] = "correct"
            result["reason"] = "Senior X I/II/III correctly detected as LEVEL_1."
            return result

        # Plain title with JD L1 signals
        jd_lower = jd.lower()
        l1_signals = {"graduate scheme","intern","internship","placement year","industrial placement","trainee","apprentice","early career","entry-level","entry level"}
        has_l1_jd = any(s in jd_lower for s in l1_signals)
        if has_l1_jd and teacher_val != "LEVEL_1":
            result["classification"] = "teacher_bug"
            result["reason"] = "JD contains explicit L1 signals. Teacher should be LEVEL_1."
            return result

        result["classification"] = "by_design"
        result["reason"] = "V16 sen rules changed — no years-of-experience L3 fallback on plain titles."
        return result

    elif field == "tech":
        text = (title + " " + jd).lower()
        swe_terms = {"engineer","developer","software","devops","technical lead","architect","cto","vp engineering","engineering manager"}
        is_swe = any(term in title.lower() or term in text[:200] for term in swe_terms)

        if not is_swe and teacher_val not in (["OOS"], "OOS"):
            result["classification"] = "teacher_bug"
            result["reason"] = "Role is not SWE per V16 filter. Teacher should assign OOS."
            return result

        # Check for specific tech mentions
        has_node = "node" in text or "nodejs" in text or "node.js" in text
        has_react = "react" in text
        has_ts = "typescript" in text or "type script" in text
        has_ai = "machine learning" in text or "pytorch" in text or "nlp" in text or "llm" in text

        rule_tech_set = set(rule_val if isinstance(rule_val, list) else [rule_val])
        teacher_tech_set = set(teacher_val if isinstance(teacher_val, list) else [teacher_val])

        if teacher_tech_set == {"OOS"} and rule_tech_set != {"OOS"}:
            result["classification"] = "teacher_bug"
            result["reason"] = f"Teacher marked OOS but text contains tracked tech: node={has_node}, react={has_react}, ts={has_ts}, ai={has_ai}"
            return result

        if rule_tech_set == {"OOS"} and teacher_tech_set != {"OOS"}:
            result["classification"] = "by_design"
            result["reason"] = "V16 SWE filter or contextual disqualification removed tech tokens."
            return result

        result["classification"] = "by_design"
        result["reason"] = "V16 tech rules changed — stricter SWE filter + contextual disqualification."
        return result

    elif field == "comp":
        # Check for disqualifiers
        text = (title + " " + jd).lower()
        if any(d in text for d in {"ote","commission","daily rate","hourly rate","per day","per hour"}):
            if teacher_val != "NO_GBP":
                result["classification"] = "teacher_bug"
                result["reason"] = "JD contains disqualifier (OTE/commission/daily/hourly). Teacher should be NO_GBP."
                return result

        # Check for non-£
        if any(c in text for c in {"$","€","usd","eur"}):
            if teacher_val != "NO_GBP":
                result["classification"] = "teacher_bug"
                result["reason"] = "Non-£ currency found. Teacher should be NO_GBP."
                return result

        # Check for up-to
        has_up_to = bool(re.search(r'up\s+to\s+£', text))
        if has_up_to:
            if teacher_val == "UP_TO_ONLY":
                result["classification"] = "teacher_bug"
                result["reason"] = "V16: 'Up to £X' → midpoint = £X/2, not UP_TO_ONLY."
                return result

        result["classification"] = "by_design"
        result["reason"] = "V16 comp rules changed — 14 granular buckets + midpoint logic."
        return result

    result["classification"] = "ambiguous"
    result["reason"] = "Could not determine root cause automatically."
    return result


# ── Run analysis ─────────────────────────────────────────────────────────────

results = []
for issue in report["genuine_issues"]:
    idx = issue["index"]
    job = jobs.get(idx)
    if not job:
        continue
    rule_out = classify_job(job)
    field = issue["field"]
    issue["rule"] = rule_out[field]  # ensure we use our fresh rule output
    analysis = analyze_issue(issue, job, rule_out)
    results.append(analysis)

# ── Summarize ────────────────────────────────────────────────────────────────

by_class = {}
by_field = {}
for r in results:
    by_class.setdefault(r["classification"], []).append(r)
    by_field.setdefault(r["field"], []).append(r)

print("=" * 80)
print("V16 GENUINE ISSUES VERIFICATION REPORT")
print("=" * 80)
print(f"\nTotal issues analyzed: {len(results)}")
print(f"\nBy classification:")
for cls, items in sorted(by_class.items(), key=lambda x: -len(x[1])):
    print(f"  {cls:20s}: {len(items):3d} ({len(items)/len(results)*100:.1f}%)")

print(f"\nBy field:")
for field, items in sorted(by_field.items()):
    print(f"  {field:5s}: {len(items):3d}")
    sub = {}
    for it in items:
        sub.setdefault(it["classification"], 0)
        sub[it["classification"]] += 1
    for cls, cnt in sorted(sub.items(), key=lambda x: -x[1]):
        print(f"    {cls:20s}: {cnt}")

# Teacher bugs detail
print("\n" + "=" * 80)
print("TEACHER BUGS (teacher label wrong per V16 rules)")
print("=" * 80)
for r in results:
    if r["classification"] == "teacher_bug":
        print(f"\n[{r['severity']}] idx={r['index']} | {r['title'][:60]}")
        print(f"  Field: {r['field']} | Teacher: {r['teacher']} | Rule: {r['rule']}")
        print(f"  Reason: {r['reason']}")
        print(f"  XGBoost conf={r['xgboost_conf']}, gap={r['xgboost_gap']}")
        print(f"  Location: {r['job_location']}")
        print(f"  JD: {r['jd_snippet'][:200]}")

# By-design detail
print("\n" + "=" * 80)
print("BY-DESIGN DIFFERENCES (V16 rule changes) — Top 15")
print("=" * 80)
count = 0
for r in results:
    if r["classification"] == "by_design":
        count += 1
        if count > 15:
            break
        print(f"\n[{r['severity']}] idx={r['index']} | {r['title'][:60]}")
        print(f"  Field: {r['field']} | Teacher: {r['teacher']} | Rule: {r['rule']}")
        print(f"  Reason: {r['reason']}")
        print(f"  XGBoost conf={r['xgboost_conf']}, gap={r['xgboost_gap']}")

# Save full report
output = {
    "total": len(results),
    "by_classification": {k: len(v) for k, v in by_class.items()},
    "by_field": {k: [r for r in v] for k, v in by_field.items()},  # too big, just keep summary
    "issues": results,
}

# Simplify for JSON
output["by_field_summary"] = {
    k: {cls: len([r for r in v if r["classification"] == cls]) for cls in set(r["classification"] for r in v)}
    for k, v in by_field.items()
}

with open("versions/v16/data/teacher_verification_report.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n\nFull report saved to: versions/v16/data/teacher_verification_report.json")

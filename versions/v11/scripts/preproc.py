import re
from typing import Dict, List

DROP_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        "equal opportunity",
        "diversity",
        "inclusion",
        "privacy policy",
        "cookies?",
        "terms and conditions",
        "how to apply",
        "apply now",
        "about the company",
        "benefits include",
        "perks",
        "vision and values",
    ]
]
FACT_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"£|€|\$|salary|compensation|per annum|p\.?.a\.?.",
        r"\bremote\b|\bhybrid\b|\bon[- ]?site\b|\bin[- ]?office\b",
        r"\blondon\b|\buk\b|\bunited kingdom\b|\bbased in\b",
        r"\bnode(\.js|js)?\b|\breact(\.js|js)?\b|\btypescript\b|\bjavascript\b|\bai\b|\bmachine learning\b|\bml\b",
        r"\bjunior\b|\bgraduate\b|\bentry[- ]?level\b|\bsenior\b|\blead\b|\bstaff\b|\bprincipal\b|\bmanager\b|\bintern(ship)?\b",
        r"\baws\b|\bpython\b|\bjava\b|\bdotnet\b|\bc\+\+\b|\bphp\b|\bruby\b|\bgo(lang)?\b|\bsnowflake\b|\bsql\b",
    ]
]
NON_TARGET_TECH = re.compile(r"\bpython\b|\bjava\b|\bdotnet\b|\bc#\b|\bc\+\+\b|\bphp\b|\bruby\b|\bgo(lang)?\b|\bscala\b|\bkotlin\b|\bswift\b|\bvue\b|\bangular\b|\bterraform\b|\bdocker\b|\bkubernetes\b|\bsnowflake\b|\bsql\b", re.I)
CURRENCY_GBP = re.compile(r"£|\bgbp\b", re.I)
CURRENCY_NON_GBP = re.compile(r"\busd\b|\beur\b|\$|€", re.I)
AI_TERMS = re.compile(r"\bai\b|\bartificial intelligence\b|\bmachine learning\b|\bml\b|\bllm\b", re.I)
NON_UK = re.compile(r"united states|usa|us\b|india|germany|australia|canada|singapore|dubai|uae|remote \(us\)|remote \(usa\)", re.I)
UK_TERMS = re.compile(r"\buk\b|united kingdom|england|scotland|wales|northern ireland|london|manchester|bristol|edinburgh", re.I)


def split_sentences(text: str) -> List[str]:
    normalized = text.replace("\r", "\n").replace("\n+", "\n").strip()
    if not normalized:
        return []
    parts: List[str] = []
    for line in normalized.split("\n"):
        for seg in re.split(r"(?<=[.!?])\s+", line):
            seg = re.sub(r"\s+", " ", seg).strip()
            if seg:
                parts.append(seg)
    seen = set()
    out: List[str] = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def keep_sentence(s: str) -> bool:
    if any(p.search(s) for p in DROP_PATTERNS):
        return False
    return any(p.search(s) for p in FACT_PATTERNS)


def clean_text(jd: str) -> str:
    sentences = split_sentences(jd)
    kept = [s for s in sentences if keep_sentence(s)]
    clipped = kept[:28]
    seen = set()
    uniq: List[str] = []
    for s in clipped:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return "\n".join(uniq)


def build_hint(text: str) -> str:
    loc = "UNK"
    arr = "UNK"
    comp = "UNK"
    tech: List[str] = []
    if re.search(r"\bremote\b", text, re.I):
        arr = "REMOTE"
    elif re.search(r"\bhybrid\b", text, re.I):
        arr = "HYBRID"
    elif re.search(r"on[- ]?site|in[- ]?office", text, re.I):
        arr = "IN_OFFICE"

    if re.search(r"london", text, re.I):
        loc = "IN_LONDON"
    elif re.search(r"uk|united kingdom|england|scotland|wales", text, re.I):
        loc = "UK_OTHER"
    elif re.search(r"remote", text, re.I):
        loc = "REMOTE"
    elif NON_UK.search(text):
        loc = "OUTSIDE_UK"

    if CURRENCY_NON_GBP.search(text) and not CURRENCY_GBP.search(text):
        comp = "NO_GBP"
    elif re.search(r"100k|100,000|\b1\d{5}\b", text, re.I) or re.search(r"£\s?(1\d{2})k", text, re.I):
        comp = "ABOVE_100K"

    if re.search(r"node(\.js|js)?", text, re.I):
        tech.append("NODE")
    if re.search(r"react", text, re.I):
        tech.append("REACT")
    if re.search(r"typescript|javascript", text, re.I):
        tech.append("JS_TS")
    if AI_TERMS.search(text):
        tech.append("AI_ML")
    if not tech:
        tech.append("OOS")

    tech_sorted = sorted({"AI_ML" if t == "AI" else t for t in tech})
    return f"loc={loc}; arr={arr}; comp={comp}; tech={','.join(tech_sorted)}"


def apply_postprocess(tokens: Dict, text: str) -> Dict:
    out = dict(tokens)
    tech = out.get("tech", []) or []
    if isinstance(tech, str):
        tech = [tech]

    if CURRENCY_NON_GBP.search(text) and not CURRENCY_GBP.search(text):
        out["comp"] = "NO_GBP"
    if re.search(r"/day|per day|day rate", text, re.I):
        out["comp"] = "UP_TO_ONLY"
    if re.search(r"£\s?(1\d{5}|\d{3}k)|\b1[01]\d,?\d{3}\b|£\s?\d{3},\d{3}", text, re.I):
        out["comp"] = "ABOVE_100K"

    if NON_UK.search(text) and not UK_TERMS.search(text) and "REMOTE" not in text.upper():
        out["loc"] = "OUTSIDE_UK"

    if AI_TERMS.search(text) and "AI_ML" not in tech:
        tech.append("AI_ML")

    target = {"NODE", "REACT", "JS_TS", "AI_ML"}
    has_target = any(t in target for t in tech)
    if not has_target:
        tech = ["OOS"]
    tech = sorted({t for t in tech})
    if not tech:
        tech = ["OOS"]
    out["tech"] = tech
    return out

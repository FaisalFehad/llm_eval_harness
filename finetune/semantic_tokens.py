"""
Semantic Token Vocabulary and Score Conversion for V5 Student Model.

The student model outputs semantic labels (e.g., "LONDON_OR_REMOTE")
instead of numeric scores. This module converts those labels to scores,
computes the total, and assigns a final label deterministically.

Used by eval_student.py and any Python-based pipeline scripts.
"""

# ── Token Vocabularies ───────────────────────────────────────────────────────

LOC_TOKENS = ("LONDON_OR_REMOTE", "UK_OTHER", "OUTSIDE_UK", "MISSING")
ROLE_TOKENS = ("SENIOR_PLUS", "MID_LEVEL", "NO_SENIORITY")
TECH_TOKENS = (
    "NONE", "JS_TS", "NODE", "NODE_JS_TS",
    "AI_ML", "JS_TS_AI_ML", "NODE_AI_ML", "NODE_JS_TS_AI_ML",
)
COMP_TOKENS = (
    "NO_GBP", "UP_TO_ONLY", "BELOW_45K",
    "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K",
)

# ── Score Maps ───────────────────────────────────────────────────────────────

LOC_MAP = {
    "LONDON_OR_REMOTE": 25,
    "UK_OTHER": 10,
    "OUTSIDE_UK": -50,
    "MISSING": 0,
}

ROLE_MAP = {
    "SENIOR_PLUS": 25,
    "MID_LEVEL": 15,
    "NO_SENIORITY": 0,
}

TECH_MAP = {
    "NONE": 0,
    "JS_TS": 5,
    "NODE": 10,
    "NODE_JS_TS": 15,
    "AI_ML": 10,
    "JS_TS_AI_ML": 15,
    "NODE_AI_ML": 20,
    "NODE_JS_TS_AI_ML": 25,
}

COMP_MAP = {
    "NO_GBP": 0,
    "UP_TO_ONLY": 0,
    "BELOW_45K": -30,
    "RANGE_55_74K": 5,
    "RANGE_75_99K": 15,
    "ABOVE_100K": 25,
}

# ── All valid tokens per field ───────────────────────────────────────────────

FIELD_TOKENS = {
    "loc": LOC_TOKENS,
    "role": ROLE_TOKENS,
    "tech": TECH_TOKENS,
    "comp": COMP_TOKENS,
}

FIELD_MAPS = {
    "loc": LOC_MAP,
    "role": ROLE_MAP,
    "tech": TECH_MAP,
    "comp": COMP_MAP,
}


# ── Fuzzy Matching ───────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def fuzzy_match(value: str, valid_tokens: tuple[str, ...]) -> str | None:
    """
    Fuzzy-match a string against valid tokens.
    Returns the match if edit distance ≤ 2, otherwise None.
    """
    if value in valid_tokens:
        return value
    upper = value.upper()
    if upper in valid_tokens:
        return upper
    best_match = None
    best_dist = float("inf")
    for token in valid_tokens:
        dist = edit_distance(upper, token)
        if dist < best_dist:
            best_dist = dist
            best_match = token
    return best_match if best_dist <= 2 else None


# ── Validation ───────────────────────────────────────────────────────────────

def validate_prediction(parsed: dict) -> dict:
    """
    Validate a parsed JSON object as a semantic prediction.
    Returns {"valid": bool, "errors": [...], "corrected": {...}, "fuzzy_corrections": [...]}.
    """
    errors = []
    fuzzy_corrections = []

    for field in ("reasoning", "loc", "role", "tech", "comp"):
        if field not in parsed:
            errors.append(f"Missing field: {field}")

    if errors:
        return {"valid": False, "errors": errors, "fuzzy_corrections": []}

    corrected = {"reasoning": str(parsed.get("reasoning", ""))}

    for field_name, tokens in FIELD_TOKENS.items():
        value = str(parsed.get(field_name, ""))
        matched = fuzzy_match(value, tokens)
        if matched:
            corrected[field_name] = matched
            if matched != value:
                fuzzy_corrections.append(f'{field_name}: "{value}" → "{matched}"')
        else:
            errors.append(f'Invalid {field_name} token: "{value}"')

    if errors:
        return {"valid": False, "errors": errors, "fuzzy_corrections": fuzzy_corrections}

    return {
        "valid": True,
        "errors": [],
        "corrected": corrected,
        "fuzzy_corrections": fuzzy_corrections,
    }


# ── Score Computation ────────────────────────────────────────────────────────

def compute_from_tokens(pred: dict) -> dict:
    """
    Compute numeric score and label from semantic tokens.
    This is the ONLY place where tokens become numbers.
    """
    loc_score = LOC_MAP[pred["loc"]]
    role_score = ROLE_MAP[pred["role"]]
    tech_score = TECH_MAP[pred["tech"]]
    comp_score = COMP_MAP[pred["comp"]]

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


# ── Numeric → Token Conversion ───────────────────────────────────────────────

def numeric_to_tokens(loc: int, role: int, tech: int, comp: int) -> dict:
    """Convert numeric scores (old format) to semantic tokens (new format)."""
    rev_loc = {v: k for k, v in LOC_MAP.items()}
    rev_role = {v: k for k, v in ROLE_MAP.items()}
    rev_tech = {v: k for k, v in TECH_MAP.items()}
    rev_comp = {v: k for k, v in COMP_MAP.items()}

    return {
        "loc": rev_loc.get(loc, "MISSING"),
        "role": rev_role.get(role, "NO_SENIORITY"),
        "tech": rev_tech.get(tech, "NONE"),
        "comp": rev_comp.get(comp, "NO_GBP"),
    }

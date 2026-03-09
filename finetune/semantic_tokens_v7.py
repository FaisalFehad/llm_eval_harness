"""
Semantic Token Vocabulary and Score Conversion for V7 Student Model.

V7 splits V6's 4 fields (loc, role, tech, comp) into 6 fields:
  - location     (was loc)  — where the job is based
  - work_arrangement (NEW)  — remote/hybrid/in-office (informational, no score)
  - scope        (NEW)      — engineering or not (gate: OUT_OF_SCOPE -> seniority=0)
  - seniority    (was role) — experience level
  - tech         (unchanged)
  - comp         (unchanged)

Scores are backward-compatible with V6:
  loc_score  = LOCATION_MAP[location]
  role_score = SENIORITY_MAP[seniority] if scope == IN_SCOPE else 0
  tech_score = TECH_MAP[tech]
  comp_score = COMP_MAP[comp]

  score = clamp(0, 100, loc_score + role_score + tech_score + comp_score)

Used by eval_student_v7.py and any Python-based V7 pipeline scripts.
"""

# ── Token Vocabularies ───────────────────────────────────────────────────────

LOCATION_TOKENS = ("IN_LONDON", "FULLY_REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNKNOWN")
WORK_ARRANGEMENT_TOKENS = ("REMOTE", "HYBRID", "IN_OFFICE", "UNKNOWN")
SCOPE_TOKENS = ("IN_SCOPE", "OUT_OF_SCOPE")
SENIORITY_TOKENS = ("LEVEL_3", "LEVEL_2", "LEVEL_1")
TECH_TOKENS = (
    "NONE", "JS_TS", "NODE", "NODE_JS_TS",
    "AI_ML", "JS_TS_AI_ML", "NODE_AI_ML", "NODE_JS_TS_AI_ML",
)
COMP_TOKENS = (
    "NO_GBP", "UP_TO_ONLY", "BELOW_45K",
    "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K",
)

# ── Score Maps ───────────────────────────────────────────────────────────────
# Only fields that contribute to the numeric score have maps.
# work_arrangement is informational (no score).
# scope is a gate (OUT_OF_SCOPE -> seniority forced to 0).

LOCATION_MAP = {
    "IN_LONDON": 25,
    "FULLY_REMOTE": 25,
    "UK_OTHER": 10,
    "OUTSIDE_UK": -50,
    "UNKNOWN": 0,
}

SENIORITY_MAP = {
    "LEVEL_3": 25,
    "LEVEL_2": 15,
    "LEVEL_1": 0,
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
    "location": LOCATION_TOKENS,
    "work_arrangement": WORK_ARRANGEMENT_TOKENS,
    "scope": SCOPE_TOKENS,
    "seniority": SENIORITY_TOKENS,
    "tech": TECH_TOKENS,
    "comp": COMP_TOKENS,
}

# Score maps for fields that have numeric scores (excludes work_arrangement, scope)
FIELD_SCORE_MAPS = {
    "location": LOCATION_MAP,
    "seniority": SENIORITY_MAP,
    "tech": TECH_MAP,
    "comp": COMP_MAP,
}

# V7 field -> V6-compatible score name (for backward-compatible output)
V7_TO_SCORE_NAME = {
    "location": "loc_score",
    "seniority": "role_score",
    "tech": "tech_score",
    "comp": "comp_score",
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
    Returns the match if edit distance <= 2, otherwise None.
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

# The 12 expected fields in V7 output (6 reason + 6 token), in order
V7_EXPECTED_FIELDS = (
    "location_reason", "location",
    "work_arrangement_reason", "work_arrangement",
    "scope_reason", "scope",
    "seniority_reason", "seniority",
    "tech_reason", "tech",
    "comp_reason", "comp",
)

# The 6 reason fields
V7_REASON_FIELDS = (
    "location_reason", "work_arrangement_reason",
    "scope_reason", "seniority_reason",
    "tech_reason", "comp_reason",
)

# The 6 token fields (keys in FIELD_TOKENS)
V7_TOKEN_FIELDS = ("location", "work_arrangement", "scope", "seniority", "tech", "comp")


def validate_prediction(parsed: dict) -> dict:
    """
    Validate a parsed JSON object as a V7 semantic prediction.
    Returns {"valid": bool, "errors": [...], "corrected": {...}, "fuzzy_corrections": [...]}.
    """
    errors = []
    fuzzy_corrections = []

    # Check all 12 fields are present
    for field in V7_EXPECTED_FIELDS:
        if field not in parsed:
            errors.append(f"Missing field: {field}")

    if errors:
        return {"valid": False, "errors": errors, "fuzzy_corrections": []}

    # Copy reason fields as-is
    corrected = {f: str(parsed.get(f, "")) for f in V7_REASON_FIELDS}

    # Validate and fuzzy-match each token field
    for field_name in V7_TOKEN_FIELDS:
        tokens = FIELD_TOKENS[field_name]
        value = str(parsed.get(field_name, ""))
        matched = fuzzy_match(value, tokens)
        if matched:
            corrected[field_name] = matched
            if matched != value:
                fuzzy_corrections.append(f'{field_name}: "{value}" -> "{matched}"')
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
    Compute numeric score and label from V7 semantic tokens.
    This is the ONLY place where tokens become numbers.

    Scope gate: if scope == OUT_OF_SCOPE, role_score is forced to 0.

    Output keys are backward-compatible with V6:
      loc_score, role_score, tech_score, comp_score, score, label
    """
    loc_score = LOCATION_MAP[pred["location"]]
    tech_score = TECH_MAP[pred["tech"]]
    comp_score = COMP_MAP[pred["comp"]]

    # Scope gate: OUT_OF_SCOPE -> seniority contributes 0
    if pred["scope"] == "OUT_OF_SCOPE":
        role_score = 0
    else:
        role_score = SENIORITY_MAP[pred["seniority"]]

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


# ── V6-Compatible Token Mapping ─────────────────────────────────────────────
# Useful for comparing V7 predictions against V6 ground truth.

# V7 location -> V6 loc token
V7_LOC_TO_V6 = {
    "IN_LONDON": "LONDON_OR_REMOTE",
    "FULLY_REMOTE": "LONDON_OR_REMOTE",
    "UK_OTHER": "UK_OTHER",
    "OUTSIDE_UK": "OUTSIDE_UK",
    "UNKNOWN": "MISSING",
}

# V7 seniority -> V6 role token (when scope == IN_SCOPE)
V7_SENIORITY_TO_V6 = {
    "LEVEL_3": "SENIOR_PLUS",
    "LEVEL_2": "MID_LEVEL",
    "LEVEL_1": "NO_SENIORITY",
}


def v7_to_v6_tokens(pred: dict) -> dict:
    """
    Convert V7 token prediction to V6-equivalent tokens.
    Useful for comparing V7 student against V6-labeled ground truth.

    Returns dict with V6 field names: loc, role, tech, comp.
    """
    loc = V7_LOC_TO_V6[pred["location"]]

    if pred["scope"] == "OUT_OF_SCOPE":
        role = "NO_SENIORITY"
    else:
        role = V7_SENIORITY_TO_V6[pred["seniority"]]

    return {
        "loc": loc,
        "role": role,
        "tech": pred["tech"],
        "comp": pred["comp"],
    }

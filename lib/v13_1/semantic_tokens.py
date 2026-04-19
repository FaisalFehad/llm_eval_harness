"""
Semantic Token Vocabulary and Score Conversion for V7 Student Model.

V7 has 5 fields (10 JSON keys):
  - loc   — where the job is based
  - arr   — remote/hybrid/in-office (informational, no score)
  - sen   — experience level
  - tech  — tracked technology array + scope gate (OOS -> seniority=0)
  - comp  — salary range

Tech field:
  - Array of individual tokens: ["NODE", "REACT", "JS_TS", "AI_ML"]
  - ["OOS"] = not an engineering role OR no tracked tech -> seniority forced to 0
  - Score = sum of individual token scores

Scores:
  loc_score  = LOCATION_MAP[loc]
  role_score = SENIORITY_MAP[sen] if tech != ["OOS"] else 0
  tech_score = sum(TECH_INDIVIDUAL_MAP[t] for t in tech)
  comp_score = COMP_MAP[comp]

No arithmetic is performed by the model — all computation is here.
"""

# ── Token Vocabularies ───────────────────────────────────────────────────────

LOCATION_TOKENS = ("IN_LONDON", "REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNK")
WORK_ARRANGEMENT_TOKENS = ("REMOTE", "HYBRID", "IN_OFFICE", "UNK")
SENIORITY_TOKENS = ("LEVEL_3", "LEVEL_2", "LEVEL_1")

# Individual tech tokens (V7 uses arrays, not combo strings)
TECH_INDIVIDUAL_TOKENS = ("OOS", "NODE", "REACT", "JS_TS", "AI_ML")

COMP_TOKENS = (
    "NO_GBP", "UP_TO_ONLY", "BELOW_45K", "RANGE_45_54K",
    "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K",
)

# ── Score Maps ───────────────────────────────────────────────────────────────
# Only fields that contribute to the numeric score have maps.
# arr is informational (no score).
# tech=["OOS"] forces seniority to 0 (scope gate).

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

# Individual tech token scores (summed for arrays)
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

# ── All valid tokens per field ───────────────────────────────────────────────
# Scalar fields use tuples; tech is validated separately as array of individual tokens.

SCALAR_FIELD_TOKENS = {
    "loc": LOCATION_TOKENS,
    "arr": WORK_ARRANGEMENT_TOKENS,
    "sen": SENIORITY_TOKENS,
    "comp": COMP_TOKENS,
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

# The 10 expected fields in V7 output (5 raw + 5 token), in order
V7_EXPECTED_FIELDS = (
    "loc_raw", "loc",
    "arr_raw", "arr",
    "sen_raw", "sen",
    "tech_raw", "tech",
    "comp_raw", "comp",
)

# The 5 raw fields
V7_RAW_FIELDS = (
    "loc_raw", "arr_raw",
    "sen_raw", "tech_raw", "comp_raw",
)

# The 5 token fields
V7_TOKEN_FIELDS = ("loc", "arr", "sen", "tech", "comp")


def validate_prediction(parsed: dict) -> dict:
    """
    Validate a parsed JSON object as a V7 semantic prediction.
    Returns {"valid": bool, "errors": [...], "corrected": {...}, "fuzzy_corrections": [...]}.

    Tech field is validated as an array of individual tokens.
    """
    errors = []
    fuzzy_corrections = []

    # Check required fields — token fields always required, raw fields optional
    for field in V7_TOKEN_FIELDS:
        if field not in parsed:
            errors.append(f"Missing field: {field}")

    if errors:
        return {"valid": False, "errors": errors, "fuzzy_corrections": []}

    # Copy raw fields as-is (optional — may not be present in student output)
    corrected = {}
    for f in V7_RAW_FIELDS:
        val = parsed.get(f)
        corrected[f] = None if val is None else str(val)

    # Validate scalar token fields (loc, arr, sen, comp)
    all_valid = True
    for field_name in ("loc", "arr", "sen", "comp"):
        tokens = SCALAR_FIELD_TOKENS[field_name]
        str_value = str(parsed.get(field_name, ""))
        matched = fuzzy_match(str_value, tokens)
        if matched:
            corrected[field_name] = matched
            if matched != str_value:
                fuzzy_corrections.append(f'{field_name}: "{str_value}" -> "{matched}"')
        else:
            errors.append(f'Invalid {field_name} token: "{str_value}"')
            all_valid = False

    # Validate tech as array of individual tokens
    raw_tech = parsed.get("tech")
    if not isinstance(raw_tech, list):
        # If it's a string, try to handle gracefully
        if isinstance(raw_tech, str):
            matched = fuzzy_match(raw_tech, TECH_INDIVIDUAL_TOKENS)
            if matched:
                corrected["tech"] = [matched]
                fuzzy_corrections.append(f'tech: wrapped string "{raw_tech}" in array')
            else:
                errors.append(f'Invalid tech token: "{raw_tech}" (expected array)')
                all_valid = False
        else:
            errors.append(f"tech must be an array, got {type(raw_tech).__name__}")
            all_valid = False
    else:
        validated_tech = []
        for item in raw_tech:
            str_item = str(item)
            matched = fuzzy_match(str_item, TECH_INDIVIDUAL_TOKENS)
            if matched:
                validated_tech.append(matched)
                if matched != str_item:
                    fuzzy_corrections.append(f'tech element: "{str_item}" -> "{matched}"')
            else:
                errors.append(f'Invalid tech token: "{str_item}"')
                all_valid = False
        if len(validated_tech) == 0 and all_valid:
            # Empty array defaults to OOS
            validated_tech.append("OOS")
            fuzzy_corrections.append('tech: empty array defaulted to ["OOS"]')
        # Deduplicate
        deduped = list(dict.fromkeys(validated_tech))  # preserves order
        if len(deduped) < len(validated_tech):
            fuzzy_corrections.append(f"tech: removed {len(validated_tech) - len(deduped)} duplicate(s)")
        # OOS must not mix with real tokens
        if "OOS" in deduped and len(deduped) > 1:
            without_oos = [t for t in deduped if t != "OOS"]
            fuzzy_corrections.append(f'tech: removed OOS (mixed with {",".join(without_oos)})')
            corrected["tech"] = without_oos
        else:
            corrected["tech"] = deduped

    if not all_valid:
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

    Scope gate: if tech includes "OOS", role_score is forced to 0.
    Tech score = sum of individual token scores.

    Output keys are backward-compatible with V6:
      loc_score, role_score, tech_score, comp_score, score, label
    """
    loc_score = LOCATION_MAP.get(pred["loc"], 0)
    comp_score = COMP_MAP.get(pred["comp"], 0)

    # Tech score = sum of individual token scores
    tech_array = pred["tech"]
    if isinstance(tech_array, str):
        # Backward compat: if somehow a string, wrap it
        tech_array = [tech_array]
    is_oos = len(tech_array) == 0 or "OOS" in tech_array
    tech_score = 0 if is_oos else sum(
        TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech_array
    )

    # Scope gate: OOS -> seniority contributes 0
    role_score = 0 if is_oos else SENIORITY_MAP.get(pred["sen"], 0)

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


# ── Tech Array Utilities ─────────────────────────────────────────────────────

TECH_ORDER = ("NODE", "REACT", "JS_TS", "AI_ML")


def tech_array_to_combo_string(tech: list[str]) -> str:
    """
    Convert tech array to a canonical combo string for backward compat.
    Follows the fixed order: NODE_REACT_JS_TS_AI_ML
    e.g., ["JS_TS", "NODE"] -> "NODE_JS_TS"
    """
    if "OOS" in tech or len(tech) == 0:
        return "OOS"
    sorted_tech = [t for t in TECH_ORDER if t in tech]
    return "_".join(sorted_tech) if sorted_tech else "OOS"


# ── V6-Compatible Token Mapping ─────────────────────────────────────────────
# Useful for comparing V7 predictions against V6 ground truth.

# V7 loc -> V6 loc token
V7_LOC_TO_V6 = {
    "IN_LONDON": "LONDON_OR_REMOTE",
    "REMOTE": "LONDON_OR_REMOTE",
    "UK_OTHER": "UK_OTHER",
    "OUTSIDE_UK": "OUTSIDE_UK",
    "UNK": "MISSING",
}

# V7 seniority -> V6 role token
V7_SENIORITY_TO_V6 = {
    "LEVEL_3": "SENIOR_PLUS",
    "LEVEL_2": "MID_LEVEL",
    "LEVEL_1": "NO_SENIORITY",
}

# V7 tech combo string -> V6 tech token
V7_COMBO_TO_V6 = {
    "OOS": "NONE",
    "NODE": "NODE", "JS_TS": "JS_TS", "AI_ML": "AI_ML",
    "NODE_JS_TS": "NODE_JS_TS", "JS_TS_AI_ML": "JS_TS_AI_ML",
    "NODE_AI_ML": "NODE_AI_ML", "NODE_JS_TS_AI_ML": "NODE_JS_TS_AI_ML",
    # REACT combos -> nearest V6 equivalent
    "REACT": "JS_TS", "NODE_REACT": "NODE_JS_TS",
    "REACT_JS_TS": "JS_TS", "REACT_AI_ML": "JS_TS_AI_ML",
    "NODE_REACT_JS_TS": "NODE_JS_TS", "NODE_REACT_AI_ML": "NODE_JS_TS_AI_ML",
    "REACT_JS_TS_AI_ML": "JS_TS_AI_ML", "NODE_REACT_JS_TS_AI_ML": "NODE_JS_TS_AI_ML",
}


def v7_to_v6_tokens(pred: dict) -> dict:
    """
    Convert V7 token prediction to V6-equivalent tokens.
    Useful for comparing V7 student against V6-labeled ground truth.

    Returns dict with V6 field names: loc, role, tech, comp.
    """
    loc = V7_LOC_TO_V6.get(pred["loc"], "MISSING")
    tech_array = pred["tech"]
    if isinstance(tech_array, str):
        tech_array = [tech_array]
    is_oos = "OOS" in tech_array or len(tech_array) == 0
    role = "NO_SENIORITY" if is_oos else V7_SENIORITY_TO_V6.get(pred["sen"], "NO_SENIORITY")
    combo = tech_array_to_combo_string(tech_array)
    tech = V7_COMBO_TO_V6.get(combo, "NONE")

    return {
        "loc": loc,
        "role": role,
        "tech": tech,
        "comp": pred["comp"],
    }

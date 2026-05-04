"""Semantic Token Vocabulary and Score Conversion for V16.

Extends V7 with V2 teacher fine-grained comp tokens while keeping all other
scoring identical. V1/V2 comp tokens coexist: V2 tokens (50_60K, 70_80K, …)
map to scores; legacy V1 tokens (RANGE_45_54K, RANGE_55_74K, …) remain
supported for backward compatibility when reading old data.
"""

# ── Token Vocabularies ───────────────────────────────────────────────────────

LOCATION_TOKENS = ("IN_LONDON", "REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNK")
WORK_ARRANGEMENT_TOKENS = ("REMOTE", "HYBRID", "IN_OFFICE", "UNK")
SENIORITY_TOKENS = ("LEVEL_3", "LEVEL_2", "LEVEL_1")
TECH_INDIVIDUAL_TOKENS = ("OOS", "NODE", "REACT", "JS_TS", "AI_ML")

# V2 fine-grained comp tokens (canonical)
COMP_TOKENS_V2 = (
    "NO_GBP", "UP_TO_ONLY", "BELOW_30K", "30_40K", "40_50K",
    "50_60K", "60_70K", "70_80K", "80_90K", "90_100K",
    "100_120K", "120_140K", "140_160K", "160_180K", "180_200K", "ABOVE_200K",
)

# Union of V2 + legacy V1 tokens (for fuzzy validation)
COMP_TOKENS = COMP_TOKENS_V2 + (
    "BELOW_45K", "RANGE_45_54K", "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K",
)

# ── Score Maps ───────────────────────────────────────────────────────────────

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

# V2 comp map — fine-grained buckets
COMP_MAP = {
    # V2 canonical
    "NO_GBP": 0,
    "UP_TO_ONLY": 0,
    "BELOW_30K": -30,
    "30_40K": -20,
    "40_50K": -10,
    "50_60K": 0,
    "60_70K": 5,
    "70_80K": 10,
    "80_90K": 15,
    "90_100K": 20,
    "100_120K": 25,
    "120_140K": 25,
    "140_160K": 25,
    "160_180K": 25,
    "180_200K": 25,
    "ABOVE_200K": 25,
    # Legacy V1 fallbacks (for reading old data)
    "BELOW_45K": -30,
    "RANGE_45_54K": 0,
    "RANGE_55_74K": 5,
    "RANGE_75_99K": 15,
    "ABOVE_100K": 25,
}

SCALAR_FIELD_TOKENS = {
    "loc": LOCATION_TOKENS,
    "arr": WORK_ARRANGEMENT_TOKENS,
    "sen": SENIORITY_TOKENS,
    "comp": COMP_TOKENS,
}

# ── Fuzzy Matching ───────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
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

V7_EXPECTED_FIELDS = (
    "loc_raw", "loc",
    "arr_raw", "arr",
    "sen_raw", "sen",
    "tech_raw", "tech",
    "comp_raw", "comp",
)

V7_RAW_FIELDS = (
    "loc_raw", "arr_raw",
    "sen_raw", "tech_raw", "comp_raw",
)

V7_TOKEN_FIELDS = ("loc", "arr", "sen", "tech", "comp")


def validate_prediction(parsed: dict) -> dict:
    errors = []
    fuzzy_corrections = []

    for field in V7_TOKEN_FIELDS:
        if field not in parsed:
            errors.append(f"Missing field: {field}")

    if errors:
        return {"valid": False, "errors": errors, "fuzzy_corrections": []}

    corrected = {}
    for f in V7_RAW_FIELDS:
        val = parsed.get(f)
        corrected[f] = None if val is None else str(val)

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

    raw_tech = parsed.get("tech")
    if not isinstance(raw_tech, list):
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
            validated_tech.append("OOS")
            fuzzy_corrections.append('tech: empty array defaulted to ["OOS"]')
        deduped = list(dict.fromkeys(validated_tech))
        if len(deduped) < len(validated_tech):
            fuzzy_corrections.append(f"tech: removed {len(validated_tech) - len(deduped)} duplicate(s)")
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
    loc_score = LOCATION_MAP.get(pred["loc"], 0)
    comp_score = COMP_MAP.get(pred["comp"], 0)

    tech_array = pred["tech"]
    if isinstance(tech_array, str):
        tech_array = [tech_array]
    is_oos = len(tech_array) == 0 or "OOS" in tech_array
    tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP.get(t, 0) for t in tech_array)

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

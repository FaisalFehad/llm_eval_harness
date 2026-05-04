"""
Promptfoo custom Python assertion for V15 semantic token scoring.

Shows clear expected vs actual for each field, with pass/fail per field.
"""
import json
import re


LOCATION_MAP = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
SENIORITY_MAP = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
TECH_MAP = {"NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10, "OOS": 0}
COMP_MAP = {"NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_30K": -30, "30_40K": -20, "40_50K": -10,
            "50_60K": 0, "60_70K": 5, "70_80K": 10, "80_90K": 15, "90_100K": 20,
            "100_120K": 25, "120_140K": 25, "140_160K": 25, "160_180K": 25,
            "180_200K": 25, "ABOVE_200K": 25,
            # Legacy V1 coarse buckets (fallback)
            "BELOW_45K": -30, "RANGE_45_54K": 0, "RANGE_55_74K": 5,
            "RANGE_75_99K": 15, "ABOVE_100K": 25}


def compute_label(tokens):
    loc_score = LOCATION_MAP.get(tokens.get("loc", "UNK"), 0)
    tech = tokens.get("tech", ["OOS"])
    if not isinstance(tech, list):
        tech = ["OOS"]
    is_oos = "OOS" in tech or len(tech) == 0
    role_score = 0 if is_oos else SENIORITY_MAP.get(tokens.get("sen", "LEVEL_1"), 0)
    tech_score = 0 if is_oos else sum(TECH_MAP.get(t, 0) for t in tech)
    comp_score = COMP_MAP.get(tokens.get("comp", "NO_GBP"), 0)
    score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
    label = "good_fit" if score >= 70 else ("maybe" if score >= 50 else "bad_fit")
    return label, score


def get_assert(output, context):
    metadata = context.get("test", {}).get("metadata", {})
    job_index = metadata.get("job_index", "?")

    # Parse model output — strip thinking prefix and find JSON
    raw = output.strip()
    # Strip various thinking prefixes
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if raw.startswith("Thinking:"):
        raw = raw.split("Thinking:")[-1].strip()
    # Find the first { in the output
    if not raw.startswith("{"):
        idx = raw.find("{")
        if idx >= 0:
            raw = raw[idx:]
        else:
            return {
                "pass": False,
                "score": 0,
                "reason": f"Job {job_index}: PARSE FAIL — no JSON found in output",
            }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            # Fix unquoted keys: only match keys right after { or ,
            fixed = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', raw)
            parsed = json.loads(fixed)
        except (json.JSONDecodeError, ValueError):
            # Try to find just the JSON object
            match = re.search(r'\{[^{}]*"loc"[^{}]*"comp"[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    return {
                        "pass": False,
                        "score": 0,
                        "reason": f"Job {job_index}: PARSE FAIL — could not parse JSON from model output",
                    }
            else:
                return {
                    "pass": False,
                    "score": 0,
                    "reason": f"Job {job_index}: PARSE FAIL — could not parse JSON from model output",
                }

    # Extract predicted tokens
    pred = {
        "loc": parsed.get("loc", "UNK"),
        "arr": parsed.get("arr", "UNK"),
        "sen": parsed.get("sen", "LEVEL_2"),
        "tech": sorted(parsed.get("tech", ["OOS"])) if isinstance(parsed.get("tech"), list) else ["OOS"],
        "comp": parsed.get("comp", "NO_GBP"),
    }

    # Golden values
    golden = {
        "loc": metadata.get("golden_loc", ""),
        "arr": metadata.get("golden_arr", ""),
        "sen": metadata.get("golden_sen", ""),
        "tech": sorted(json.loads(metadata.get("golden_tech", '["OOS"]'))),
        "comp": metadata.get("golden_comp", ""),
    }
    golden_label = metadata.get("golden_label", "")

    # Compute predicted label
    pred_label, pred_score = compute_label(pred)
    label_ok = pred_label == golden_label

    # Per-field comparison
    fields = ["loc", "arr", "sen", "tech", "comp"]
    field_results = []
    wrong_fields = []

    for f in fields:
        ok = pred[f] == golden[f]
        icon = "✓" if ok else "✗"
        if not ok:
            wrong_fields.append(f)
        field_results.append(f"{icon} {f}: expected={golden[f]} got={pred[f]}")

    # Build clear summary
    lines = [f"Job {job_index}: {golden_label}({compute_label(golden)[1]}pts) → {pred_label}({pred_score}pts)"]
    if label_ok and not wrong_fields:
        lines.append("ALL FIELDS CORRECT")
    elif label_ok:
        lines.append(f"Label OK but {len(wrong_fields)} field(s) wrong: {', '.join(wrong_fields)}")
    else:
        lines.append(f"LABEL WRONG ({golden_label} → {pred_label}) — {len(wrong_fields)} field(s) wrong")
    lines.append("")
    lines.extend(field_results)

    reason = "\n".join(lines)

    # Score
    if label_ok and not wrong_fields:
        score = 1.0
    elif label_ok:
        score = 0.7
    else:
        score = 0.0

    return {
        "pass": label_ok,
        "score": score,
        "reason": reason,
        "componentResults": [
            {"pass": pred[f] == golden[f], "score": 1.0 if pred[f] == golden[f] else 0.0,
             "reason": f"{f}: expected={golden[f]} got={pred[f]}"}
            for f in fields
        ],
    }

/**
 * Semantic Token Vocabulary and Score Conversion for V7 Student Model.
 *
 * V7 has 5 fields (10 JSON keys):
 *   - loc   — where the job is based
 *   - arr   — remote/hybrid/in-office (informational, no score)
 *   - sen   — experience level
 *   - tech  — tracked technology array + scope gate (OOS -> seniority=0)
 *   - comp  — salary range
 *
 * Tech field:
 *   - Array of individual tokens: ["NODE", "REACT", "JS_TS", "AI_ML"]
 *   - ["OOS"] = not an engineering role OR no tracked tech -> seniority forced to 0
 *   - Score = sum of individual token scores
 *
 * Scores:
 *   loc_score  = LOCATION_MAP[loc]
 *   role_score = SENIORITY_MAP[sen] if tech != ["OOS"] else 0
 *   tech_score = sum(TECH_INDIVIDUAL_MAP[t] for t in tech)
 *   comp_score = COMP_MAP[comp]
 *
 * No arithmetic is performed by the model — all computation is here.
 */

// ── Token Vocabularies ──────────────────────────────────────────────────────

export const LOCATION_TOKENS = [
  "IN_LONDON",
  "REMOTE",
  "UK_OTHER",
  "OUTSIDE_UK",
  "UNK",
] as const;

export const WORK_ARRANGEMENT_TOKENS = [
  "REMOTE",
  "HYBRID",
  "IN_OFFICE",
  "UNK",
] as const;

export const SENIORITY_TOKENS = ["LEVEL_3", "LEVEL_2", "LEVEL_1"] as const;

// Individual tech tokens (V7 uses arrays, not combo strings)
export const TECH_INDIVIDUAL_TOKENS = [
  "OOS",
  "NODE",
  "REACT",
  "JS_TS",
  "AI_ML",
] as const;

export const COMP_TOKENS = [
  "NO_GBP",
  "UP_TO_ONLY",
  "BELOW_45K",
  "RANGE_45_54K",
  "RANGE_55_74K",
  "RANGE_75_99K",
  "ABOVE_100K",
] as const;

export type LocationToken = (typeof LOCATION_TOKENS)[number];
export type WorkArrangementToken = (typeof WORK_ARRANGEMENT_TOKENS)[number];
export type SeniorityToken = (typeof SENIORITY_TOKENS)[number];
export type TechIndividualToken = (typeof TECH_INDIVIDUAL_TOKENS)[number];
export type CompToken = (typeof COMP_TOKENS)[number];

// ── Score Maps ──────────────────────────────────────────────────────────────
// Only fields that contribute to the numeric score have maps.
// arr is informational (no score).
// tech=["OOS"] forces seniority to 0 (scope gate).

export const LOCATION_MAP: Record<LocationToken, number> = {
  IN_LONDON: 25,
  REMOTE: 25,
  UK_OTHER: 10,
  OUTSIDE_UK: -50,
  UNK: 0,
};

export const SENIORITY_MAP: Record<SeniorityToken, number> = {
  LEVEL_3: 25,
  LEVEL_2: 15,
  LEVEL_1: 0,
};

// Individual tech token scores (summed for arrays)
export const TECH_INDIVIDUAL_MAP: Record<TechIndividualToken, number> = {
  OOS: 0,
  NODE: 10,
  REACT: 5,
  JS_TS: 5,
  AI_ML: 10,
};

export const COMP_MAP: Record<CompToken, number> = {
  NO_GBP: 0,
  UP_TO_ONLY: 0,
  BELOW_45K: -30,
  RANGE_45_54K: 0,
  RANGE_55_74K: 5,
  RANGE_75_99K: 15,
  ABOVE_100K: 25,
};

// ── Validation Sets ─────────────────────────────────────────────────────────

const LOCATION_SET = new Set<string>(LOCATION_TOKENS);
const WORK_ARRANGEMENT_SET = new Set<string>(WORK_ARRANGEMENT_TOKENS);
const SENIORITY_SET = new Set<string>(SENIORITY_TOKENS);
const TECH_INDIVIDUAL_SET = new Set<string>(TECH_INDIVIDUAL_TOKENS);
const COMP_SET = new Set<string>(COMP_TOKENS);

// Scalar token fields and their valid sets (tech handled separately as array)
const SCALAR_FIELD_TOKEN_SETS: Record<string, { tokens: ReadonlyArray<string>; set: Set<string> }> = {
  loc: { tokens: LOCATION_TOKENS, set: LOCATION_SET },
  arr: { tokens: WORK_ARRANGEMENT_TOKENS, set: WORK_ARRANGEMENT_SET },
  sen: { tokens: SENIORITY_TOKENS, set: SENIORITY_SET },
  comp: { tokens: COMP_TOKENS, set: COMP_SET },
};

// ── Types ───────────────────────────────────────────────────────────────────

export type V7SemanticPrediction = {
  loc_raw: string | null;
  loc: LocationToken;
  arr_raw: string | null;
  arr: WorkArrangementToken;
  sen_raw: string | null;
  sen: SeniorityToken;
  tech_raw: string | null;
  tech: TechIndividualToken[];
  comp_raw: string | null;
  comp: CompToken;
};

export type ComputedResult = {
  loc_score: number;
  role_score: number;
  tech_score: number;
  comp_score: number;
  score: number;
  label: "good_fit" | "maybe" | "bad_fit";
};

// ── The 10 expected fields in V7 output ─────────────────────────────────────

export const V7_EXPECTED_FIELDS = [
  "loc_raw",
  "loc",
  "arr_raw",
  "arr",
  "sen_raw",
  "sen",
  "tech_raw",
  "tech",
  "comp_raw",
  "comp",
] as const;

export const V7_RAW_FIELDS = [
  "loc_raw",
  "arr_raw",
  "sen_raw",
  "tech_raw",
  "comp_raw",
] as const;

export const V7_TOKEN_FIELDS = [
  "loc",
  "arr",
  "sen",
  "tech",
  "comp",
] as const;

// ── Fuzzy Matching ──────────────────────────────────────────────────────────

/**
 * Levenshtein edit distance between two strings.
 */
function editDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0),
  );
  for (let i = 0; i <= m; i++) dp[i]![0] = i;
  for (let j = 0; j <= n; j++) dp[0]![j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i]![j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1]![j - 1]!
          : 1 + Math.min(dp[i - 1]![j]!, dp[i]![j - 1]!, dp[i - 1]![j - 1]!);
    }
  }
  return dp[m]![n]!;
}

/**
 * Fuzzy-match a string against a set of valid tokens.
 * Returns the match if edit distance <= 2, otherwise null.
 */
function fuzzyMatch(
  value: string,
  validTokens: ReadonlyArray<string>,
): string | null {
  if (validTokens.includes(value)) return value;

  const upper = value.toUpperCase();
  if (validTokens.includes(upper)) return upper;

  let bestMatch: string | null = null;
  let bestDist = Infinity;
  for (const token of validTokens) {
    const dist = editDistance(upper, token);
    if (dist < bestDist) {
      bestDist = dist;
      bestMatch = token;
    }
  }
  return bestDist <= 2 ? bestMatch : null;
}

// ── Validation ──────────────────────────────────────────────────────────────

export type ValidationResult = {
  valid: boolean;
  errors: string[];
  corrected?: V7SemanticPrediction;
  fuzzyCorrections: string[];
};

/**
 * Validate a parsed JSON object as a V7 semantic prediction.
 * Applies fuzzy matching for minor misspellings.
 * Tech field is validated as an array of individual tokens.
 */
export function validateSemanticPrediction(
  parsed: Record<string, unknown>,
): ValidationResult {
  const errors: string[] = [];
  const fuzzyCorrections: string[] = [];

  // Check token fields are present (raw fields are optional)
  for (const field of V7_TOKEN_FIELDS) {
    if (!(field in parsed)) {
      errors.push(`Missing field: ${field}`);
    }
  }
  if (errors.length > 0) {
    return { valid: false, errors, fuzzyCorrections };
  }

  const corrected: Record<string, unknown> = {};

  // Copy reason fields (string or null)
  for (const f of V7_RAW_FIELDS) {
    const val = parsed[f];
    corrected[f] = val === null || val === undefined ? null : String(val);
  }

  // Validate scalar token fields (loc, arr, sen, comp)
  let allValid = true;
  for (const fieldName of ["loc", "arr", "sen", "comp"] as const) {
    const { tokens } = SCALAR_FIELD_TOKEN_SETS[fieldName]!;
    const strValue = String(parsed[fieldName] ?? "");
    const matched = fuzzyMatch(strValue, tokens);
    if (matched) {
      corrected[fieldName] = matched;
      if (matched !== strValue) {
        fuzzyCorrections.push(`${fieldName}: "${strValue}" -> "${matched}"`);
      }
    } else {
      errors.push(`Invalid ${fieldName} token: "${strValue}"`);
      allValid = false;
    }
  }

  // Validate tech as array of individual tokens
  const rawTech = parsed.tech;
  if (!Array.isArray(rawTech)) {
    // If it's a string, try to handle gracefully
    if (typeof rawTech === "string") {
      // Single token as string — wrap in array
      const matched = fuzzyMatch(rawTech, TECH_INDIVIDUAL_TOKENS);
      if (matched) {
        corrected.tech = [matched];
        fuzzyCorrections.push(`tech: wrapped string "${rawTech}" in array`);
      } else {
        errors.push(`Invalid tech token: "${rawTech}" (expected array)`);
        allValid = false;
      }
    } else {
      errors.push(`tech must be an array, got ${typeof rawTech}`);
      allValid = false;
    }
  } else {
    const validatedTech: string[] = [];
    for (const item of rawTech) {
      const strItem = String(item);
      const matched = fuzzyMatch(strItem, TECH_INDIVIDUAL_TOKENS);
      if (matched) {
        validatedTech.push(matched);
        if (matched !== strItem) {
          fuzzyCorrections.push(`tech element: "${strItem}" -> "${matched}"`);
        }
      } else {
        errors.push(`Invalid tech token: "${strItem}"`);
        allValid = false;
      }
    }
    if (validatedTech.length === 0 && allValid) {
      // Empty array defaults to OOS
      validatedTech.push("OOS");
      fuzzyCorrections.push(`tech: empty array defaulted to ["OOS"]`);
    }
    // Deduplicate
    const deduped = [...new Set(validatedTech)];
    if (deduped.length < validatedTech.length) {
      fuzzyCorrections.push(`tech: removed ${validatedTech.length - deduped.length} duplicate(s)`);
    }
    // OOS must not mix with real tokens
    if (deduped.includes("OOS") && deduped.length > 1) {
      const withoutOOS = deduped.filter(t => t !== "OOS");
      fuzzyCorrections.push(`tech: removed OOS (mixed with ${withoutOOS.join(",")})`);
      corrected.tech = withoutOOS;
    } else {
      corrected.tech = deduped;
    }
  }

  if (!allValid) {
    return { valid: false, errors, fuzzyCorrections };
  }

  return {
    valid: true,
    errors: [],
    corrected: corrected as unknown as V7SemanticPrediction,
    fuzzyCorrections,
  };
}

// ── Score Computation ───────────────────────────────────────────────────────

/**
 * Compute numeric score and label from V7 semantic tokens.
 * This is the ONLY place where tokens become numbers.
 *
 * Scope gate: if tech includes "OOS", role_score is forced to 0.
 *
 * Output keys are backward-compatible with V6:
 *   loc_score, role_score, tech_score, comp_score, score, label
 */
export function computeFromTokens(pred: V7SemanticPrediction): ComputedResult {
  const loc_score = LOCATION_MAP[pred.loc];
  const comp_score = COMP_MAP[pred.comp];

  // Tech score = sum of individual token scores
  const techArray = pred.tech;
  const isOOS = techArray.length === 0 || techArray.includes("OOS");
  const tech_score = isOOS
    ? 0
    : techArray.reduce((sum, t) => sum + (TECH_INDIVIDUAL_MAP[t as TechIndividualToken] ?? 0), 0);

  // Scope gate: OOS -> seniority contributes 0
  const role_score = isOOS ? 0 : SENIORITY_MAP[pred.sen];

  const raw = loc_score + role_score + tech_score + comp_score;
  const score = Math.max(0, Math.min(100, raw));

  let label: "good_fit" | "maybe" | "bad_fit";
  if (score >= 70) label = "good_fit";
  else if (score >= 50) label = "maybe";
  else label = "bad_fit";

  return { loc_score, role_score, tech_score, comp_score, score, label };
}

// ── Tech Array Utilities ────────────────────────────────────────────────────

/**
 * Convert tech array to a canonical combo string for backward compat.
 * Follows the fixed order: NODE_REACT_JS_TS_AI_ML
 * e.g., ["JS_TS", "NODE"] -> "NODE_JS_TS"
 */
const TECH_ORDER = ["NODE", "REACT", "JS_TS", "AI_ML"] as const;

export function techArrayToComboString(tech: string[]): string {
  if (tech.includes("OOS") || tech.length === 0) return "OOS";
  const sorted = TECH_ORDER.filter(t => tech.includes(t));
  return sorted.length > 0 ? sorted.join("_") : "OOS";
}

// ── V6-Compatible Token Mapping ─────────────────────────────────────────────
// Useful for comparing V7 predictions against V6 ground truth.

import type { LocToken, RoleToken, TechToken as V6TechToken } from "./semantic-tokens.js";

/** V7 loc -> V6 loc token */
export const V7_LOC_TO_V6: Record<LocationToken, LocToken> = {
  IN_LONDON: "LONDON_OR_REMOTE",
  REMOTE: "LONDON_OR_REMOTE",
  UK_OTHER: "UK_OTHER",
  OUTSIDE_UK: "OUTSIDE_UK",
  UNK: "MISSING",
};

/** V7 seniority -> V6 role token */
export const V7_SENIORITY_TO_V6: Record<SeniorityToken, RoleToken> = {
  LEVEL_3: "SENIOR_PLUS",
  LEVEL_2: "MID_LEVEL",
  LEVEL_1: "NO_SENIORITY",
};

/** Map V7 tech combo string to V6 tech token */
const V7_COMBO_TO_V6: Record<string, V6TechToken> = {
  OOS: "NONE",
  NODE: "NODE",
  JS_TS: "JS_TS",
  AI_ML: "AI_ML",
  NODE_JS_TS: "NODE_JS_TS",
  JS_TS_AI_ML: "JS_TS_AI_ML",
  NODE_AI_ML: "NODE_AI_ML",
  NODE_JS_TS_AI_ML: "NODE_JS_TS_AI_ML",
  // REACT combos -> nearest V6 equivalent
  REACT: "JS_TS",
  NODE_REACT: "NODE_JS_TS",
  REACT_JS_TS: "JS_TS",
  REACT_AI_ML: "JS_TS_AI_ML",
  NODE_REACT_JS_TS: "NODE_JS_TS",
  NODE_REACT_AI_ML: "NODE_JS_TS_AI_ML",
  REACT_JS_TS_AI_ML: "JS_TS_AI_ML",
  NODE_REACT_JS_TS_AI_ML: "NODE_JS_TS_AI_ML",
};

/**
 * Convert V7 token prediction to V6-equivalent tokens.
 * Useful for comparing V7 student against V6-labeled ground truth.
 */
export function v7ToV6Tokens(pred: V7SemanticPrediction): {
  loc: LocToken;
  role: RoleToken;
  tech: V6TechToken;
  comp: CompToken;
} {
  const loc = V7_LOC_TO_V6[pred.loc];
  const isOOS = pred.tech.includes("OOS") || pred.tech.length === 0;
  const role: RoleToken = isOOS
    ? "NO_SENIORITY"
    : V7_SENIORITY_TO_V6[pred.sen];
  const combo = techArrayToComboString(pred.tech);
  const tech = V7_COMBO_TO_V6[combo] ?? "NONE";

  return { loc, role, tech, comp: pred.comp };
}

/**
 * Check if a string is a valid token for a given V7 field.
 * For tech, checks individual tokens.
 */
export function isValidToken(
  field: "loc" | "arr" | "sen" | "tech" | "comp",
  value: string,
): boolean {
  if (field === "tech") return TECH_INDIVIDUAL_SET.has(value);
  return SCALAR_FIELD_TOKEN_SETS[field]!.set.has(value);
}

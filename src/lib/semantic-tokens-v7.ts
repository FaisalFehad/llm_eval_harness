/**
 * Semantic Token Vocabulary and Score Conversion for V7 Student Model.
 *
 * V7 splits V6's 4 fields (loc, role, tech, comp) into 6 fields:
 *   - location          (was loc)  — where the job is based
 *   - work_arrangement  (NEW)      — remote/hybrid/in-office (informational, no score)
 *   - scope             (NEW)      — engineering or not (gate: OUT_OF_SCOPE -> seniority=0)
 *   - seniority         (was role) — experience level
 *   - tech              (unchanged)
 *   - comp              (unchanged)
 *
 * Scores are backward-compatible with V6:
 *   loc_score  = LOCATION_MAP[location]
 *   role_score = SENIORITY_MAP[seniority] if scope == IN_SCOPE else 0
 *   tech_score = TECH_MAP[tech]
 *   comp_score = COMP_MAP[comp]
 *
 * No arithmetic is performed by the model — all computation is here.
 */

// ── Token Vocabularies ──────────────────────────────────────────────────────

export const LOCATION_TOKENS = [
  "IN_LONDON",
  "FULLY_REMOTE",
  "UK_OTHER",
  "OUTSIDE_UK",
  "UNKNOWN",
] as const;

export const WORK_ARRANGEMENT_TOKENS = [
  "REMOTE",
  "HYBRID",
  "IN_OFFICE",
  "UNKNOWN",
] as const;

export const SCOPE_TOKENS = ["IN_SCOPE", "OUT_OF_SCOPE"] as const;

export const SENIORITY_TOKENS = ["LEVEL_3", "LEVEL_2", "LEVEL_1"] as const;

export const TECH_TOKENS = [
  "NONE",
  "JS_TS",
  "NODE",
  "NODE_JS_TS",
  "AI_ML",
  "JS_TS_AI_ML",
  "NODE_AI_ML",
  "NODE_JS_TS_AI_ML",
] as const;

export const COMP_TOKENS = [
  "NO_GBP",
  "UP_TO_ONLY",
  "BELOW_45K",
  "RANGE_55_74K",
  "RANGE_75_99K",
  "ABOVE_100K",
] as const;

export type LocationToken = (typeof LOCATION_TOKENS)[number];
export type WorkArrangementToken = (typeof WORK_ARRANGEMENT_TOKENS)[number];
export type ScopeToken = (typeof SCOPE_TOKENS)[number];
export type SeniorityToken = (typeof SENIORITY_TOKENS)[number];
export type TechToken = (typeof TECH_TOKENS)[number];
export type CompToken = (typeof COMP_TOKENS)[number];

// ── Score Maps ──────────────────────────────────────────────────────────────
// Only fields that contribute to the numeric score have maps.
// work_arrangement is informational (no score).
// scope is a gate (OUT_OF_SCOPE -> seniority forced to 0).

export const LOCATION_MAP: Record<LocationToken, number> = {
  IN_LONDON: 25,
  FULLY_REMOTE: 25,
  UK_OTHER: 10,
  OUTSIDE_UK: -50,
  UNKNOWN: 0,
};

export const SENIORITY_MAP: Record<SeniorityToken, number> = {
  LEVEL_3: 25,
  LEVEL_2: 15,
  LEVEL_1: 0,
};

export const TECH_MAP: Record<TechToken, number> = {
  NONE: 0,
  JS_TS: 5,
  NODE: 10,
  NODE_JS_TS: 15,
  AI_ML: 10,
  JS_TS_AI_ML: 15,
  NODE_AI_ML: 20,
  NODE_JS_TS_AI_ML: 25,
};

export const COMP_MAP: Record<CompToken, number> = {
  NO_GBP: 0,
  UP_TO_ONLY: 0,
  BELOW_45K: -30,
  RANGE_55_74K: 5,
  RANGE_75_99K: 15,
  ABOVE_100K: 25,
};

// ── Validation Sets ─────────────────────────────────────────────────────────

const LOCATION_SET = new Set<string>(LOCATION_TOKENS);
const WORK_ARRANGEMENT_SET = new Set<string>(WORK_ARRANGEMENT_TOKENS);
const SCOPE_SET = new Set<string>(SCOPE_TOKENS);
const SENIORITY_SET = new Set<string>(SENIORITY_TOKENS);
const TECH_SET = new Set<string>(TECH_TOKENS);
const COMP_SET = new Set<string>(COMP_TOKENS);

// All token fields and their valid sets (for iteration)
const FIELD_TOKEN_SETS: Record<string, { tokens: ReadonlyArray<string>; set: Set<string> }> = {
  location: { tokens: LOCATION_TOKENS, set: LOCATION_SET },
  work_arrangement: { tokens: WORK_ARRANGEMENT_TOKENS, set: WORK_ARRANGEMENT_SET },
  scope: { tokens: SCOPE_TOKENS, set: SCOPE_SET },
  seniority: { tokens: SENIORITY_TOKENS, set: SENIORITY_SET },
  tech: { tokens: TECH_TOKENS, set: TECH_SET },
  comp: { tokens: COMP_TOKENS, set: COMP_SET },
};

// ── Types ───────────────────────────────────────────────────────────────────

export type V7SemanticPrediction = {
  location_reason: string;
  location: LocationToken;
  work_arrangement_reason: string;
  work_arrangement: WorkArrangementToken;
  scope_reason: string;
  scope: ScopeToken;
  seniority_reason: string;
  seniority: SeniorityToken;
  tech_reason: string;
  tech: TechToken;
  comp_reason: string;
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

// ── The 12 expected fields in V7 output ─────────────────────────────────────

export const V7_EXPECTED_FIELDS = [
  "location_reason",
  "location",
  "work_arrangement_reason",
  "work_arrangement",
  "scope_reason",
  "scope",
  "seniority_reason",
  "seniority",
  "tech_reason",
  "tech",
  "comp_reason",
  "comp",
] as const;

export const V7_REASON_FIELDS = [
  "location_reason",
  "work_arrangement_reason",
  "scope_reason",
  "seniority_reason",
  "tech_reason",
  "comp_reason",
] as const;

export const V7_TOKEN_FIELDS = [
  "location",
  "work_arrangement",
  "scope",
  "seniority",
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
 */
export function validateSemanticPrediction(
  parsed: Record<string, unknown>,
): ValidationResult {
  const errors: string[] = [];
  const fuzzyCorrections: string[] = [];

  // Check all 12 fields are present
  for (const field of V7_EXPECTED_FIELDS) {
    if (!(field in parsed)) {
      errors.push(`Missing field: ${field}`);
    }
  }
  if (errors.length > 0) {
    return { valid: false, errors, fuzzyCorrections };
  }

  // Copy reason fields as-is
  const corrected: Record<string, unknown> = {};
  for (const f of V7_REASON_FIELDS) {
    corrected[f] = String(parsed[f] ?? "");
  }

  // Validate and fuzzy-match each token field
  let allValid = true;
  for (const fieldName of V7_TOKEN_FIELDS) {
    const { tokens } = FIELD_TOKEN_SETS[fieldName]!;
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
 * Scope gate: if scope == OUT_OF_SCOPE, role_score is forced to 0.
 *
 * Output keys are backward-compatible with V6:
 *   loc_score, role_score, tech_score, comp_score, score, label
 */
export function computeFromTokens(pred: V7SemanticPrediction): ComputedResult {
  const loc_score = LOCATION_MAP[pred.location];
  const tech_score = TECH_MAP[pred.tech];
  const comp_score = COMP_MAP[pred.comp];

  // Scope gate: OUT_OF_SCOPE -> seniority contributes 0
  const role_score =
    pred.scope === "OUT_OF_SCOPE" ? 0 : SENIORITY_MAP[pred.seniority];

  const raw = loc_score + role_score + tech_score + comp_score;
  const score = Math.max(0, Math.min(100, raw));

  let label: "good_fit" | "maybe" | "bad_fit";
  if (score >= 70) label = "good_fit";
  else if (score >= 50) label = "maybe";
  else label = "bad_fit";

  return { loc_score, role_score, tech_score, comp_score, score, label };
}

// ── V6-Compatible Token Mapping ─────────────────────────────────────────────
// Useful for comparing V7 predictions against V6 ground truth.

import type { LocToken, RoleToken } from "./semantic-tokens.js";

/** V7 location -> V6 loc token */
export const V7_LOC_TO_V6: Record<LocationToken, LocToken> = {
  IN_LONDON: "LONDON_OR_REMOTE",
  FULLY_REMOTE: "LONDON_OR_REMOTE",
  UK_OTHER: "UK_OTHER",
  OUTSIDE_UK: "OUTSIDE_UK",
  UNKNOWN: "MISSING",
};

/** V7 seniority -> V6 role token (when scope == IN_SCOPE) */
export const V7_SENIORITY_TO_V6: Record<SeniorityToken, RoleToken> = {
  LEVEL_3: "SENIOR_PLUS",
  LEVEL_2: "MID_LEVEL",
  LEVEL_1: "NO_SENIORITY",
};

/**
 * Convert V7 token prediction to V6-equivalent tokens.
 * Useful for comparing V7 student against V6-labeled ground truth.
 */
export function v7ToV6Tokens(pred: V7SemanticPrediction): {
  loc: LocToken;
  role: RoleToken;
  tech: TechToken;
  comp: CompToken;
} {
  const loc = V7_LOC_TO_V6[pred.location];
  const role: RoleToken =
    pred.scope === "OUT_OF_SCOPE"
      ? "NO_SENIORITY"
      : V7_SENIORITY_TO_V6[pred.seniority];

  return { loc, role, tech: pred.tech, comp: pred.comp };
}

/**
 * Check if a string is a valid token for a given V7 field.
 */
export function isValidToken(
  field: "location" | "work_arrangement" | "scope" | "seniority" | "tech" | "comp",
  value: string,
): boolean {
  return FIELD_TOKEN_SETS[field]!.set.has(value);
}

// ── Reasoning Cross-Check ───────────────────────────────────────────────────

export type ConsistencyCheck = {
  consistent: boolean;
  issues: string[];
};

/**
 * Cross-check V7 reason fields against their paired token fields.
 *
 * V7 reasons follow the pattern "...phrase -> TOKEN". We check:
 * 1. Each reason's arrow-token matches the actual token field
 * 2. Scope gate: if scope=OUT_OF_SCOPE, seniority should be LEVEL_1
 */
export function crossCheckReasoning(pred: V7SemanticPrediction): ConsistencyCheck {
  const issues: string[] = [];

  // Check each reason field's arrow-token matches the token field
  const pairs: Array<{ reasonField: keyof V7SemanticPrediction; tokenField: keyof V7SemanticPrediction }> = [
    { reasonField: "location_reason", tokenField: "location" },
    { reasonField: "work_arrangement_reason", tokenField: "work_arrangement" },
    { reasonField: "scope_reason", tokenField: "scope" },
    { reasonField: "seniority_reason", tokenField: "seniority" },
    { reasonField: "tech_reason", tokenField: "tech" },
    { reasonField: "comp_reason", tokenField: "comp" },
  ];

  for (const { reasonField, tokenField } of pairs) {
    const reason = String(pred[reasonField]);
    const token = String(pred[tokenField]);

    // Extract token after last "->" in the reason
    const arrowIdx = reason.lastIndexOf("->");
    if (arrowIdx !== -1) {
      const reasonToken = reason.slice(arrowIdx + 2).trim();
      if (reasonToken !== token) {
        issues.push(
          `${String(tokenField)}: reason says "${reasonToken}" but token is "${token}"`,
        );
      }
    }
  }

  // Scope gate consistency: OUT_OF_SCOPE should have LEVEL_1 seniority
  if (pred.scope === "OUT_OF_SCOPE" && pred.seniority !== "LEVEL_1") {
    issues.push(
      `scope=OUT_OF_SCOPE but seniority=${pred.seniority} (expected LEVEL_1)`,
    );
  }

  return { consistent: issues.length === 0, issues };
}

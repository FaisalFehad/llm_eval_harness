/**
 * Semantic Token Vocabulary and Score Conversion for V5 Student Model.
 *
 * The student model outputs semantic labels (e.g., "LONDON_OR_REMOTE")
 * instead of numeric scores. This module converts those labels to scores,
 * computes the total, and assigns a final label deterministically.
 *
 * No arithmetic is performed by the model — all computation is here.
 */

// ── Token Vocabularies ──────────────────────────────────────────────────────

export const LOC_TOKENS = [
  "LONDON_OR_REMOTE",
  "UK_OTHER",
  "OUTSIDE_UK",
  "MISSING",
] as const;

export const ROLE_TOKENS = [
  "SENIOR_PLUS",
  "MID_LEVEL",
  "NO_SENIORITY",
] as const;

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

export type LocToken = (typeof LOC_TOKENS)[number];
export type RoleToken = (typeof ROLE_TOKENS)[number];
export type TechToken = (typeof TECH_TOKENS)[number];
export type CompToken = (typeof COMP_TOKENS)[number];

// ── Score Maps ──────────────────────────────────────────────────────────────

export const LOC_MAP: Record<LocToken, number> = {
  LONDON_OR_REMOTE: 25,
  UK_OTHER: 10,
  OUTSIDE_UK: -50,
  MISSING: 0,
};

export const ROLE_MAP: Record<RoleToken, number> = {
  SENIOR_PLUS: 25,
  MID_LEVEL: 15,
  NO_SENIORITY: 0,
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

// ── All valid tokens per field (for validation) ─────────────────────────────

const LOC_SET = new Set<string>(LOC_TOKENS);
const ROLE_SET = new Set<string>(ROLE_TOKENS);
const TECH_SET = new Set<string>(TECH_TOKENS);
const COMP_SET = new Set<string>(COMP_TOKENS);

// ── Types ───────────────────────────────────────────────────────────────────

export type SemanticPrediction = {
  loc_reason: string;
  loc: LocToken;
  role_reason: string;
  role: RoleToken;
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
 * Returns the match if edit distance ≤ 2, otherwise null.
 */
function fuzzyMatch(value: string, validTokens: ReadonlyArray<string>): string | null {
  // Exact match first
  if (validTokens.includes(value)) return value;

  // Try uppercase
  const upper = value.toUpperCase();
  if (validTokens.includes(upper)) return upper;

  // Fuzzy match
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
  corrected?: SemanticPrediction;
  fuzzyCorrections: string[];
};

/**
 * Validate a parsed JSON object as a semantic prediction.
 * Applies fuzzy matching for minor misspellings.
 */
export function validateSemanticPrediction(
  parsed: Record<string, unknown>,
): ValidationResult {
  const errors: string[] = [];
  const fuzzyCorrections: string[] = [];

  // Check required fields exist (8-field interleaved format)
  for (const field of ["loc_reason", "loc", "role_reason", "role", "tech_reason", "tech", "comp_reason", "comp"]) {
    if (!(field in parsed)) {
      errors.push(`Missing field: ${field}`);
    }
  }
  if (errors.length > 0) {
    return { valid: false, errors, fuzzyCorrections };
  }

  // Validate each token field (with fuzzy matching)
  const fields: Array<{
    name: string;
    reasonField: string;
    value: unknown;
    tokens: ReadonlyArray<string>;
  }> = [
    { name: "loc", reasonField: "loc_reason", value: parsed.loc, tokens: LOC_TOKENS },
    { name: "role", reasonField: "role_reason", value: parsed.role, tokens: ROLE_TOKENS },
    { name: "tech", reasonField: "tech_reason", value: parsed.tech, tokens: TECH_TOKENS },
    { name: "comp", reasonField: "comp_reason", value: parsed.comp, tokens: COMP_TOKENS },
  ];

  const corrected: Record<string, unknown> = {
    loc_reason: String(parsed.loc_reason ?? ""),
    role_reason: String(parsed.role_reason ?? ""),
    tech_reason: String(parsed.tech_reason ?? ""),
    comp_reason: String(parsed.comp_reason ?? ""),
  };
  let allValid = true;

  for (const { name, value, tokens } of fields) {
    const strValue = String(value ?? "");
    const matched = fuzzyMatch(strValue, tokens);
    if (matched) {
      corrected[name] = matched;
      if (matched !== strValue) {
        fuzzyCorrections.push(`${name}: "${strValue}" → "${matched}"`);
      }
    } else {
      errors.push(`Invalid ${name} token: "${strValue}"`);
      allValid = false;
    }
  }

  if (!allValid) {
    return { valid: false, errors, fuzzyCorrections };
  }

  return {
    valid: true,
    errors: [],
    corrected: corrected as unknown as SemanticPrediction,
    fuzzyCorrections,
  };
}

// ── Score Computation ───────────────────────────────────────────────────────

/**
 * Compute numeric score and label from semantic tokens.
 * This is the ONLY place where tokens become numbers.
 */
export function computeFromTokens(pred: SemanticPrediction): ComputedResult {
  const loc_score = LOC_MAP[pred.loc];
  const role_score = ROLE_MAP[pred.role];
  const tech_score = TECH_MAP[pred.tech];
  const comp_score = COMP_MAP[pred.comp];

  const raw = loc_score + role_score + tech_score + comp_score;
  const score = Math.max(0, Math.min(100, raw));

  let label: "good_fit" | "maybe" | "bad_fit";
  if (score >= 70) label = "good_fit";
  else if (score >= 50) label = "maybe";
  else label = "bad_fit";

  return { loc_score, role_score, tech_score, comp_score, score, label };
}

// ── Reasoning Cross-Check ───────────────────────────────────────────────────

export type ConsistencyCheck = {
  consistent: boolean;
  issues: string[];
};

/**
 * Check if reasoning contains a phrase as a genuine positive mention,
 * not preceded by "no " and not followed by "but ignored" / "ignored".
 */
function hasUnqualifiedMatch(reasoning: string, phrases: string[]): boolean {
  for (const phrase of phrases) {
    let idx = 0;
    while ((idx = reasoning.indexOf(phrase, idx)) !== -1) {
      // Check for negation: "no " immediately before the phrase
      const prefix = reasoning.slice(Math.max(0, idx - 3), idx);
      if (prefix.endsWith("no ")) {
        idx += phrase.length;
        continue;
      }

      // Check for qualification: "but ignored" or "ignored" after the phrase
      const after = reasoning.slice(idx + phrase.length, idx + phrase.length + 20);
      if (after.includes("but ignored") || after.includes("ignored") || after.includes("but not")) {
        idx += phrase.length;
        continue;
      }

      // Genuine unqualified match
      return true;
    }
  }
  return false;
}

/**
 * Cross-check per-field reasoning against the semantic tokens.
 * With the 8-field format, each _reason field is checked against its own token.
 * Each _reason should end with "-> TOKEN" matching the actual token field.
 */
export function crossCheckReasoning(pred: SemanticPrediction): ConsistencyCheck {
  const issues: string[] = [];

  // Check each _reason field ends with -> matching token
  const fieldPairs: Array<{ reason: string; token: string; name: string }> = [
    { reason: pred.loc_reason, token: pred.loc, name: "loc" },
    { reason: pred.role_reason, token: pred.role, name: "role" },
    { reason: pred.tech_reason, token: pred.tech, name: "tech" },
    { reason: pred.comp_reason, token: pred.comp, name: "comp" },
  ];

  for (const { reason, token, name } of fieldPairs) {
    if (!reason) continue;

    // Check that reason ends with -> TOKEN (the key consistency check)
    const arrowMatch = reason.match(/->\s*([A-Z][A-Z0-9_]*)\s*$/);
    if (arrowMatch) {
      if (arrowMatch[1] !== token) {
        issues.push(`${name}_reason ends with "-> ${arrowMatch[1]}" but ${name}="${token}"`);
      }
    } else {
      issues.push(`${name}_reason doesn't end with "-> TOKEN" pattern`);
    }
  }

  // Location-specific checks
  const lr = pred.loc_reason.toLowerCase();
  if (pred.loc === "LONDON_OR_REMOTE") {
    if (!lr.includes("london") && !lr.includes("remote")) {
      issues.push("loc=LONDON_OR_REMOTE but loc_reason doesn't mention London or Remote");
    }
  }

  // Tech-specific checks
  const tr = pred.tech_reason.toLowerCase();
  if (
    (pred.tech === "NONE" || pred.tech === "JS_TS") &&
    hasUnqualifiedMatch(tr, ["node.js", "nodejs", "node"])
  ) {
    issues.push(`tech=${pred.tech} but tech_reason mentions Node without "ignored"`);
  }
  if (
    (pred.tech === "NONE" || pred.tech === "NODE") &&
    hasUnqualifiedMatch(tr, ["typescript", "javascript"])
  ) {
    issues.push(`tech=${pred.tech} but tech_reason mentions JS/TS without "ignored"`);
  }

  // Comp-specific checks
  const cr = pred.comp_reason.toLowerCase();
  if (pred.comp === "NO_GBP" && cr.includes("£") && !cr.includes("ignored") && !cr.includes("not gbp")) {
    if (cr.includes("mid") && !cr.includes("£45") && !cr.includes("£46") && !cr.includes("£47") &&
        !cr.includes("£48") && !cr.includes("£49") && !cr.includes("£50") && !cr.includes("£51") &&
        !cr.includes("£52") && !cr.includes("£53") && !cr.includes("£54")) {
      issues.push("comp=NO_GBP but comp_reason mentions £ salary without 'ignored' or midpoint in 45-54k range");
    }
  }

  if (pred.comp !== "UP_TO_ONLY" && (cr.includes("up to £") || cr.includes("no lower bound"))) {
    issues.push(`comp=${pred.comp} but comp_reason mentions 'Up to' / 'no lower bound' -> should be UP_TO_ONLY`);
  }

  return { consistent: issues.length === 0, issues };
}

// ── Token ↔ Numeric Score Conversion (for comparing with old format) ────────

/**
 * Convert numeric scores (old format) to semantic tokens (new format).
 * Useful for converting existing labeled data.
 */
export function numericToTokens(scores: {
  loc: number;
  role: number;
  tech: number;
  comp: number;
}): { loc: LocToken; role: RoleToken; tech: TechToken; comp: CompToken } {
  // Reverse lookup: find the token that maps to this score
  const loc = (Object.entries(LOC_MAP).find(([, v]) => v === scores.loc)?.[0] ??
    "MISSING") as LocToken;
  const role = (Object.entries(ROLE_MAP).find(([, v]) => v === scores.role)?.[0] ??
    "NO_SENIORITY") as RoleToken;
  const tech = (Object.entries(TECH_MAP).find(([, v]) => v === scores.tech)?.[0] ??
    "NONE") as TechToken;
  const comp = (Object.entries(COMP_MAP).find(([, v]) => v === scores.comp)?.[0] ??
    "NO_GBP") as CompToken;

  return { loc, role, tech, comp };
}

/**
 * Check if a string is a valid token for a given field.
 */
export function isValidToken(field: "loc" | "role" | "tech" | "comp", value: string): boolean {
  switch (field) {
    case "loc": return LOC_SET.has(value);
    case "role": return ROLE_SET.has(value);
    case "tech": return TECH_SET.has(value);
    case "comp": return COMP_SET.has(value);
  }
}

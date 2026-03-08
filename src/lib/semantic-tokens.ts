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
  reasoning: string;
  loc: LocToken;
  role: RoleToken;
  tech: TechToken;
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

  // Check required fields exist
  for (const field of ["reasoning", "loc", "role", "tech", "comp"]) {
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
    value: unknown;
    tokens: ReadonlyArray<string>;
  }> = [
    { name: "loc", value: parsed.loc, tokens: LOC_TOKENS },
    { name: "role", value: parsed.role, tokens: ROLE_TOKENS },
    { name: "tech", value: parsed.tech, tokens: TECH_TOKENS },
    { name: "comp", value: parsed.comp, tokens: COMP_TOKENS },
  ];

  const corrected: Record<string, unknown> = {
    reasoning: String(parsed.reasoning ?? ""),
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
 * Cross-check reasoning text against the semantic tokens.
 * Flags cases where the reasoning contradicts the token.
 */
export function crossCheckReasoning(pred: SemanticPrediction): ConsistencyCheck {
  const issues: string[] = [];
  const r = pred.reasoning.toLowerCase();

  // Location checks
  if (pred.loc === "LONDON_OR_REMOTE") {
    if (!r.includes("london") && !r.includes("remote")) {
      issues.push("loc=LONDON_OR_REMOTE but reasoning doesn't mention London or Remote");
    }
  }
  if (pred.loc === "OUTSIDE_UK") {
    if (r.includes("→ london") || r.includes("→ uk_other")) {
      issues.push("loc=OUTSIDE_UK but reasoning suggests UK location");
    }
  }

  // Tech checks — exclude negated ("no X found") and qualified ("X found but ignored")
  if (
    (pred.tech === "NONE" || pred.tech === "JS_TS") &&
    hasUnqualifiedMatch(r, ["node.js found", "nodejs found", "node found"])
  ) {
    issues.push(`tech=${pred.tech} but reasoning says Node was found`);
  }
  if (
    (pred.tech === "NONE" || pred.tech === "NODE") &&
    hasUnqualifiedMatch(r, ["typescript found", "javascript found"])
  ) {
    issues.push(`tech=${pred.tech} but reasoning says JS/TS was found`);
  }

  // Comp checks
  if (pred.comp === "NO_GBP" && r.includes("£") && !r.includes("ignored") && !r.includes("daily rate")) {
    // Might have a GBP salary mentioned but not ignored
    if (r.includes("midpoint") && !r.includes("£45") && !r.includes("£46") && !r.includes("£47") &&
        !r.includes("£48") && !r.includes("£49") && !r.includes("£50") && !r.includes("£51") &&
        !r.includes("£52") && !r.includes("£53") && !r.includes("£54")) {
      issues.push("comp=NO_GBP but reasoning mentions £ salary without 'ignored' or midpoint in 45-54k range");
    }
  }

  // Comp: "Up to" pattern should map to UP_TO_ONLY
  if (pred.comp !== "UP_TO_ONLY" && (r.includes("up to £") || r.includes("no lower bound"))) {
    issues.push(`comp=${pred.comp} but reasoning mentions 'Up to' / 'no lower bound' → should be UP_TO_ONLY`);
  }

  // Role: title keyword mismatch
  if (pred.role === "NO_SENIORITY" && (r.includes("'senior'") || r.includes("senior in title") || r.includes("'lead'") || r.includes("lead in title"))) {
    issues.push("role=NO_SENIORITY but reasoning says senior/lead found in title");
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

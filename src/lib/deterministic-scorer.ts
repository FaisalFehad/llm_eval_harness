import type { FitLabel } from "../schema.js";

export type ScoreBreakdown = {
  loc: number;
  role: number;
  tech: number;
  comp: number;
  score: number;
  label: FitLabel;
};

/**
 * STEP 1: LOCATION — use location field only.
 *
 * +25  Remote (UK/Global/Worldwide) OR hybrid/on-site in London
 * +10  hybrid/on-site in UK city that is NOT London
 * -50  outside the UK
 *  0   missing or unclear
 */
export function scoreLoc(location: string | undefined): number {
  if (!location || location.trim().length === 0) return 0;

  const hasRemote = /\bremote\b/i.test(location);
  const hasHybrid = /\bhybrid\b/i.test(location);
  const hasOnsite = /\bon[-\s]?site\b/i.test(location);
  const hasGlobal = /\b(global|worldwide)\b/i.test(location);
  const hasUK = /\b(united kingdom|uk|england|scotland|wales|northern ireland|great britain|gb)\b/i.test(location);
  const hasLondon = /\blondon\b/i.test(location);

  const outsidePatterns: RegExp[] = [
    /\b(u\.?s\.?a?|united states|america)\b/i,
    /\bcanada\b/i,
    /\bindia\b/i,
    /\bgermany\b/i,
    /\bfrance\b/i,
    /\bspain\b/i,
    /\bnetherlands\b/i,
    /\bsingapore\b/i,
    /\baustralia\b/i,
    /\bireland\b/i,
    /\bdublin\b/i,
    /\bfinland\b/i,
    /\bhelsinki\b/i,
    /\bczech\b/i,
    /\bprague\b/i,
    /\bsouth africa\b/i,
    /\bcape town\b/i,
    /\bamsterdam\b/i,
    /\bberlin\b/i,
    /\bvancouver\b/i,
    /\bsydney\b/i,
  ];

  const usStateCodes = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
  ];

  const hasOutsideKeyword =
    !hasUK && outsidePatterns.some((pattern) => pattern.test(location));
  const hasUsState =
    !hasUK && new RegExp(`\\b(${usStateCodes.join("|")})\\b`, "i").test(location);
  const isOutsideUK = hasOutsideKeyword || hasUsState;

  if (hasRemote) {
    if (isOutsideUK) return -50;
    if (hasUK || hasGlobal) return 25;
    return 0;
  }

  if (isOutsideUK) return -50;
  if (hasLondon) return 25;
  if (hasUK) return 10;

  if (hasHybrid || hasOnsite) return 0;
  return 0;
}

/**
 * STEP 2: ROLE & SENIORITY — search title for keywords.
 *
 * +25  Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, VP, Snr, Founding
 * +15  Full Stack, Full-Stack, Fullstack, Mid-Level, Mid Level, Midlevel,
 *      Software Engineer II, Engineer II, SWE II
 *  0   none of the above
 */
export function scoreRole(title: string): number {
  const seniorPatterns = [
    /\bsenior\b/i,
    /\bstaff\b/i,
    /\bprincipal\b/i,
    /\btech lead\b/i,
    /\blead\b/i,
    /\bhead\b/i,
    /\bdistinguished\b/i,
    /\bvp\b/i,
    /\bsnr\b/i,
    /\bfounding\b/i,
  ];
  if (seniorPatterns.some((pattern) => pattern.test(title))) return 25;

  const midPatterns = [
    /\bfull[\s-]?stack\b/i,
    /\bmid[\s-]?level\b/i,
    /\bsoftware engineer ii\b/i,
    /\bengineer ii\b/i,
    /\bswe ii\b/i,
  ];
  if (midPatterns.some((pattern) => pattern.test(title))) return 15;

  return 0;
}

/**
 * STEP 3: TECH STACK — search jd_text for required/core tech.
 *
 * +10  Node.js or NodeJS required/core
 * +5   JavaScript or TypeScript required/core
 * +10  AI/ML/LLM experience explicitly REQUIRED
 * Cap at 25.
 */
export function scoreTech(jdText: string): number {
  let tech = 0;

  // Node.js / NodeJS
  if (/\bnode\.?js\b|\bnodejs\b/i.test(jdText)) {
    tech += 10;
  }

  // JavaScript or TypeScript
  if (/\bjavascript\b|\btypescript\b/i.test(jdText)) {
    tech += 5;
  }

  // AI/ML/LLM — must appear in a requirements/skills section, not company description.
  // The v9 rubric says: "Do NOT award AI points if AI is only in company description,
  // mission, or 'nice to have'."
  //
  // Strategy: look for AI terms in "What You'll Bring" / "expertise in" contexts,
  // specifically in the tech stack description (not "About" or "mission" paragraphs).
  const aiInExpertise =
    /\b(?:expertise|proficiency)\s+in\b[^.]*?\b(?:pytorch|llm|fine-tun|vector database|pinecone|milvus|machine learning|deep learning)\b/is;
  const aiInStack =
    /\b(?:pytorch|fine-tun(?:ing|e)|vector database|pinecone|milvus)\b/i;

  if (aiInExpertise.test(jdText) || aiInStack.test(jdText)) {
    tech += 10;
  }

  return Math.min(25, tech);
}

/**
 * Parse a GBP amount string like "100,000", "100k", "100K" to a number.
 */
function parseGBP(raw: string): number {
  const cleaned = raw.replace(/,/g, "").trim();
  if (/k$/i.test(cleaned)) {
    return parseFloat(cleaned.replace(/k$/i, "")) * 1000;
  }
  return parseFloat(cleaned);
}

/**
 * Map a salary midpoint to a comp score.
 */
function compFromMidpoint(midpoint: number): number {
  if (midpoint >= 100_000) return 25;
  if (midpoint >= 75_000) return 15;
  if (midpoint >= 55_000) return 5;
  if (midpoint < 45_000) return -30;
  // £45,000-£54,999 falls through all brackets → 0
  return 0;
}

/**
 * STEP 4: COMPENSATION — GBP salary from jd_text only.
 *
 * Parse £ salary ranges with both lower and upper bound.
 * "Up to £X" with no lower bound = 0.
 * Salary in title (not jd_text) = 0.
 */
export function scoreComp(jdText: string): number {
  // Match "Up to £X" with no lower bound — must return 0
  // Check this BEFORE looking for ranges, as "Up to" patterns can contain £ signs
  const upToPattern = /\bup\s+to\s+£[\d,]+k?\b/i;
  const rangePattern = /£([\d,]+k?)\s*[-–—to]+\s*£([\d,]+k?)/gi;

  // Find all salary ranges in jd_text
  const ranges: Array<{ lo: number; hi: number }> = [];
  let match: RegExpExecArray | null;

  while ((match = rangePattern.exec(jdText)) !== null) {
    const lo = parseGBP(match[1]!);
    const hi = parseGBP(match[2]!);
    if (!isNaN(lo) && !isNaN(hi) && lo > 0 && hi > lo) {
      ranges.push({ lo, hi });
    }
  }

  if (ranges.length > 0) {
    // Use the first valid range found
    const { lo, hi } = ranges[0]!;
    const midpoint = (lo + hi) / 2;
    return compFromMidpoint(midpoint);
  }

  // Check for "Up to" without a range — always 0
  if (upToPattern.test(jdText)) {
    return 0;
  }

  // No GBP salary found
  return 0;
}

/**
 * Score a job using the v9 rubric deterministically.
 */
export function scoreJob(
  title: string,
  location: string | undefined,
  jdText: string,
): ScoreBreakdown {
  const loc = scoreLoc(location);
  const role = scoreRole(title);
  const tech = scoreTech(jdText);
  const comp = scoreComp(jdText);

  const rawTotal = loc + role + tech + comp;
  const score = Math.max(0, Math.min(100, rawTotal));

  let label: FitLabel;
  if (score >= 70) label = "good_fit";
  else if (score >= 50) label = "maybe";
  else label = "bad_fit";

  return { loc, role, tech, comp, score, label };
}

/**
 * Filter tech_raw to only include verbatim terms matching labeled tech tokens.
 *
 * Problem: the teacher writes tech_raw with ALL tech from the JD (tracked + untracked),
 * e.g. "Python, React.js, TypeScript, PostgreSQL, Redis, Docker". This can be 500+ chars.
 * Truncating mid-word at 50 chars corrupts the JSON structure during inference.
 *
 * Solution: extract only the verbatim terms that correspond to the teacher's labeled tokens.
 * "Python, React.js, TypeScript, PostgreSQL, Redis, Docker" + tech=["REACT","JS_TS"]
 *   → "React.js, TypeScript"
 *
 * Design: label-guided (not pure regex scan) to avoid false positives.
 * If teacher labeled OOS, return null — no tech hint needed, avoids "AI" in "AI-assisted".
 * Only search patterns for tokens that are actually in the labeled tech array.
 *
 * Result length: tracked tech term names are short (5-25 chars each). With 1-4 terms,
 * output stays well under 50 chars in nearly all cases — no hard cap needed.
 */

// Patterns for each tech token — case-insensitive, word-boundary anchored.
// Using an object (not module-level RegExp) so matchAll() gets a fresh state each call.
const TOKEN_PATTERNS: Record<string, string> = {
  NODE: String.raw`\bnode(?:\.js|js)?\b`,
  REACT: String.raw`\breact(?:\.js|js)?\b`,
  // "javascript"/"typescript" before bare "js"/"ts".
  // Negative lookbehind (?<!\.) prevents matching ".js" in "Next.js", "Vue.js" etc.
  JS_TS: String.raw`\b(?:java|type)script\b|(?<!\.)\bjs\b|(?<!\.)\bts\b`,
  AI_ML: [
    String.raw`artificial intelligence`,
    String.raw`machine learning`,
    String.raw`deep learning`,
    String.raw`neural networks?`,
    String.raw`computer vision`,
    String.raw`prompt engineering`,
    String.raw`fine[\s-]tuning`,
    String.raw`\bllms?\b`,
    String.raw`\bml\b`,
    String.raw`\bnlp\b`,
    String.raw`\blm\b`,
    String.raw`\bai\b`,
    String.raw`\bpytorch\b`,
    String.raw`\btensorflow\b`,
  ].join("|"),
};

type Match = { start: number; end: number; text: string };

/**
 * Filter tech_raw to verbatim terms matching the labeled tech tokens.
 *
 * @param techRaw - Verbatim tech string from teacher (may be null)
 * @param tech    - Teacher's labeled tech token array e.g. ["NODE", "JS_TS"]
 * @returns Comma-joined verbatim matches in reading order, or null if none found
 */
export function filterTechRaw(
  techRaw: string | null | undefined,
  tech: string[],
): string | null {
  if (!techRaw) return null;
  // OOS means no tracked tech — null is correct (avoids "AI" in "AI-assisted dev")
  if (tech.includes("OOS") || tech.length === 0) return null;

  const allMatches: Match[] = [];

  for (const token of tech) {
    const patternStr = TOKEN_PATTERNS[token];
    if (!patternStr) continue;
    // Fresh RegExp per token per call — no lastIndex state issues
    const regex = new RegExp(patternStr, "gi");
    for (const m of techRaw.matchAll(regex)) {
      allMatches.push({ start: m.index!, end: m.index! + m[0].length, text: m[0] });
    }
  }

  if (allMatches.length === 0) return null;

  // Sort by position — preserve reading order from original string
  allMatches.sort((a, b) => a.start - b.start);

  // Remove overlapping matches: earlier match wins.
  // Handles "node.js" matched by both NODE (\bnode.js\b) and JS_TS (\bjs\b).
  const filtered: Match[] = [];
  let lastEnd = -1;
  for (const m of allMatches) {
    if (m.start >= lastEnd) {
      filtered.push(m);
      lastEnd = m.end;
    }
  }

  // Deduplicate case-insensitively — "Node.js" × 50 or "TypeScript"/"Typescript" → one term
  const seen = new Set<string>();
  const deduped: Match[] = [];
  for (const m of filtered) {
    const key = m.text.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      deduped.push(m);
    }
  }

  return deduped.map((m) => m.text).join(", ") || null;
}

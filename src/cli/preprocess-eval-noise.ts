/**
 * Preprocess eval rows by trimming JD noise while keeping supervision unchanged.
 *
 * This script only changes input text fields used for prompting:
 * - jd_text (signal extraction + shortening)
 * - job_location/location (light normalization)
 *
 * It does NOT modify token labels (loc/arr/sen/tech/comp).
 *
 * Usage:
 *   npx tsx src/cli/preprocess-eval-noise.ts \
 *     --input data/v7/test_labeled.jsonl \
 *     --output data/v7/test_labeled.preprocessed.jsonl \
 *     --report data/v7/test_labeled.preprocess_report.json \
 *     --max-chars 2200
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type EvalRow = {
  job_id?: string;
  title?: string;
  company?: string;
  job_location?: string;
  location?: string;
  jd_text: string;
  [key: string]: unknown;
};

type Chunk = {
  text: string;
  index: number;
  score: number;
  salience: number;
  drop: boolean;
  critical: boolean;
  counter: boolean;
  hasSalary: boolean;
  hasLocation: boolean;
  hasTrackedTech: boolean;
  hasNonTargetTech: boolean;
  hasOutsideUk: boolean;
  hasNonGbpComp: boolean;
  hasSeniority: boolean;
  hasRequirement: boolean;
};

const DROP_PATTERNS: RegExp[] = [
  /\bequal opportunity\b/i,
  /\bdiversity\b/i,
  /\binclusion\b/i,
  /\bprivacy policy\b/i,
  /\bcookie(s)?\b/i,
  /\bterms and conditions\b/i,
  /\bhow to apply\b/i,
  /\bapply now\b/i,
  /\bour mission\b/i,
  /\babout (the )?company\b/i,
  /\bbenefits include\b/i,
  /\bperks\b/i,
];

const SALARY_PATTERNS: RegExp[] = [
  /£/,
  /\bsalary\b/i,
  /\bcompensation\b/i,
  /\bpay\b/i,
  /\bper annum\b/i,
  /\bannum\b/i,
  /\bp\.?a\.?\b/i,
  /\bday rate\b/i,
];

const NON_GBP_COMP_PATTERNS: RegExp[] = [
  /\$/i,
  /€/i,
  /\busd\b/i,
  /\beur\b/i,
  /\bdollar(s)?\b/i,
  /\beuro(s)?\b/i,
];

const LOCATION_PATTERNS: RegExp[] = [
  /\bremote\b/i,
  /\bhybrid\b/i,
  /\bon[- ]?site\b/i,
  /\bin[- ]?office\b/i,
  /\blondon\b/i,
  /\buk\b/i,
  /\bunited kingdom\b/i,
  /\blocation\b/i,
  /\bbased in\b/i,
];

const OUTSIDE_UK_PATTERNS: RegExp[] = [
  /\boutside (the )?uk\b/i,
  /\b(united states|usa|u\.s\.a\.|canada|australia|germany|france|spain|italy|india|singapore|poland|netherlands|ireland)\b/i,
  /\b(san francisco|new york|toronto|sydney|berlin|dublin|amsterdam|warsaw|bengaluru)\b/i,
];

const TECH_PATTERNS: RegExp[] = [
  /\bnode(\.js|js)?\b/i,
  /\breact(\.js|js)?\b/i,
  /\btypescript\b/i,
  /\bjavascript\b/i,
  /\bai\b/i,
  /\bmachine learning\b/i,
  /\bml\b/i,
  /\bbackend\b/i,
  /\bfrontend\b/i,
  /\bfull[- ]?stack\b/i,
];

const NON_TARGET_TECH_PATTERNS: RegExp[] = [
  /\bpython\b/i,
  /\bjava\b/i,
  /\bc#\b/i,
  /\bc\+\+\b/i,
  /\bphp\b/i,
  /\bruby\b/i,
  /\bgolang\b/i,
  /\bgo\b/i,
  /\bdotnet\b/i,
  /\b\.net\b/i,
  /\bvue\b/i,
  /\bangular\b/i,
  /\baws\b/i,
  /\bterraform\b/i,
  /\bdocker\b/i,
  /\bkubernetes\b/i,
  /\bsql\b/i,
  /\bsnowflake\b/i,
];

const SEN_PATTERNS: RegExp[] = [
  /\bjunior\b/i,
  /\bgraduate\b/i,
  /\bentry[- ]?level\b/i,
  /\bsenior\b/i,
  /\blead\b/i,
  /\bstaff\b/i,
  /\bprincipal\b/i,
  /\bmanager\b/i,
  /\bintern(ship)?\b/i,
];

const REQUIREMENT_PATTERNS: RegExp[] = [
  /\brequirements?\b/i,
  /\bmust have\b/i,
  /\bexperience with\b/i,
  /\bwhat (you('|’)ll|you will) bring\b/i,
  /\bskills?\b/i,
  /\bresponsibilit(y|ies)\b/i,
];

function norm(text: string | null | undefined): string {
  return (text ?? "").replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

function preview(text: string, max = 140): string {
  const t = norm(text);
  if (t.length <= max) return t;
  return `${t.slice(0, max).trimEnd()}...`;
}

function cleanLocation(raw: string | null | undefined): string {
  let text = norm(raw);
  if (!text) return "";
  text = text.replace(/\bArea,\s*/gi, "");

  const pipeSplit = text.split("|").map(norm).filter(Boolean);
  if (pipeSplit.length > 0) {
    text = pipeSplit[0]!;
  }

  const parts = text.split(",").map(norm).filter(Boolean);
  if (parts.length >= 3) {
    text = `${parts[0]}, ${parts[1]}, ${parts[2]}`;
  } else if (parts.length >= 2) {
    text = `${parts[0]}, ${parts[1]}`;
  }

  if (text.length > 90) text = text.slice(0, 90).trimEnd();
  return text;
}

function splitIntoChunks(text: string): string[] {
  const clean = text.replace(/\r/g, "").trim();
  if (!clean) return [];

  const lines = clean.split(/\n+/).map(norm).filter(Boolean);
  if (lines.length >= 8) return lines;

  return clean.split(/(?<=[.!?])\s+/).map(norm).filter(Boolean);
}

function anyMatch(text: string, patterns: RegExp[]): boolean {
  return patterns.some((p) => p.test(text));
}

function scoreChunk(text: string): number {
  const t = text;
  let score = 0;

  if (anyMatch(t, SALARY_PATTERNS)) score += 6;
  if (anyMatch(t, LOCATION_PATTERNS)) score += 5;
  if (anyMatch(t, TECH_PATTERNS)) score += 4;
  if (anyMatch(t, SEN_PATTERNS)) score += 3;
  if (anyMatch(t, REQUIREMENT_PATTERNS)) score += 2;
  if (/^\s*[-*•]/.test(t)) score += 1;

  if (anyMatch(t, DROP_PATTERNS)) score -= 8;
  if (norm(t).length < 24) score -= 1;

  return score;
}

function toScoredChunks(input: string): Chunk[] {
  const chunksRaw = splitIntoChunks(input);
  if (chunksRaw.length === 0) return [];

  const seen = new Set<string>();
  const chunks: Chunk[] = [];
  for (let i = 0; i < chunksRaw.length; i += 1) {
    const text = norm(chunksRaw[i]);
    if (!text) continue;
    const key = text.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const score = scoreChunk(text);
    const hasSalary = anyMatch(text, SALARY_PATTERNS);
    const hasLocation = anyMatch(text, LOCATION_PATTERNS);
    const hasTrackedTech = anyMatch(text, TECH_PATTERNS);
    const hasNonTargetTech = anyMatch(text, NON_TARGET_TECH_PATTERNS);
    const hasOutsideUk = anyMatch(text, OUTSIDE_UK_PATTERNS);
    const hasNonGbpComp = anyMatch(text, NON_GBP_COMP_PATTERNS);
    const hasSeniority = anyMatch(text, SEN_PATTERNS);
    const hasRequirement = anyMatch(text, REQUIREMENT_PATTERNS);
    const drop = anyMatch(text, DROP_PATTERNS);
    const counter = hasOutsideUk || hasNonTargetTech || hasNonGbpComp;
    const critical =
      hasSalary ||
      hasLocation ||
      hasTrackedTech ||
      hasNonTargetTech ||
      hasOutsideUk ||
      hasNonGbpComp ||
      hasSeniority ||
      hasRequirement;

    let salience = score;
    if (counter) salience += 4;
    if (hasSalary) salience += 2;
    if (hasLocation) salience += 1;
    if (drop && !critical) salience -= 5;

    chunks.push({
      text,
      index: i,
      score,
      salience,
      drop,
      critical,
      counter,
      hasSalary,
      hasLocation,
      hasTrackedTech,
      hasNonTargetTech,
      hasOutsideUk,
      hasNonGbpComp,
      hasSeniority,
      hasRequirement,
    });
  }
  return chunks;
}

function renderSelectedChunks(chunks: Chunk[]): string {
  if (chunks.length === 0) return "";
  return chunks
    .sort((a, b) => a.index - b.index)
    .map((c) => c.text)
    .join("\n")
    .trim();
}

function buildRankedSignalText(input: string, maxChars: number, minScore: number): string {
  const chunks = toScoredChunks(input);
  if (chunks.length === 0) return norm(input).slice(0, maxChars);

  const selected: Chunk[] = [];
  if (chunks.length > 0) selected.push(chunks[0]!);

  const ranked = chunks
    .filter((c) => c.index !== 0 && c.score >= minScore)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.index - b.index;
    });

  let used = selected.reduce((sum, c) => sum + c.text.length + 1, 0);
  for (const chunk of ranked) {
    if (used + chunk.text.length + 1 > maxChars) continue;
    selected.push(chunk);
    used += chunk.text.length + 1;
  }

  if (selected.length <= 1) {
    const fallback = renderSelectedChunks(chunks);
    return fallback.slice(0, maxChars).trim();
  }

  return renderSelectedChunks(selected).slice(0, maxChars).trim();
}

function buildSelectiveSignalText(input: string, maxChars: number, minScore: number): string {
  const chunks = toScoredChunks(input);
  if (chunks.length === 0) return norm(input).slice(0, maxChars);

  const byIndex = [...chunks].sort((a, b) => a.index - b.index);
  const bySalience = [...chunks]
    .filter((c) => c.critical || c.score >= minScore || !c.drop)
    .sort((a, b) => {
      if (b.salience !== a.salience) return b.salience - a.salience;
      return a.index - b.index;
    });

  const selected = new Set<number>();
  let used = 0;

  const tryAdd = (chunk: Chunk): boolean => {
    if (selected.has(chunk.index)) return false;
    if (used + chunk.text.length + 1 > maxChars) return false;
    selected.add(chunk.index);
    used += chunk.text.length + 1;
    return true;
  };

  // Keep early context and title framing.
  if (byIndex.length > 0) tryAdd(byIndex[0]!);
  if (byIndex.length > 1) tryAdd(byIndex[1]!);

  // Force-preserve balancing evidence first.
  const counterQuota = Math.max(3, Math.floor(maxChars / 900));
  let counterAdded = 0;
  for (const chunk of byIndex) {
    if (!chunk.counter) continue;
    if (counterAdded >= counterQuota) break;
    if (tryAdd(chunk)) counterAdded += 1;
  }

  // Keep a few tail critical chunks (salary/location often appears late).
  let tailAdded = 0;
  for (const chunk of [...byIndex].reverse()) {
    if (!(chunk.critical || chunk.counter)) continue;
    if (tailAdded >= 3) break;
    if (tryAdd(chunk)) tailAdded += 1;
  }

  // Ensure at least one non-target tech chunk if present.
  const hasSelectedNonTarget = [...selected].some((idx) =>
    byIndex.find((c) => c.index === idx)?.hasNonTargetTech === true,
  );
  if (!hasSelectedNonTarget) {
    const nonTarget = byIndex.find((c) => c.hasNonTargetTech);
    if (nonTarget) tryAdd(nonTarget);
  }

  // Keep salary evidence in both directions when available.
  const gbpSalary = byIndex.find((c) => c.hasSalary && !c.hasNonGbpComp);
  const nonGbpSalary = byIndex.find((c) => c.hasSalary && c.hasNonGbpComp);
  if (gbpSalary) tryAdd(gbpSalary);
  if (nonGbpSalary) tryAdd(nonGbpSalary);

  // Add salient chunks but cap to avoid over-concentration.
  const coreQuota = Math.max(8, Math.floor(maxChars / 300));
  let coreAdded = 0;
  for (const chunk of bySalience) {
    if (coreAdded >= coreQuota) break;
    if (tryAdd(chunk)) coreAdded += 1;
  }

  // Fill remaining budget with in-order context, skipping pure boilerplate.
  for (const chunk of byIndex) {
    if (chunk.drop && !chunk.critical) continue;
    tryAdd(chunk);
  }

  const selectedChunks = byIndex.filter((c) => selected.has(c.index));
  if (selectedChunks.length <= 1) {
    return renderSelectedChunks(byIndex).slice(0, maxChars).trim();
  }

  return renderSelectedChunks(selectedChunks).slice(0, maxChars).trim();
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v7/test_labeled.jsonl";
  const outputPath = getStringArg(args, "output") ?? "data/v7/test_labeled.preprocessed.jsonl";
  const reportPath = getStringArg(args, "report") ?? "data/v7/test_labeled.preprocess_report.json";
  const maxChars = getNumberArg(args, "max-chars") ?? 2200;
  const minScore = getNumberArg(args, "min-score") ?? 2;
  const strategy = (getStringArg(args, "strategy") ?? "ranked").toLowerCase();
  const sampleCount = getNumberArg(args, "sample-count") ?? 8;
  const rewriteLocation = args.flags["rewrite-location"] === true;
  if (strategy !== "ranked" && strategy !== "selective") {
    throw new Error(`Invalid --strategy "${strategy}". Use "ranked" or "selective".`);
  }

  const rows = await readJsonlFile<EvalRow>(inputPath);
  if (rows.length === 0) throw new Error(`No rows in ${inputPath}`);

  let changedJd = 0;
  let changedLoc = 0;
  let sumBefore = 0;
  let sumAfter = 0;
  const samples: Array<Record<string, unknown>> = [];

  const out = rows.map((row) => {
    const beforeJd = norm(row.jd_text);
    const afterJd =
      strategy === "selective"
        ? buildSelectiveSignalText(beforeJd, maxChars, minScore)
        : buildRankedSignalText(beforeJd, maxChars, minScore);
    sumBefore += beforeJd.length;
    sumAfter += afterJd.length;
    if (beforeJd !== afterJd) changedJd += 1;

    const originalLocation = typeof row.job_location === "string" ? row.job_location : row.location;
    const cleanedLocation = cleanLocation(originalLocation ?? "");
    const hasJobLocation = typeof row.job_location === "string";
    const hasLocation = typeof row.location === "string";

    const next: EvalRow = {
      ...row,
      jd_text: afterJd,
    };

    if (rewriteLocation && cleanedLocation && hasJobLocation && row.job_location !== cleanedLocation) {
      next.job_location = cleanedLocation;
      changedLoc += 1;
    } else if (rewriteLocation && cleanedLocation && !hasJobLocation && hasLocation && row.location !== cleanedLocation) {
      next.location = cleanedLocation;
      changedLoc += 1;
    }

    if (samples.length < sampleCount && beforeJd !== afterJd) {
      samples.push({
        job_id: row.job_id ?? null,
        title: row.title ?? null,
        before_len: beforeJd.length,
        after_len: afterJd.length,
        before_preview: preview(beforeJd),
        after_preview: preview(afterJd),
      });
    }

    return next;
  });

  await writeJsonlFile(outputPath, out);
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });

  const report = {
    timestamp: new Date().toISOString(),
    input: inputPath,
    output: outputPath,
    n_rows: rows.length,
    changed_jd_rows: changedJd,
    changed_location_rows: changedLoc,
    avg_jd_len_before: Number((sumBefore / rows.length).toFixed(2)),
    avg_jd_len_after: Number((sumAfter / rows.length).toFixed(2)),
    strategy,
    max_chars: maxChars,
    min_score: minScore,
    rewrite_location: rewriteLocation,
    samples,
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");

  console.log(`Preprocessed eval rows: ${rows.length}`);
  console.log(`Output: ${outputPath}`);
  console.log(`Report: ${reportPath}`);
  console.log(`Strategy: ${strategy}`);
  console.log(`Changed jd_text rows: ${changedJd}`);
  console.log(`Changed location rows: ${changedLoc}`);
  console.log(`Avg jd len: ${report.avg_jd_len_before} -> ${report.avg_jd_len_after}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

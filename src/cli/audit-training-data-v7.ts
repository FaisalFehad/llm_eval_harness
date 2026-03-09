/**
 * Audit V7 training data for quality issues.
 *
 * V7 version: uses 6 fields (location, work_arrangement, scope, seniority,
 * tech, comp) with 12-field interleaved output format.
 *
 * Runs all checks in one pass and produces:
 *   - quarantine/duplicates.jsonl   (moved duplicates)
 *   - quarantine/bad_data.jsonl     (short JDs, missing fields, corrupt)
 *   - quarantine/suspicious.jsonl   (token mismatches, title mismatches)
 *   - stdout: summary report
 *
 * Exit codes:
 *   0 = clean (warnings only)
 *   1 = critical issues found (blocks pipeline)
 *
 * Usage:
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl
 *
 *   # With eval set contamination check:
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --eval-set data/v7/eval.jsonl
 *
 *   # Custom thresholds:
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --min-jd-length 100 \
 *     --max-jd-length 15000 \
 *     --source-cap 5
 *
 *   # Dry run (report only, no quarantine files):
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --dry-run
 *
 *   # Clean mode — produce a fixed file with issues removed:
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --eval-set data/v7/eval.jsonl \
 *     --clean --output data/v7/labeled_clean.jsonl
 *
 *   # Also remove suspicious items (title mismatches, token combos):
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --clean --output data/v7/labeled_clean.jsonl \
 *     --remove-suspicious
 *
 *   # Also remove trivially easy bad_fit jobs:
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --clean --output data/v7/labeled_clean.jsonl \
 *     --remove-trivial
 *
 *   # Pre-label mode (for unlabeled input — before running label-jobs-v7.ts):
 *   npx tsx src/cli/audit-training-data-v7.ts \
 *     --input data/v7/input_pool.jsonl \
 *     --eval-set data/v7/eval.jsonl \
 *     --pre-label
 */

import * as fs from "node:fs";
import * as crypto from "node:crypto";
import * as path from "node:path";
import { parseArgs, getStringArg, getNumberArg, getBooleanArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

// ── Valid V7 token vocabulary ─────────────────────────────────────────────

const VALID_TOKENS: Record<string, Set<string>> = {
  location: new Set(["IN_LONDON", "FULLY_REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNKNOWN"]),
  work_arrangement: new Set(["REMOTE", "HYBRID", "IN_OFFICE", "UNKNOWN"]),
  scope: new Set(["IN_SCOPE", "OUT_OF_SCOPE"]),
  seniority: new Set(["LEVEL_3", "LEVEL_2", "LEVEL_1"]),
  tech: new Set([
    "NONE", "JS_TS", "NODE", "NODE_JS_TS",
    "AI_ML", "JS_TS_AI_ML", "NODE_AI_ML", "NODE_JS_TS_AI_ML",
  ]),
  comp: new Set([
    "NO_GBP", "UP_TO_ONLY", "BELOW_45K",
    "RANGE_55_74K", "RANGE_75_99K", "ABOVE_100K",
  ]),
};

// V7 token fields (the 6 main token fields)
const V7_TOKEN_FIELDS = ["location", "work_arrangement", "scope", "seniority", "tech", "comp"] as const;

// V7 reason fields (paired with each token field)
const V7_REASON_FIELDS = V7_TOKEN_FIELDS.map((f) => `${f}_reason` as const);

// ── Token → score mapping ───────────────────────────────────────────────

const TOKEN_SCORES: Record<string, Record<string, number>> = {
  location: { IN_LONDON: 25, FULLY_REMOTE: 25, UK_OTHER: 10, OUTSIDE_UK: -50, UNKNOWN: 0 },
  seniority: { LEVEL_3: 25, LEVEL_2: 15, LEVEL_1: 0 },
  tech: {
    NONE: 0, JS_TS: 5, NODE: 10, NODE_JS_TS: 15,
    AI_ML: 10, JS_TS_AI_ML: 15, NODE_AI_ML: 20, NODE_JS_TS_AI_ML: 25,
  },
  comp: {
    NO_GBP: 0, UP_TO_ONLY: 0, BELOW_45K: -30,
    RANGE_55_74K: 5, RANGE_75_99K: 15, ABOVE_100K: 25,
  },
};

// ── Types ───────────────────────────────────────────────────────────────

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  job_location: string; // raw location (renamed from "location" to avoid V7 token collision)
  jd_text: string;
  // V7 12-field interleaved output
  location_reason: string;
  location: string;
  work_arrangement_reason: string;
  work_arrangement: string;
  scope_reason: string;
  scope: string;
  seniority_reason: string;
  seniority: string;
  tech_reason: string;
  tech: string;
  comp_reason: string;
  comp: string;
  // Computed
  loc_score?: number;
  role_score?: number;
  tech_score?: number;
  comp_score?: number;
  score?: number;
  label?: string;
  source_file?: string;
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
};

type Issue = {
  line: number;
  job_id: string;
  title: string;
  severity: "CRITICAL" | "WARNING";
  check: string;
  detail: string;
};

type QuarantineEntry = LabeledJob & { _quarantine_reason: string; _line: number };

// ── Helpers ─────────────────────────────────────────────────────────────

function jdFingerprint(jdText: string): string {
  const normalized = jdText.slice(0, 500).toLowerCase().replace(/\s+/g, " ").trim();
  return crypto.createHash("sha256").update(normalized).digest("hex").slice(0, 16);
}

function normalizeCompany(company: string): string {
  return company
    .toLowerCase()
    .replace(/\b(ltd|llc|inc|plc|limited|corp|corporation)\b/gi, "")
    .replace(/[^a-z0-9\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function dedupKey(title: string, company: string): string {
  return `${title.toLowerCase().trim()}||${normalizeCompany(company || "")}`;
}

/**
 * Compute V7 score with scope gate:
 *   - loc_score from location tokens (IN_LONDON/FULLY_REMOTE both = 25)
 *   - role_score from seniority tokens, BUT zeroed if scope=OUT_OF_SCOPE
 *   - tech_score from tech tokens
 *   - comp_score from comp tokens
 */
function computeScore(job: LabeledJob): { score: number; label: string } {
  const locScore = TOKEN_SCORES.location[job.location] ?? 0;
  let roleScore = TOKEN_SCORES.seniority[job.seniority] ?? 0;
  const techScore = TOKEN_SCORES.tech[job.tech] ?? 0;
  const compScore = TOKEN_SCORES.comp[job.comp] ?? 0;

  // Scope gate: OUT_OF_SCOPE zeroes seniority contribution
  if (job.scope === "OUT_OF_SCOPE") {
    roleScore = 0;
  }

  const raw = locScore + roleScore + techScore + compScore;
  const score = Math.max(0, Math.min(100, raw));
  const label = score >= 70 ? "good_fit" : score >= 50 ? "maybe" : "bad_fit";
  return { score, label };
}

function hasHtmlArtifacts(text: string): boolean {
  return /&(amp|lt|gt|nbsp|quot|apos|#\d+);/i.test(text) ||
    /<\/?[a-z][\s\S]*?>/i.test(text);
}

// ── Suspicious token combination checks ─────────────────────────────────

const SENIOR_KEYWORDS = /\b(senior|staff|principal|lead|leader|head|chief|director|vp|cto|founding|snr|sr|iii|distinguished)\b/i;
const ENGINEERING_KEYWORDS = /\b(engineer|developer|programmer|architect|sre|devops|cto|data scientist|ml engineer|tech lead|team lead|technical lead)\b/i;

function checkSuspiciousTokens(job: LabeledJob): string[] {
  const issues: string[] = [];
  const title = job.title || "";

  // "Senior X Engineer" but LEVEL_1 (no seniority)
  if (SENIOR_KEYWORDS.test(title) && ENGINEERING_KEYWORDS.test(title) && job.seniority === "LEVEL_1") {
    issues.push(`Title "${title}" has senior+engineering keywords but seniority=LEVEL_1`);
  }

  // Non-engineering title but LEVEL_3 (unless it's a legitimate engineering management role)
  if (!ENGINEERING_KEYWORDS.test(title) && job.seniority === "LEVEL_3") {
    // Allow "Engineering Manager" and similar where "engineer" isn't in title
    if (!/\bengineering\b/i.test(title) && !/\bmanager\b.*\bengineering\b/i.test(title)) {
      issues.push(`Title "${title}" has no engineering keywords but seniority=LEVEL_3`);
    }
  }

  // "London" in raw location but location != IN_LONDON
  const rawLoc = job.job_location || "";
  if (/\blondon\b/i.test(rawLoc) && job.location !== "IN_LONDON") {
    issues.push(`Raw location contains "London" but location=${job.location}`);
  }

  // "Remote" in raw location — could be IN_LONDON (if UK) or FULLY_REMOTE
  if (/\bremote\b/i.test(rawLoc) && job.location !== "IN_LONDON" && job.location !== "FULLY_REMOTE") {
    issues.push(`Raw location contains "Remote" but location=${job.location}`);
  }

  // Scope-seniority consistency: OUT_OF_SCOPE should typically have LEVEL_1
  // (because score gates seniority to 0 anyway, teacher should use LEVEL_1)
  if (job.scope === "OUT_OF_SCOPE" && job.seniority !== "LEVEL_1") {
    issues.push(`scope=OUT_OF_SCOPE but seniority=${job.seniority} (expected LEVEL_1 when out of scope)`);
  }

  // IN_SCOPE engineering job — title should have engineering keywords
  if (job.scope === "IN_SCOPE" && !ENGINEERING_KEYWORDS.test(title) && !/\bengineering\b/i.test(title)) {
    issues.push(`scope=IN_SCOPE but title "${title}" has no engineering keywords`);
  }

  return issues;
}

// ── Reasoning-token mismatch check ──────────────────────────────────────

/**
 * Check each _reason field ends with "-> TOKEN" matching the actual token field.
 * V7 version: 6 reason→token pairs (12-field interleaved format).
 */
function checkReasoningTokenMismatch(job: LabeledJob): string[] {
  const issues: string[] = [];

  const fieldPairs: Array<{ reasonField: string; tokenField: string; name: string }> = [
    { reasonField: "location_reason", tokenField: "location", name: "location" },
    { reasonField: "work_arrangement_reason", tokenField: "work_arrangement", name: "work_arrangement" },
    { reasonField: "scope_reason", tokenField: "scope", name: "scope" },
    { reasonField: "seniority_reason", tokenField: "seniority", name: "seniority" },
    { reasonField: "tech_reason", tokenField: "tech", name: "tech" },
    { reasonField: "comp_reason", tokenField: "comp", name: "comp" },
  ];

  for (const { reasonField, tokenField, name } of fieldPairs) {
    const reason = (job as Record<string, unknown>)[reasonField] as string | undefined;
    const token = (job as Record<string, unknown>)[tokenField] as string | undefined;

    if (!reason) {
      issues.push(`Missing ${reasonField} field`);
      continue;
    }

    // Check that reason ends with -> TOKEN matching the actual token
    const arrowMatch = reason.match(/->\s*([A-Z][A-Z0-9_]*)\s*$/);
    if (arrowMatch) {
      if (arrowMatch[1] !== token) {
        issues.push(
          `${reasonField} ends with "-> ${arrowMatch[1]}" but ${tokenField}="${token}"`,
        );
      }
    } else {
      issues.push(`${reasonField} doesn't end with "-> TOKEN" pattern: "${reason.slice(-30)}"`);
    }
  }

  return issues;
}

// ── Title-data mismatch check (Finding 19) ──────────────────────────────

function checkTitleMismatch(job: LabeledJob): string | null {
  // V7 uses seniority_reason instead of V6's role_reason
  if (!job.seniority_reason) return null;

  // With 12-field format, seniority_reason contains the seniority evidence directly
  // e.g., "Senior Software Engineer, engineering role -> LEVEL_3"
  // Try to extract what title the teacher actually saw
  const roleMatch = job.seniority_reason.match(/^['"]?([^'",\n->]+?)['"]?\s*(,|->|in title|is|found)/i);
  if (!roleMatch) return null;

  const reasoningTitle = roleMatch[1].trim().toLowerCase();
  const storedTitle = job.title.toLowerCase().trim();

  // Check if the reasoning title is a reasonable match for the stored title
  // Allow partial matches (reasoning might abbreviate)
  if (reasoningTitle.length > 5 &&
      !storedTitle.includes(reasoningTitle) &&
      !reasoningTitle.includes(storedTitle)) {
    return `Stored title "${job.title}" but seniority reasoning references "${roleMatch[1].trim()}"`;
  }

  return null;
}

// ── Pre-label: JD-embedded title mismatch check ─────────────────────────

// Matches "Senior Engineer", "Senior Software Engineer", "Lead Backend Developer", etc.
const SENIOR_TITLE_RE = /\b(senior|staff|principal|lead|head|chief|director|vp|cto|founding|snr|sr\.?)\s+(?:(?:software|backend|frontend|full[- ]?stack|data|devops|cloud|platform|ml|machine learning|site reliability)\s+)?(engineer|developer|architect|scientist|sre)\b/i;

function checkJdEmbeddedTitleMismatch(title: string, jdText: string): string | null {
  // Only flag if the stored title does NOT contain seniority keywords
  if (SENIOR_KEYWORDS.test(title)) return null;

  // Scan first 300 chars of JD for a senior engineering title
  const jdHead = jdText.slice(0, 300);
  const match = jdHead.match(SENIOR_TITLE_RE);
  if (match) {
    return `Title "${title}" has no seniority, but JD starts with "${match[0]}" — GPT may use JD title instead`;
  }
  return null;
}

// ── Main ────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input");
  const evalSetPath = getStringArg(args, "eval-set");
  const minJdLength = getNumberArg(args, "min-jd-length") ?? 100;
  const maxJdLength = getNumberArg(args, "max-jd-length") ?? 15000;
  const sourceCapPct = getNumberArg(args, "source-cap") ?? 5;
  const dryRun = getBooleanArg(args, "dry-run");
  const cleanMode = getBooleanArg(args, "clean");
  const outputPath = getStringArg(args, "output");
  const removeSuspicious = getBooleanArg(args, "remove-suspicious");
  const removeTrivial = getBooleanArg(args, "remove-trivial");
  const preLabel = getBooleanArg(args, "pre-label");

  if (!inputPath) {
    console.error("Usage: npx tsx src/cli/audit-training-data-v7.ts --input <file.jsonl> [--eval-set <eval.jsonl>] [--dry-run]");
    console.error("       npx tsx src/cli/audit-training-data-v7.ts --input <file.jsonl> --clean --output <out.jsonl>");
    process.exit(1);
  }

  if (cleanMode && !outputPath) {
    console.error("--clean requires --output <path>");
    process.exit(1);
  }

  // ── Load data ───────────────────────────────────────────────────────

  if (preLabel) {
    console.log("\n  *** PRE-LABEL MODE — skipping token/score/label checks ***\n");
  }

  console.log(`\nLoading ${inputPath}...`);
  const jobs = await readJsonlFile<LabeledJob>(inputPath);
  console.log(`Loaded ${jobs.length} jobs.\n`);

  let evalJobs: LabeledJob[] = [];
  const evalIds = new Set<string>();
  const evalKeys = new Set<string>();
  const evalFingerprints = new Set<string>();

  if (evalSetPath) {
    evalJobs = await readJsonlFile<LabeledJob>(evalSetPath);
    for (const ej of evalJobs) {
      evalIds.add(ej.job_id);
      evalKeys.add(dedupKey(ej.title, ej.company));
      evalFingerprints.add(jdFingerprint(ej.jd_text));
    }
    console.log(`Loaded ${evalJobs.length} eval jobs for contamination check.\n`);
  }

  // ── Run checks ──────────────────────────────────────────────────────

  const issues: Issue[] = [];
  const quarantineDuplicates: QuarantineEntry[] = [];
  const quarantineBadData: QuarantineEntry[] = [];
  const quarantineSuspicious: QuarantineEntry[] = [];

  // Lines to remove in clean mode (0-indexed into jobs array)
  const removeInClean = new Set<number>();     // critical + duplicates + bad data
  const removeIfSuspicious = new Set<number>(); // suspicious (only removed with --remove-suspicious)
  const removeIfTrivial = new Set<number>();    // trivially easy bad_fit (only with --remove-trivial)

  // Removal reason tracking for accurate clean output breakdown
  const removedAsCritical = new Set<number>();
  const removedAsDuplicate = new Set<number>();
  const removedAsBadData = new Set<number>();
  const removedAsTrivial = new Set<number>();

  // Contrastive pair tracking — jobs sharing source_job_id are intentional variants
  const sourceJobGroups = new Map<string, number[]>(); // source_job_id → [indices]
  for (let i = 0; i < jobs.length; i++) {
    const sjid = jobs[i].source_job_id;
    if (sjid) {
      const group = sourceJobGroups.get(sjid) || [];
      group.push(i);
      sourceJobGroups.set(sjid, group);
    }
  }
  const contrastiveIndices = new Set<number>();
  for (const [, indices] of sourceJobGroups) {
    if (indices.length > 1) {
      for (const idx of indices) contrastiveIndices.add(idx);
    }
  }

  // Dedup tracking
  const seenIds = new Map<string, number>();        // job_id → first line
  const seenKeys = new Map<string, number>();        // title+company → first line
  const seenFingerprints = new Map<string, number>(); // jd fingerprint → first line
  const sourceCounts = new Map<string, number>();     // source_file → count

  for (let i = 0; i < jobs.length; i++) {
    const job = jobs[i];
    const line = i + 1;
    const id = job.job_id || `(no id, line ${line})`;
    const title = job.title || "(no title)";

    // ── CRITICAL: Malformed / missing fields ────────────────────────

    if (!job.job_id) {
      issues.push({ line, job_id: id, title, severity: "CRITICAL", check: "missing_job_id", detail: "No job_id field" });
      quarantineBadData.push({ ...job, _quarantine_reason: "missing_job_id", _line: line });
      removeInClean.add(i); removedAsCritical.add(i); removedAsBadData.add(i);
    }

    if (!job.title || job.title.trim().length === 0) {
      issues.push({ line, job_id: id, title, severity: "CRITICAL", check: "missing_title", detail: "Empty or missing title" });
      quarantineBadData.push({ ...job, _quarantine_reason: "missing_title", _line: line });
      removeInClean.add(i); removedAsCritical.add(i); removedAsBadData.add(i);
    }

    if (!job.jd_text || job.jd_text.trim().length === 0) {
      issues.push({ line, job_id: id, title, severity: "CRITICAL", check: "missing_jd", detail: "Empty or missing jd_text" });
      quarantineBadData.push({ ...job, _quarantine_reason: "missing_jd", _line: line });
      removeInClean.add(i); removedAsCritical.add(i); removedAsBadData.add(i);
      continue; // Can't check much more without JD
    }

    // ── CRITICAL: Invalid tokens (skip in pre-label mode) ──────────

    if (!preLabel) {
      for (const field of V7_TOKEN_FIELDS) {
        const value = (job as unknown as Record<string, string>)[field];
        if (!value || !VALID_TOKENS[field].has(value)) {
          issues.push({
            line, job_id: id, title, severity: "CRITICAL",
            check: "invalid_token",
            detail: `${field}="${value}" is not a valid V7 token`,
          });
          removeInClean.add(i); removedAsCritical.add(i);
        }
      }
    }

    // ── CRITICAL: Score-token mismatch (skip in pre-label mode) ─────

    if (!preLabel && job.location && job.seniority && job.tech && job.comp &&
        VALID_TOKENS.location.has(job.location) && VALID_TOKENS.seniority.has(job.seniority) &&
        VALID_TOKENS.tech.has(job.tech) && VALID_TOKENS.comp.has(job.comp)) {
      const computed = computeScore(job);

      if (job.score !== undefined && job.score !== computed.score) {
        issues.push({
          line, job_id: id, title, severity: "CRITICAL",
          check: "score_mismatch",
          detail: `Stored score=${job.score} but tokens compute to ${computed.score}`,
        });
        removeInClean.add(i); removedAsCritical.add(i);
      }

      if (job.label && job.label !== computed.label) {
        issues.push({
          line, job_id: id, title, severity: "CRITICAL",
          check: "label_mismatch",
          detail: `Stored label="${job.label}" but score ${computed.score} -> "${computed.label}"`,
        });
        removeInClean.add(i); removedAsCritical.add(i);
      }
    }

    // ── CRITICAL: Eval contamination ────────────────────────────────

    if (evalSetPath) {
      if (evalIds.has(job.job_id)) {
        issues.push({
          line, job_id: id, title, severity: "CRITICAL",
          check: "eval_contamination_id",
          detail: `job_id matches eval set`,
        });
        removeInClean.add(i); removedAsCritical.add(i);
      }
      const key = dedupKey(job.title, job.company);
      if (evalKeys.has(key)) {
        issues.push({
          line, job_id: id, title, severity: "CRITICAL",
          check: "eval_contamination_key",
          detail: `title+company matches eval set`,
        });
        removeInClean.add(i); removedAsCritical.add(i);
      }
      const fp = jdFingerprint(job.jd_text);
      if (evalFingerprints.has(fp)) {
        issues.push({
          line, job_id: id, title, severity: "CRITICAL",
          check: "eval_contamination_jd",
          detail: `JD fingerprint matches eval set`,
        });
        removeInClean.add(i); removedAsCritical.add(i);
      }
    }

    // ── WARNING: Duplicates ─────────────────────────────────────────

    // By job_id
    if (job.job_id && seenIds.has(job.job_id)) {
      issues.push({
        line, job_id: id, title, severity: "WARNING",
        check: "duplicate_id",
        detail: `Duplicate job_id (first seen line ${seenIds.get(job.job_id)})`,
      });
      quarantineDuplicates.push({ ...job, _quarantine_reason: `duplicate_id (first: line ${seenIds.get(job.job_id)})`, _line: line });
      removeInClean.add(i); removedAsDuplicate.add(i);
    } else if (job.job_id) {
      seenIds.set(job.job_id, line);
    }

    // By title+company — but skip contrastive pairs (intentional variants)
    const key = dedupKey(job.title, job.company);
    if (seenKeys.has(key)) {
      const isContrastive = contrastiveIndices.has(i);
      if (isContrastive) {
        // Intentional variant — report but don't remove
        issues.push({
          line, job_id: id, title, severity: "WARNING",
          check: "contrastive_pair",
          detail: `Same title+company as line ${seenKeys.get(key)} but is a contrastive variant (source_job_id=${job.source_job_id}) — KEPT`,
        });
      } else {
        issues.push({
          line, job_id: id, title, severity: "WARNING",
          check: "duplicate_title_company",
          detail: `Duplicate title+company (first seen line ${seenKeys.get(key)})`,
        });
        // Only quarantine if not already caught by ID duplicate
        if (!seenIds.has(job.job_id) || seenIds.get(job.job_id) === line) {
          quarantineDuplicates.push({ ...job, _quarantine_reason: `duplicate_title_company (first: line ${seenKeys.get(key)})`, _line: line });
        }
        removeInClean.add(i); removedAsDuplicate.add(i);
      }
    } else {
      seenKeys.set(key, line);
    }

    // By JD fingerprint
    const fp = jdFingerprint(job.jd_text);
    if (seenFingerprints.has(fp)) {
      issues.push({
        line, job_id: id, title, severity: "WARNING",
        check: "duplicate_jd_fingerprint",
        detail: `JD text fingerprint matches line ${seenFingerprints.get(fp)} (cross-posted?)`,
      });
    } else {
      seenFingerprints.set(fp, line);
    }

    // ── WARNING: Short / long JD ────────────────────────────────────

    const jdLen = job.jd_text.trim().length;
    if (jdLen < minJdLength) {
      issues.push({
        line, job_id: id, title, severity: "WARNING",
        check: "short_jd",
        detail: `JD is only ${jdLen} chars (threshold: ${minJdLength})`,
      });
      quarantineBadData.push({ ...job, _quarantine_reason: `short_jd (${jdLen} chars)`, _line: line });
      removeInClean.add(i); removedAsBadData.add(i);
    }
    if (jdLen > maxJdLength) {
      issues.push({
        line, job_id: id, title, severity: "WARNING",
        check: "long_jd",
        detail: `JD is ${jdLen} chars (threshold: ${maxJdLength}) — may be truncated during tokenization`,
      });
    }

    // ── WARNING: HTML/encoding artifacts ────────────────────────────

    if (hasHtmlArtifacts(job.jd_text)) {
      issues.push({
        line, job_id: id, title, severity: "WARNING",
        check: "html_artifacts",
        detail: `JD contains HTML tags or entities`,
      });
    }

    // ── WARNING: Title-data mismatch (Finding 19) — post-label only ─

    if (!preLabel) {
      const mismatch = checkTitleMismatch(job);
      if (mismatch) {
        issues.push({
          line, job_id: id, title, severity: "WARNING",
          check: "title_mismatch",
          detail: mismatch,
        });
        quarantineSuspicious.push({ ...job, _quarantine_reason: mismatch, _line: line });
        removeIfSuspicious.add(i);
      }
    }

    // ── WARNING: JD-embedded title mismatch — pre-label only ────────

    if (preLabel && job.jd_text && job.title) {
      const jdMismatch = checkJdEmbeddedTitleMismatch(job.title, job.jd_text);
      if (jdMismatch) {
        issues.push({
          line, job_id: id, title, severity: "WARNING",
          check: "jd_title_mismatch",
          detail: jdMismatch,
        });
        quarantineSuspicious.push({ ...job, _quarantine_reason: jdMismatch, _line: line });
        removeIfSuspicious.add(i);
      }
    }

    // ── WARNING: Suspicious token combinations (post-label only) ────

    if (!preLabel) {
      const suspicious = checkSuspiciousTokens(job);
      for (const s of suspicious) {
        issues.push({
          line, job_id: id, title, severity: "WARNING",
          check: "suspicious_token",
          detail: s,
        });
        quarantineSuspicious.push({ ...job, _quarantine_reason: s, _line: line });
        removeIfSuspicious.add(i);
      }
    }

    // ── WARNING: Reasoning-token mismatch (post-label only) ────────

    if (!preLabel) {
      const hasAnyReason = V7_REASON_FIELDS.some(
        (f) => (job as unknown as Record<string, string>)[f],
      );
      if (hasAnyReason) {
        const reasoningIssues = checkReasoningTokenMismatch(job);
        for (const ri of reasoningIssues) {
          issues.push({
            line, job_id: id, title, severity: "WARNING",
            check: "reasoning_token_mismatch",
            detail: ri,
          });
          quarantineSuspicious.push({ ...job, _quarantine_reason: ri, _line: line });
          removeIfSuspicious.add(i);
        }
      }
    }

    // ── WARNING: Trivially easy bad_fit (post-label only) ──────────

    if (!preLabel) {
      // V7 trivial bad_fit: outside UK or unknown location + out of scope + no tech + no GBP comp
      const isTrivialBadFit =
        (job.location === "OUTSIDE_UK" || job.location === "UNKNOWN") &&
        job.scope === "OUT_OF_SCOPE" &&
        job.tech === "NONE" &&
        job.comp === "NO_GBP";

      if (isTrivialBadFit) {
        issues.push({
          line, job_id: id, title, severity: "WARNING",
          check: "trivial_bad_fit",
          detail: `All tokens at baseline (${job.location} + OUT_OF_SCOPE + NONE + NO_GBP) — teaches model nothing new`,
        });
        removeIfTrivial.add(i); removedAsTrivial.add(i);
      }
    }

    // ── Track source distribution ───────────────────────────────────

    const source = job.source_file || "(unknown)";
    sourceCounts.set(source, (sourceCounts.get(source) || 0) + 1);
  }

  // ── WARNING: Source-weight cap ──────────────────────────────────────

  const totalJobs = jobs.length;
  for (const [source, count] of sourceCounts) {
    const pct = (count / totalJobs) * 100;
    if (pct > sourceCapPct && source !== "(unknown)") {
      issues.push({
        line: 0, job_id: "-", title: "-", severity: "WARNING",
        check: "source_weight_cap",
        detail: `Source "${source}" has ${count} jobs (${pct.toFixed(1)}%) — exceeds ${sourceCapPct}% cap`,
      });
    }
  }

  // ── WARNING: Distribution balance (post-label only) ────────────────

  if (!preLabel) {
    for (const field of V7_TOKEN_FIELDS) {
      const dist = new Map<string, number>();
      for (const job of jobs) {
        const val = (job as unknown as Record<string, string>)[field] || "(missing)";
        dist.set(val, (dist.get(val) || 0) + 1);
      }
      for (const [token, count] of dist) {
        const pct = (count / totalJobs) * 100;
        if (pct > 50) {
          issues.push({
            line: 0, job_id: "-", title: "-", severity: "WARNING",
            check: "distribution_imbalance",
            detail: `${field}="${token}" is ${pct.toFixed(1)}% of all jobs (>${50}% threshold)`,
          });
        }
      }
    }

    // Boundary zone check: score 50-74 should be >= 15% of total
    let boundaryCount = 0;
    for (const job of jobs) {
      if (job.score !== undefined && job.score >= 50 && job.score <= 74) {
        boundaryCount++;
      }
    }
    const boundaryPct = (boundaryCount / totalJobs) * 100;
    if (boundaryPct < 15) {
      issues.push({
        line: 0, job_id: "-", title: "-", severity: "WARNING",
        check: "boundary_starvation",
        detail: `Boundary zone (score 50-74) has only ${boundaryCount} jobs (${boundaryPct.toFixed(1)}%) — below 15% threshold`,
      });
    }
  }

  // ── WARNING: Label consistency (post-label only) ──────────────────

  if (!preLabel) {
    // V7: check seniority consistency by title (was "role" in V6)
    const titleToSeniority = new Map<string, Set<string>>();
    for (const job of jobs) {
      const normTitle = job.title.toLowerCase().trim();
      if (!titleToSeniority.has(normTitle)) titleToSeniority.set(normTitle, new Set());
      if (job.seniority) titleToSeniority.get(normTitle)!.add(job.seniority);
    }
    for (const [normTitle, levels] of titleToSeniority) {
      if (levels.size > 1) {
        issues.push({
          line: 0, job_id: "-", title: normTitle, severity: "WARNING",
          check: "label_inconsistency",
          detail: `Title "${normTitle}" has ${levels.size} different seniority tokens: ${[...levels].join(", ")}`,
        });
      }
    }

    // V7-specific: check scope consistency by title
    const titleToScope = new Map<string, Set<string>>();
    for (const job of jobs) {
      const normTitle = job.title.toLowerCase().trim();
      if (!titleToScope.has(normTitle)) titleToScope.set(normTitle, new Set());
      if (job.scope) titleToScope.get(normTitle)!.add(job.scope);
    }
    for (const [normTitle, scopes] of titleToScope) {
      if (scopes.size > 1) {
        issues.push({
          line: 0, job_id: "-", title: normTitle, severity: "WARNING",
          check: "label_inconsistency",
          detail: `Title "${normTitle}" has ${scopes.size} different scope tokens: ${[...scopes].join(", ")}`,
        });
      }
    }
  }

  // ── Report ──────────────────────────────────────────────────────────

  const critical = issues.filter((i) => i.severity === "CRITICAL");
  const warnings = issues.filter((i) => i.severity === "WARNING");

  console.log("═══════════════════════════════════════════════════════════");
  console.log(`  V7 TRAINING DATA AUDIT REPORT${preLabel ? " (PRE-LABEL)" : ""}`);
  console.log("═══════════════════════════════════════════════════════════");
  console.log(`  Input:       ${inputPath}`);
  console.log(`  Total jobs:  ${totalJobs}`);
  if (evalSetPath) console.log(`  Eval set:    ${evalSetPath} (${evalJobs.length} jobs)`);
  console.log(`  Thresholds:  min JD=${minJdLength} chars, max JD=${maxJdLength} chars, source cap=${sourceCapPct}%`);
  console.log("───────────────────────────────────────────────────────────");

  // Summary by check type
  const checkCounts = new Map<string, { critical: number; warning: number }>();
  for (const issue of issues) {
    const entry = checkCounts.get(issue.check) || { critical: 0, warning: 0 };
    if (issue.severity === "CRITICAL") entry.critical++;
    else entry.warning++;
    checkCounts.set(issue.check, entry);
  }

  console.log("\n  CHECK SUMMARY:");
  console.log("  ─────────────────────────────────────────────────────────");
  for (const [check, counts] of [...checkCounts.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const tag = counts.critical > 0 ? "CRITICAL" : "WARNING ";
    const count = counts.critical + counts.warning;
    console.log(`  [${tag}] ${check}: ${count}`);
  }

  // Distribution summary (skip in pre-label mode — no tokens yet)
  if (!preLabel) {
    console.log("\n  TOKEN DISTRIBUTION:");
    console.log("  ─────────────────────────────────────────────────────────");
    for (const field of V7_TOKEN_FIELDS) {
      const dist = new Map<string, number>();
      for (const job of jobs) {
        const val = (job as unknown as Record<string, string>)[field] || "(missing)";
        dist.set(val, (dist.get(val) || 0) + 1);
      }
      console.log(`  ${field}:`);
      for (const [val, count] of [...dist.entries()].sort((a, b) => b[1] - a[1])) {
        const pct = ((count / totalJobs) * 100).toFixed(1);
        const bar = "█".repeat(Math.round((count / totalJobs) * 40));
        console.log(`    ${val.padEnd(20)} ${String(count).padStart(4)}  (${pct.padStart(5)}%)  ${bar}`);
      }
    }

    // Label distribution
    const labelDist = new Map<string, number>();
    for (const job of jobs) {
      const val = job.label || "(missing)";
      labelDist.set(val, (labelDist.get(val) || 0) + 1);
    }
    console.log(`  label:`);
    for (const [val, count] of [...labelDist.entries()].sort((a, b) => b[1] - a[1])) {
      const pct = ((count / totalJobs) * 100).toFixed(1);
      const bar = "█".repeat(Math.round((count / totalJobs) * 40));
      console.log(`    ${val.padEnd(20)} ${String(count).padStart(4)}  (${pct.padStart(5)}%)  ${bar}`);
    }
  }

  // Source distribution
  if (sourceCounts.size > 1) {
    console.log("\n  SOURCE DISTRIBUTION:");
    console.log("  ─────────────────────────────────────────────────────────");
    for (const [source, count] of [...sourceCounts.entries()].sort((a, b) => b[1] - a[1])) {
      const pct = ((count / totalJobs) * 100).toFixed(1);
      const cap = parseFloat(pct) > sourceCapPct ? " ⚠ OVER CAP" : "";
      console.log(`    ${source.padEnd(40)} ${String(count).padStart(4)}  (${pct.padStart(5)}%)${cap}`);
    }
  }

  // Critical details
  if (critical.length > 0) {
    console.log("\n  CRITICAL ISSUES (must fix before training):");
    console.log("  ─────────────────────────────────────────────────────────");
    for (const issue of critical.slice(0, 50)) {
      console.log(`  Line ${String(issue.line).padStart(4)}: [${issue.check}] ${issue.detail}`);
      console.log(`          job_id=${issue.job_id}  title="${issue.title}"`);
    }
    if (critical.length > 50) {
      console.log(`  ... and ${critical.length - 50} more critical issues`);
    }
  }

  // Warning details (top 30)
  if (warnings.length > 0) {
    console.log("\n  WARNINGS (review recommended):");
    console.log("  ─────────────────────────────────────────────────────────");
    for (const issue of warnings.slice(0, 30)) {
      console.log(`  Line ${String(issue.line).padStart(4)}: [${issue.check}] ${issue.detail}`);
    }
    if (warnings.length > 30) {
      console.log(`  ... and ${warnings.length - 30} more warnings`);
    }
  }

  // Final verdict
  console.log("\n═══════════════════════════════════════════════════════════");
  if (critical.length > 0) {
    console.log(`  ❌ FAILED: ${critical.length} critical issues, ${warnings.length} warnings`);
    console.log("  Fix critical issues before training.");
  } else if (warnings.length > 0) {
    console.log(`  ⚠  PASSED with ${warnings.length} warnings`);
    console.log("  Review quarantined items, then proceed.");
  } else {
    console.log("  ✅ CLEAN: No issues found.");
  }
  console.log("═══════════════════════════════════════════════════════════\n");

  // ── Write quarantine files ──────────────────────────────────────────

  if (!dryRun) {
    const outDir = path.join(path.dirname(inputPath), "quarantine");
    if (!fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }

    if (quarantineDuplicates.length > 0) {
      const outPath = path.join(outDir, "duplicates.jsonl");
      await writeJsonlFile(outPath, quarantineDuplicates);
      console.log(`Wrote ${quarantineDuplicates.length} duplicates -> ${outPath}`);
    }

    if (quarantineBadData.length > 0) {
      const outPath = path.join(outDir, "bad_data.jsonl");
      await writeJsonlFile(outPath, quarantineBadData);
      console.log(`Wrote ${quarantineBadData.length} bad data items -> ${outPath}`);
    }

    if (quarantineSuspicious.length > 0) {
      const outPath = path.join(outDir, "suspicious.jsonl");
      await writeJsonlFile(outPath, quarantineSuspicious);
      console.log(`Wrote ${quarantineSuspicious.length} suspicious items -> ${outPath}`);
    }

    if (quarantineDuplicates.length + quarantineBadData.length + quarantineSuspicious.length === 0) {
      console.log("No items to quarantine.");
    }
  } else {
    console.log("(Dry run — no quarantine files written)");
  }

  // ── Clean mode: write filtered output ────────────────────────────────

  if (cleanMode && outputPath) {
    const toRemove = new Set(removeInClean);
    if (removeSuspicious) {
      for (const idx of removeIfSuspicious) toRemove.add(idx);
    }
    if (removeTrivial) {
      for (const idx of removeIfTrivial) toRemove.add(idx);
    }

    const cleanJobs: LabeledJob[] = [];
    for (let i = 0; i < jobs.length; i++) {
      if (!toRemove.has(i)) cleanJobs.push(jobs[i]);
    }

    await writeJsonlFile(outputPath, cleanJobs);

    const removed = jobs.length - cleanJobs.length;

    // Count unique removals per reason (a job may have multiple reasons)
    const criticalOnly = new Set([...removedAsCritical].filter((x) => !removedAsDuplicate.has(x) && !removedAsBadData.has(x)));
    const dupOnly = new Set([...removedAsDuplicate].filter((x) => !removedAsCritical.has(x)));

    console.log("\n───────────────────────────────────────────────────────────");
    console.log(`  CLEAN OUTPUT: ${outputPath}`);
    console.log(`  Kept:    ${cleanJobs.length} jobs`);
    console.log(`  Removed: ${removed} jobs (breakdown below, jobs may overlap):`);
    console.log(`    - Eval contamination:  ${[...removedAsCritical].filter((x) => !removedAsBadData.has(x)).length}`);
    console.log(`    - Bad data:            ${removedAsBadData.size} (missing fields, short JDs)`);
    console.log(`    - Duplicates:          ${removedAsDuplicate.size} (kept first occurrence)`);
    if (removeSuspicious) {
      console.log(`    - Suspicious:          ${removeIfSuspicious.size} removed (--remove-suspicious)`);
    } else {
      console.log(`    - Suspicious:          ${removeIfSuspicious.size} kept (use --remove-suspicious to remove)`);
    }
    if (removeTrivial) {
      console.log(`    - Trivial bad_fit:     ${removedAsTrivial.size} removed (--remove-trivial)`);
    } else {
      console.log(`    - Trivial bad_fit:     ${removeIfTrivial.size} kept (use --remove-trivial to remove)`);
    }
    console.log(`    Contrastive pairs:     ${contrastiveIndices.size} protected from dedup`);
    console.log("───────────────────────────────────────────────────────────\n");
  }

  // Exit with error if critical issues found (but still writes clean file first)
  if (critical.length > 0 && !cleanMode) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(2);
});

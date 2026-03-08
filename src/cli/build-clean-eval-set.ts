/**
 * Build a clean (non-contaminated) eval dataset by:
 * 1. Collecting all training job titles across v1/v2/v2b
 * 2. Reading all eval-candidate JSONL files
 * 3. Deduplicating across files
 * 4. Filtering out any job that appeared in training
 * 5. Re-scoring with the human-corrected deterministic scorer
 * 6. Writing clean_eval.jsonl
 *
 * Usage:
 *   npx tsx src/cli/build-clean-eval-set.ts
 *   npx tsx src/cli/build-clean-eval-set.ts --output data/clean_eval.jsonl
 */

import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { parseArgs, getStringArg } from "../lib/args.js";
import { scoreJob } from "../lib/deterministic-scorer-human-corrected.js";
import type { JobExportRecord } from "../schema.js";

// ── Training data files (the contamination sources) ──────────────
const TRAINING_FILES = [
  "data/finetune/train.jsonl",
  "data/finetune/train_v2.jsonl",
  "data/finetune/train_v2b.jsonl",
];

// MLX chat-format files (title is embedded in the user prompt)
const MLX_TRAINING_FILES = [
  "data/finetune/mlx/train.jsonl",
  "data/finetune/mlx_v2/train.jsonl",
  "data/finetune/mlx_v2b/train.jsonl",
];

// ── Eval candidate files (we pick clean jobs from these) ─────────
const EVAL_SOURCE_FILES = [
  "data/finetune/test.jsonl",
  "data/new_uk_jobs_golden.jsonl",
  "data/linkedin_teacher_v2_eval_human_corrected.jsonl",
];

// ── Helpers ──────────────────────────────────────────────────────

function normalizeText(value: string | undefined): string {
  if (!value) return "";
  return value.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

/**
 * Dedup key: normalized title + company.
 *
 * - Lowercase, strip common suffixes (Ltd, LLC, Inc, etc.), collapse whitespace
 * - title+company is the sweet spot: unique enough to avoid false matches
 *   ("Software Engineer" at Google ≠ at Meta) without being brittle to
 *   formatting differences in location or job_id across datasets.
 */
function normalizeCompany(raw: string): string {
  return raw
    .toLowerCase()
    .replace(/\b(ltd|llc|inc|plc|limited|corporation|corp|group|gmbh)\.?\b/g, "")
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function jobDedupeKey(job: {
  title: string;
  company: string;
  location: string;
  job_id: string;
}): string {
  return `${job.title.toLowerCase().trim()}||${normalizeCompany(job.company)}`;
}

function extractTitle(content: string): string | undefined {
  const match = content.match(/Job title:\s*(.+)/);
  return match?.[1]?.trim();
}

function buildReasoning(breakdown: { loc: number; role: number; tech: number; comp: number }): string {
  const parts: string[] = [];

  if (breakdown.loc === 25) parts.push("London/Remote UK (+25)");
  else if (breakdown.loc === 10) parts.push("UK outside London (+10)");
  else if (breakdown.loc === -50) parts.push("Outside UK (-50)");
  else parts.push("Location unclear (0)");

  if (breakdown.role === 25) parts.push("senior-level role (+25)");
  else if (breakdown.role === 15) parts.push("mid-level role (+15)");
  else parts.push("no seniority keyword (0)");

  parts.push(`tech stack (${breakdown.tech})`);

  if (breakdown.comp === 25) parts.push("salary ≥£100k (+25)");
  else if (breakdown.comp === 15) parts.push("salary £75-99k (+15)");
  else if (breakdown.comp === 5) parts.push("salary £55-74k (+5)");
  else if (breakdown.comp === -30) parts.push("salary <£45k (-30)");
  else parts.push("no GBP salary (0)");

  return parts.join(", ") + ".";
}

// ── Main ─────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const outputPath = getStringArg(args, "output") ?? "data/clean_eval.jsonl";

  // Step 1: Build contamination set from all training data
  console.log("Building contamination set from training data...");
  const trainTitles = new Set<string>();

  for (const file of TRAINING_FILES) {
    try {
      const rows = await readJsonlFile<JobExportRecord>(file);
      for (const row of rows) {
        const title = normalizeText(row.title ?? row.job_title as string);
        if (title) trainTitles.add(title.toLowerCase());
      }
    } catch {
      console.warn(`  Skipping ${file} (not found)`);
    }
  }

  for (const file of MLX_TRAINING_FILES) {
    try {
      const rows = await readJsonlFile<{ messages: Array<{ role: string; content: string }> }>(file);
      for (const row of rows) {
        const userMsg = row.messages.find((m) => m.role === "user");
        if (userMsg) {
          const title = extractTitle(userMsg.content);
          if (title) trainTitles.add(title.toLowerCase());
        }
      }
    } catch {
      console.warn(`  Skipping ${file} (not found)`);
    }
  }

  console.log(`  Contamination set: ${trainTitles.size} unique training titles\n`);

  // Step 2: Read eval candidates, deduplicate, filter contaminated
  const seen = new Set<string>();
  const cleanJobs: Array<{
    job_id: string;
    title: string;
    company: string;
    location: string;
    jd_text: string;
    source_file: string;
    source_url?: string;
  }> = [];

  let totalRead = 0;
  let duplicates = 0;
  let contaminated = 0;

  for (const file of EVAL_SOURCE_FILES) {
    const shortName = file.split("/").pop()!;
    try {
      const rows = await readJsonlFile<JobExportRecord>(file);
      let fileClean = 0;
      let fileContam = 0;
      let fileDup = 0;

      for (const row of rows) {
        totalRead++;
        const title = normalizeText(row.title ?? row.job_title as string);
        const company = normalizeText(row.company ?? row.company_name);
        const location = normalizeText(row.location);
        const jdText = normalizeText(row.jd_text ?? row.description);
        const jobId = String(row.job_id ?? row.id ?? "");

        if (!title || !jdText) continue;

        // Contamination check
        if (trainTitles.has(title.toLowerCase())) {
          contaminated++;
          fileContam++;
          continue;
        }

        // Deduplication
        const key = jobDedupeKey({ title, company, location, job_id: jobId });
        if (seen.has(key)) {
          duplicates++;
          fileDup++;
          continue;
        }
        seen.add(key);

        cleanJobs.push({
          job_id: jobId,
          title,
          company,
          location,
          jd_text: jdText,
          source_file: shortName,
          source_url: (row.source_url ?? row.url) as string | undefined,
        });
        fileClean++;
      }

      console.log(`  ${shortName}: ${rows.length} total → ${fileClean} clean, ${fileContam} contaminated, ${fileDup} duplicate`);
    } catch (error) {
      console.error(`  Failed to read ${file}: ${error instanceof Error ? error.message : error}`);
    }
  }

  console.log(`\n  Summary: ${totalRead} read → ${contaminated} contaminated, ${duplicates} duplicate, ${cleanJobs.length} clean\n`);

  // Step 3: Re-score every clean job with deterministic scorer
  console.log("Scoring clean jobs with deterministic scorer...");

  const scoredJobs = cleanJobs.map((job) => {
    const breakdown = scoreJob(job.title, job.location, job.jd_text);
    const reasoning = buildReasoning(breakdown);

    return {
      job_id: job.job_id,
      title: job.title,
      company: job.company,
      location: job.location,
      jd_text: job.jd_text,
      label: breakdown.label,
      score: breakdown.score,
      reasoning,
      loc: breakdown.loc,
      role: breakdown.role,
      tech: breakdown.tech,
      comp: breakdown.comp,
      source_file: job.source_file,
      ...(job.source_url ? { source_url: job.source_url } : {}),
    };
  });

  // Step 4: Print label distribution
  const labelCounts = { good_fit: 0, maybe: 0, bad_fit: 0 };
  for (const job of scoredJobs) {
    labelCounts[job.label]++;
  }
  console.log(`\n  Label distribution:`);
  console.log(`    good_fit: ${labelCounts.good_fit}`);
  console.log(`    maybe:    ${labelCounts.maybe}`);
  console.log(`    bad_fit:  ${labelCounts.bad_fit}`);

  // Step 5: Write output
  await writeJsonlFile(outputPath, scoredJobs);
  console.log(`\n  Wrote ${scoredJobs.length} clean scored jobs to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

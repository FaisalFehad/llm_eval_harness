import { createHash } from "node:crypto";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { scoreJob } from "../lib/deterministic-scorer.js";
import type { FitLabel } from "../schema.js";

type RawJob = {
  id?: string | number;
  job_id?: string | number;
  title?: string;
  company?: string;
  company_name?: string;
  location?: string;
  jd_text?: string;
  description?: string;
  source_url?: string;
  url?: string;
};

type ScoredJob = {
  job_id: string;
  title: string;
  company: string;
  location?: string;
  jd_text: string;
  label: FitLabel;
  score: number;
  reasoning: string;
  loc: number;
  role: number;
  tech: number;
  comp: number;
  source_url?: string;
};

function normalizeText(value: string | undefined): string {
  if (!value) return "";
  return value.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

function buildReasoning(
  breakdown: ReturnType<typeof scoreJob>,
  location: string | undefined,
): string {
  const parts: string[] = [];

  if (breakdown.loc === 25) parts.push("London/Remote UK (+25)");
  else if (breakdown.loc === 10) parts.push("UK outside London (+10)");
  else if (breakdown.loc === -50) parts.push("Outside UK (-50)");
  else if (!location || location.trim().length === 0)
    parts.push("Location unknown (0)");
  else parts.push("Location unclear (0)");

  if (breakdown.role === 25) parts.push("senior-level role (+25)");
  else if (breakdown.role === 15) parts.push("mid-level role (+15)");
  else parts.push("no seniority keyword (0)");

  if (breakdown.tech > 0) parts.push(`tech stack (${breakdown.tech})`);
  else parts.push("no matching tech (0)");

  if (breakdown.comp === 25) parts.push("salary ≥£100k (+25)");
  else if (breakdown.comp === 15) parts.push("salary £75-99k (+15)");
  else if (breakdown.comp === 5) parts.push("salary £55-74k (+5)");
  else if (breakdown.comp === -30) parts.push("salary <£45k (-30)");
  else parts.push("no GBP salary (0)");

  const reasoning = parts.join(", ") + ".";
  return reasoning.length > 600 ? reasoning.slice(0, 600) : reasoning;
}

function extractJobId(
  rawId: string | number | undefined,
  sourceUrl: string,
  title: string,
  company: string,
  location: string,
): string {
  if (rawId !== undefined && String(rawId).trim().length > 0) {
    return String(rawId).trim();
  }

  if (sourceUrl) {
    const matches = sourceUrl.match(/\b\d{6,}\b/g);
    if (matches && matches.length > 0) {
      return matches[matches.length - 1]!;
    }
  }

  const basis = `${title}|${company}|${location}`;
  const hash = createHash("sha1").update(basis).digest("hex").slice(0, 12);
  return `ukjob_${hash}`;
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ?? "data/new_uk_jobs.jsonl";
  const outputPath =
    getStringArg(args, "output") ?? "data/new_uk_jobs_golden.jsonl";
  const minLength = getNumberArg(args, "min-length") ?? 50;

  const rows = await readJsonlFile<RawJob>(inputPath);
  const scored: ScoredJob[] = [];

  let emptyCount = 0;
  let shortCount = 0;
  let missingIdCount = 0;
  let missingTitleCount = 0;
  let missingCompanyCount = 0;

  for (const row of rows) {
    const title = normalizeText(row.title);
    const company = normalizeText(row.company ?? row.company_name);
    const location = normalizeText(row.location);
    const jdText = normalizeText(row.jd_text ?? row.description);

    if (!jdText) {
      emptyCount++;
      continue;
    }
    if (jdText.length < minLength) {
      shortCount++;
      continue;
    }

    if (!title) {
      missingTitleCount++;
      continue;
    }
    if (!company) {
      missingCompanyCount++;
      continue;
    }

    const sourceUrl = normalizeText(row.source_url ?? row.url);
    const jobId = extractJobId(
      row.job_id ?? row.id,
      sourceUrl,
      title,
      company,
      location,
    );
    if (!jobId) {
      missingIdCount++;
      continue;
    }

    const breakdown = scoreJob(title, location || undefined, jdText);
    const reasoning = buildReasoning(breakdown, location || undefined);

    scored.push({
      job_id: jobId,
      title,
      company,
      location: location || undefined,
      jd_text: jdText,
      label: breakdown.label,
      score: breakdown.score,
      reasoning,
      loc: breakdown.loc,
      role: breakdown.role,
      tech: breakdown.tech,
      comp: breakdown.comp,
      source_url: sourceUrl || undefined,
    });
  }

  await writeJsonlFile(outputPath, scored);

  console.log(`Loaded ${rows.length} jobs from ${inputPath}`);
  console.log(
    `Filtered: empty=${emptyCount}, short(<${minLength})=${shortCount}, missing_id=${missingIdCount}, missing_title=${missingTitleCount}, missing_company=${missingCompanyCount}`,
  );
  console.log(`Wrote ${scored.length} scored jobs to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

import { readFile } from "node:fs/promises";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import type { GoldenJob, JobExportRecord } from "../schema.js";

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickString(
  record: JobExportRecord,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim().length > 0) {
      return value.trim();
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value);
    }
  }

  return undefined;
}

function normalizeRecord(
  record: JobExportRecord,
  index: number,
): GoldenJob | null {
  const jobId =
    pickString(record, ["job_id", "id", "external_id"]) ??
    `job-${String(index + 1)}`;
  const title = pickString(record, ["title", "job_title"]);
  const company = pickString(record, [
    "company",
    "company_name",
    "employer_name",
  ]);
  const jdText = pickString(record, [
    "jd_text",
    "description",
    "job_description",
  ]);
  const location = pickString(record, ["location", "city"]);

  if (!title || !company || !jdText) {
    return null;
  }

  const normalized: GoldenJob = {
    job_id: jobId,
    title,
    company,
    jd_text: jdText,
    label: "maybe",
    score: 50,
    reasoning:
      "TODO: Replace with 1-2 sentence rationale after manual review of this role.",
  };

  if (location) {
    normalized.location = location;
  }

  return normalized;
}

function sampleInPlace<T>(items: T[], count: number, seed: number): T[] {
  const rng = mulberry32(seed);
  const copy = [...items];

  for (let i = copy.length - 1; i > 0; i -= 1) {
    const swapIndex = Math.floor(rng() * (i + 1));
    const left = copy[i];
    const right = copy[swapIndex];
    if (left === undefined || right === undefined) {
      continue;
    }
    copy[i] = right;
    copy[swapIndex] = left;
  }

  return copy.slice(0, Math.min(count, copy.length));
}

async function loadExportRecords(filePath: string): Promise<JobExportRecord[]> {
  if (filePath.endsWith(".jsonl")) {
    return readJsonlFile<JobExportRecord>(filePath);
  }

  const raw = await readFile(filePath, "utf8");
  const parsed = JSON.parse(raw);

  if (Array.isArray(parsed)) {
    return parsed as JobExportRecord[];
  }

  if (
    typeof parsed === "object" &&
    parsed !== null &&
    "jobs" in parsed &&
    Array.isArray((parsed as { jobs: unknown[] }).jobs)
  ) {
    return (parsed as { jobs: JobExportRecord[] }).jobs;
  }

  throw new Error(
    `Unsupported export shape in ${filePath}. Expected an array or an object with a jobs array.`,
  );
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input");
  const outputPath = getStringArg(args, "output") ?? "data/golden_jobs.jsonl";
  const count = getNumberArg(args, "count") ?? 100;
  const seed = getNumberArg(args, "seed") ?? 42;

  if (!inputPath) {
    console.error(
      "Missing required --input path. Example: npm run golden:sample -- --input data/jobs_export.jsonl",
    );
    process.exit(1);
  }

  const records = await loadExportRecords(inputPath);
  const normalized = records
    .map((record, index) => normalizeRecord(record, index))
    .filter((record): record is GoldenJob => record !== null);

  if (normalized.length === 0) {
    console.error(
      "No usable records found in export. Ensure each row has title, company, and jd_text/description.",
    );
    process.exit(1);
  }

  const sampled = sampleInPlace(normalized, count, seed);
  await writeJsonlFile(outputPath, sampled);

  console.log(
    `Wrote ${sampled.length} candidate rows to ${outputPath}. Now manually label label/score/reasoning fields.`,
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

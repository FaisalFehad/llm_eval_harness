/**
 * Re-score eval data using the updated deterministic scorer.
 * Reads a JSONL file, re-computes loc/role/tech/comp/score/label for each job,
 * and writes back to the same file (or --out for a different path).
 *
 * Usage: npx tsx src/cli/rescore-eval-v9.4.ts [--file <path>] [--out <path>] [--dry-run]
 */
import { readFileSync, writeFileSync } from "fs";
import { scoreJob } from "../lib/deterministic-scorer-human-corrected.js";

const args = process.argv.slice(2);
const fileIdx = args.indexOf("--file");
const outIdx = args.indexOf("--out");
const dryRun = args.includes("--dry-run");

const inputFile = fileIdx >= 0 ? args[fileIdx + 1]! : "data/linkedin_teacher_v2_eval_human_corrected.jsonl";
const outputFile = outIdx >= 0 ? args[outIdx + 1]! : inputFile;

const lines = readFileSync(inputFile, "utf-8").trim().split("\n");
const updated: string[] = [];
let changed = 0;

for (let i = 0; i < lines.length; i++) {
  const job = JSON.parse(lines[i]!);
  const result = scoreJob(job.title, job.location, job.jd_text);

  const diffs: string[] = [];
  for (const k of ["loc", "role", "tech", "comp"] as const) {
    if (result[k] !== job[k]) {
      diffs.push(`${k}: ${job[k]}→${result[k]}`);
    }
  }
  if (result.score !== job.score) diffs.push(`score: ${job.score}→${result.score}`);
  if (result.label !== job.label) diffs.push(`label: ${job.label}→${result.label}`);

  if (diffs.length > 0) {
    changed++;
    console.log(`[${i + 1}] ${job.title.substring(0, 55)}`);
    console.log(`  ${diffs.join(", ")}`);

    job.loc = result.loc;
    job.role = result.role;
    job.tech = result.tech;
    job.comp = result.comp;
    job.score = result.score;
    job.label = result.label;
  }

  updated.push(JSON.stringify(job));
}

console.log(`\n${changed} of ${lines.length} jobs changed.`);

if (!dryRun && changed > 0) {
  writeFileSync(outputFile, updated.join("\n") + "\n");
  console.log(`Written to ${outputFile}`);
} else if (dryRun) {
  console.log("(dry run — no files written)");
}

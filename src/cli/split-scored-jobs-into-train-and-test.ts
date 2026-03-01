import { writeFile } from "node:fs/promises";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { ensureParentDir } from "../lib/paths.js";
import { stratifiedSample } from "../lib/sampling.js";
import { GoldenJobSchema, type GoldenJob } from "../schema.js";

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ?? "data/golden_jobs_scored.jsonl";
  const trainOutput =
    getStringArg(args, "train-output") ?? "data/finetune/train.jsonl";
  const testOutput =
    getStringArg(args, "test-output") ?? "data/finetune/test.jsonl";
  const trainCount = getNumberArg(args, "train-count") ?? 70;
  const seed = getNumberArg(args, "seed") ?? 42;

  // Read all scored jobs (these have loc/role/tech/comp breakdowns)
  const rows = await readJsonlFile<Record<string, unknown>>(inputPath);

  // Validate with golden schema (extra fields like loc/role/tech/comp pass through)
  const jobs: Array<GoldenJob & Record<string, unknown>> = [];
  const errors: string[] = [];

  rows.forEach((row, index) => {
    const parsed = GoldenJobSchema.safeParse(row);
    if (!parsed.success) {
      const issue = parsed.error.issues
        .map((entry) => `${entry.path.join(".") || "root"}: ${entry.message}`)
        .join("; ");
      errors.push(`line ${index + 1}: ${issue}`);
      return;
    }
    // Keep all fields (including breakdowns) from original row
    jobs.push(row as GoldenJob & Record<string, unknown>);
  });

  if (errors.length > 0) {
    console.error(`Validation errors in ${inputPath}:`);
    errors.forEach((e) => console.error(`  - ${e}`));
    process.exit(1);
  }

  const total = jobs.length;
  const testCount = total - trainCount;

  if (testCount <= 0) {
    console.error(
      `Train count (${trainCount}) must be less than total jobs (${total}).`,
    );
    process.exit(1);
  }

  console.log(
    `Splitting ${total} jobs: ${trainCount} train / ${testCount} test (seed=${seed})\n`,
  );

  // Use stratifiedSample to pick test set, remainder is train
  const testJobs = stratifiedSample(
    jobs as GoldenJob[],
    testCount,
    seed,
  );
  const testIds = new Set(testJobs.map((j) => j.job_id));
  const trainJobs = jobs.filter((j) => !testIds.has(j.job_id));

  // Print distribution
  const printDist = (label: string, set: Array<{ label: string }>) => {
    const dist: Record<string, number> = {};
    for (const j of set) {
      dist[j.label] = (dist[j.label] ?? 0) + 1;
    }
    console.log(
      `${label}: ${set.length} jobs — ${Object.entries(dist)
        .map(([k, v]) => `${k}=${v}`)
        .join(", ")}`,
    );
  };

  printDist("Train", trainJobs as Array<{ label: string }>);
  printDist("Test ", testJobs);

  // Write outputs
  await writeJsonlFile(trainOutput, trainJobs);
  await writeJsonlFile(testOutput, testJobs);
  console.log(`\nWrote train → ${trainOutput}`);
  console.log(`Wrote test  → ${testOutput}`);

  // Write manifest
  const manifestPath =
    getStringArg(args, "manifest") ?? "data/finetune/split_manifest.json";
  const manifest = {
    timestamp: new Date().toISOString(),
    seed,
    total,
    train_count: trainJobs.length,
    test_count: testJobs.length,
    train_ids: trainJobs.map((j) => j.job_id),
    test_ids: [...testIds],
    train_distribution: labelDist(trainJobs as Array<{ label: string }>),
    test_distribution: labelDist(testJobs),
  };
  await ensureParentDir(manifestPath);
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");
  console.log(`Wrote manifest → ${manifestPath}`);
}

function labelDist(
  jobs: Array<{ label: string }>,
): Record<string, number> {
  const dist: Record<string, number> = {};
  for (const j of jobs) {
    dist[j.label] = (dist[j.label] ?? 0) + 1;
  }
  return dist;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

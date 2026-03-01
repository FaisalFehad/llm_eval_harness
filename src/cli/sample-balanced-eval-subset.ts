import { writeFile } from "node:fs/promises";

import YAML from "yaml";

import { getBooleanArg, getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";
import { ensureParentDir } from "../lib/paths.js";
import { stratifiedSample, balancedSample } from "../lib/sampling.js";
import { GoldenJobSchema, type GoldenJob } from "../schema.js";

export { stratifiedSample, balancedSample } from "../lib/sampling.js";

type PromptfooAssert = {
  type: "javascript";
  value: string;
};

type PromptfooTest = {
  description: string;
  vars: Record<string, string | number>;
  metadata: Record<string, string | number>;
  assert: PromptfooAssert[];
};

function toPromptfooTest(job: GoldenJob, scoreTolerance: number): PromptfooTest {
  const safeLabel = JSON.stringify(job.label);

  return {
    description: `${job.job_id} | ${job.title} @ ${job.company}`,
    vars: {
      jd_text: job.jd_text,
      job_title: job.title,
      job_location: job.location ?? "",
      expected_label: job.label,
      expected_score: job.score,
    },
    metadata: {
      job_id: job.job_id,
      expected_label: job.label,
      expected_score: job.score,
    },
    assert: [
      {
        type: "javascript",
        value: `(() => { const parsed = JSON.parse(output); return parsed.label === ${safeLabel}; })()`,
      },
      {
        type: "javascript",
        value: `(() => { const parsed = JSON.parse(output); return typeof parsed.score === 'number' && Math.abs(parsed.score - ${job.score}) <= ${scoreTolerance}; })()`,
      },
      {
        type: "javascript",
        value:
          "(() => { const parsed = JSON.parse(output); return typeof parsed.reasoning === 'string' && parsed.reasoning.length >= 15; })()",
      },
    ],
  };
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/golden_jobs.jsonl";
  const outputPath = getStringArg(args, "output") ?? "data/promptfoo_tests_subset.yaml";
  const count = getNumberArg(args, "count") ?? 10;
  const seed = getNumberArg(args, "seed") ?? 42;
  const scoreTolerance = getNumberArg(args, "score-tolerance") ?? 20;
  const balanced = getBooleanArg(args, "balanced");

  const rows = await readJsonlFile<unknown>(inputPath);
  const jobs: GoldenJob[] = [];
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
    jobs.push(parsed.data);
  });

  if (errors.length > 0) {
    console.error(`Validation errors in ${inputPath}:`);
    errors.forEach((e) => console.error(`  - ${e}`));
    process.exit(1);
  }

  if (jobs.length === 0) {
    console.error(`No jobs found in ${inputPath}.`);
    process.exit(1);
  }

  const sampleFn = balanced ? balancedSample : stratifiedSample;
  const sampled = sampleFn(jobs, Math.min(count, jobs.length), seed);

  // Print label distribution
  const dist: Record<string, number> = {};
  for (const job of sampled) {
    dist[job.label] = (dist[job.label] ?? 0) + 1;
  }
  console.log(
    `Sampled ${sampled.length} jobs (seed=${seed}): ${Object.entries(dist)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ")}`,
  );

  const tests = sampled.map((job) => toPromptfooTest(job, scoreTolerance));
  const yamlPayload = YAML.stringify(tests, { lineWidth: 0 });

  await ensureParentDir(outputPath);
  await writeFile(outputPath, yamlPayload, "utf8");
  console.log(`Wrote ${tests.length} tests to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

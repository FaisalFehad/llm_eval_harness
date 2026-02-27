import { writeFile } from "node:fs/promises";

import YAML from "yaml";

import { getBooleanArg, getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";
import { ensureParentDir } from "../lib/paths.js";
import { GoldenJobSchema, type GoldenJob, type FitLabel } from "../schema.js";

/**
 * Simple seeded PRNG (mulberry32) so stratified samples are reproducible.
 */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleArray<T>(arr: T[], rng: () => number): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j]!, copy[i]!];
  }
  return copy;
}

/**
 * Stratified sample: pick N jobs maintaining approximate label distribution.
 * Guarantees at least 1 job per label if available.
 */
export function stratifiedSample(
  jobs: GoldenJob[],
  count: number,
  seed: number,
): GoldenJob[] {
  const rng = mulberry32(seed);

  const buckets: Record<FitLabel, GoldenJob[]> = {
    good_fit: [],
    maybe: [],
    bad_fit: [],
  };

  for (const job of jobs) {
    buckets[job.label].push(job);
  }

  // Shuffle each bucket
  for (const label of Object.keys(buckets) as FitLabel[]) {
    buckets[label] = shuffleArray(buckets[label], rng);
  }

  const total = jobs.length;
  const result: GoldenJob[] = [];

  // Allocate proportionally, but guarantee at least 1 per non-empty bucket
  const labels = (Object.keys(buckets) as FitLabel[]).filter(
    (l) => buckets[l].length > 0,
  );

  const allocations: Record<string, number> = {};
  let allocated = 0;

  for (const label of labels) {
    const proportion = buckets[label].length / total;
    const raw = Math.max(1, Math.round(proportion * count));
    allocations[label] = Math.min(raw, buckets[label].length);
    allocated += allocations[label]!;
  }

  // Adjust if we over/under-allocated
  while (allocated > count) {
    // Remove from the largest allocation
    const largest = labels.reduce((a, b) =>
      (allocations[a] ?? 0) > (allocations[b] ?? 0) ? a : b,
    );
    if ((allocations[largest] ?? 0) > 1) {
      allocations[largest] = (allocations[largest] ?? 0) - 1;
      allocated--;
    } else {
      break;
    }
  }

  while (allocated < count) {
    // Add to the bucket with most remaining
    const best = labels.reduce((a, b) => {
      const remainA = buckets[a].length - (allocations[a] ?? 0);
      const remainB = buckets[b].length - (allocations[b] ?? 0);
      return remainA > remainB ? a : b;
    });
    if ((allocations[best] ?? 0) < buckets[best].length) {
      allocations[best] = (allocations[best] ?? 0) + 1;
      allocated++;
    } else {
      break;
    }
  }

  for (const label of labels) {
    const take = allocations[label] ?? 0;
    result.push(...buckets[label].slice(0, take));
  }

  // Final shuffle so labels aren't grouped
  return shuffleArray(result, rng);
}

/**
 * Balanced sample: equal jobs per label (e.g. 3/4/3 for 10 jobs).
 * Better than proportional for small samples where the minority class
 * (good_fit = 11% of golden set) would get only 1 representative,
 * making per-class accuracy meaningless.
 */
export function balancedSample(
  jobs: GoldenJob[],
  count: number,
  seed: number,
): GoldenJob[] {
  const rng = mulberry32(seed);

  const buckets: Record<FitLabel, GoldenJob[]> = {
    good_fit: [],
    maybe: [],
    bad_fit: [],
  };

  for (const job of jobs) {
    buckets[job.label].push(job);
  }

  for (const label of Object.keys(buckets) as FitLabel[]) {
    buckets[label] = shuffleArray(buckets[label], rng);
  }

  const labels = (Object.keys(buckets) as FitLabel[]).filter(
    (l) => buckets[l].length > 0,
  );

  const perLabel = Math.floor(count / labels.length);
  let remainder = count - perLabel * labels.length;

  const result: GoldenJob[] = [];

  for (const label of labels) {
    const extra = remainder > 0 ? 1 : 0;
    if (extra) remainder--;
    const take = Math.min(perLabel + extra, buckets[label].length);
    result.push(...buckets[label].slice(0, take));
  }

  return shuffleArray(result, rng);
}

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
      job_location: job.location,
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

import { getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { scoreJob } from "../lib/deterministic-scorer.js";
import { GoldenJobSchema, type GoldenJob } from "../schema.js";

type ScoredJob = GoldenJob & {
  loc: number;
  role: number;
  tech: number;
  comp: number;
  computed_score: number;
  computed_label: string;
  score_match: boolean;
  label_match: boolean;
};

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/golden_jobs.jsonl";
  const outputPath =
    getStringArg(args, "output") ?? "data/golden_jobs_scored.jsonl";

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

  console.log(`Loaded ${jobs.length} golden jobs from ${inputPath}\n`);

  const scored: ScoredJob[] = [];
  const discrepancies: Array<{
    job_id: string;
    title: string;
    golden_score: number;
    computed_score: number;
    delta: number;
    golden_label: string;
    computed_label: string;
    breakdown: string;
  }> = [];

  let matchCount = 0;
  let labelMatchCount = 0;

  for (const job of jobs) {
    const breakdown = scoreJob(job.title, job.location, job.jd_text);

    const scoreMatch = breakdown.score === job.score;
    const labelMatch = breakdown.label === job.label;

    if (scoreMatch) matchCount++;
    if (labelMatch) labelMatchCount++;

    const scoredJob: ScoredJob = {
      ...job,
      loc: breakdown.loc,
      role: breakdown.role,
      tech: breakdown.tech,
      comp: breakdown.comp,
      computed_score: breakdown.score,
      computed_label: breakdown.label,
      score_match: scoreMatch,
      label_match: labelMatch,
    };
    scored.push(scoredJob);

    if (!scoreMatch || !labelMatch) {
      discrepancies.push({
        job_id: job.job_id,
        title: job.title,
        golden_score: job.score,
        computed_score: breakdown.score,
        delta: breakdown.score - job.score,
        golden_label: job.label,
        computed_label: breakdown.label,
        breakdown: `loc=${breakdown.loc} role=${breakdown.role} tech=${breakdown.tech} comp=${breakdown.comp}`,
      });
    }
  }

  // Print summary
  console.log("=== Breakdown Computation Results ===\n");
  console.log(
    `Score match:  ${matchCount}/${jobs.length} (${Math.round((matchCount / jobs.length) * 100)}%)`,
  );
  console.log(
    `Label match:  ${labelMatchCount}/${jobs.length} (${Math.round((labelMatchCount / jobs.length) * 100)}%)`,
  );
  console.log(`Discrepancies: ${discrepancies.length}/${jobs.length}\n`);

  if (discrepancies.length > 0) {
    console.log("=== Discrepancies (computed vs golden) ===\n");
    console.log(
      "job_id       | delta | golden → computed | label match | breakdown                   | title",
    );
    console.log(
      "-------------|-------|-------------------|-------------|-----------------------------|------",
    );
    for (const d of discrepancies) {
      const delta = d.delta > 0 ? `+${d.delta}` : `${d.delta}`;
      const labelOk = d.golden_label === d.computed_label ? "OK" : "MISS";
      console.log(
        `${d.job_id.padEnd(12)} | ${delta.padStart(5)} | ${String(d.golden_score).padStart(3)} → ${String(d.computed_score).padStart(3)}     | ${labelOk.padEnd(11)} | ${d.breakdown.padEnd(27)} | ${d.title.slice(0, 50)}`,
      );
    }
    console.log(
      "\nReview discrepancies before using for training. Golden score/label is the ground truth.",
    );
  }

  await writeJsonlFile(outputPath, scored);
  console.log(`\nWrote ${scored.length} scored jobs to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

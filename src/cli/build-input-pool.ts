/**
 * Build the combined input pool for V5 training.
 *
 * Combines all data sources into a single deduplicated JSONL file.
 * Strips old labels (loc/role/tech/comp/score/label) — everything
 * will be re-labeled with the V5 semantic token teacher prompt.
 *
 * Sources:
 *   - curated_training_set.jsonl (450 jobs)
 *   - preprocessed_balanced.jsonl (640 jobs)
 *   - salary_augmented.jsonl (245 augmented)
 *   - contrastive_pairs.jsonl (68 augmented)
 *   - location_variants.jsonl (50 augmented)
 *   - truncated_jds.jsonl (100 augmented)
 *
 * Usage:
 *   npx tsx src/cli/build-input-pool.ts
 */

import * as fs from "node:fs";
import { readJsonlFile } from "../lib/jsonl.js";

type RawJob = {
  job_id: string;
  title: string;
  company?: string;
  location: string;
  jd_text: string;
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
  [key: string]: unknown;
};

type CleanJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
  source_file: string;
};

const SOURCES = [
  {
    path: "data/Student Training Data/curated_training_set.jsonl",
    tag: "curated_training",
  },
  {
    path: "data/Student Training Data/preprocessed_balanced.jsonl",
    tag: "preprocessed_balanced",
  },
  {
    path: "data/Student Training Data/salary_augmented.jsonl",
    tag: "salary_augmented",
  },
  {
    path: "data/Student Training Data/contrastive_pairs.jsonl",
    tag: "contrastive_pairs",
  },
  {
    path: "data/Student Training Data/location_variants.jsonl",
    tag: "location_variants",
  },
  {
    path: "data/Student Training Data/truncated_jds.jsonl",
    tag: "truncated_jds",
  },
];

const OUTPUT = "data/v5/all_input_pool.jsonl";

async function main(): Promise<void> {
  const seen = new Set<string>();
  const pool: CleanJob[] = [];
  const stats: Record<string, { total: number; deduped: number }> = {};

  for (const source of SOURCES) {
    if (!fs.existsSync(source.path)) {
      console.warn(`SKIP: ${source.path} not found`);
      continue;
    }

    const jobs = await readJsonlFile<RawJob>(source.path);
    let added = 0;

    for (const job of jobs) {
      const id = job.job_id ?? job.id as string ?? "";
      if (!id) {
        console.warn(`  SKIP: job without ID in ${source.tag}`);
        continue;
      }
      if (seen.has(id)) continue;
      seen.add(id);

      // Skip jobs with empty/very short JD
      const jdText = (job.jd_text ?? job.description as string ?? "").trim();
      if (jdText.length < 30) {
        continue;
      }

      // Strip old labels, keep only raw fields
      const clean: CleanJob = {
        job_id: id,
        title: (job.title ?? "").trim(),
        company: (job.company ?? job.company_name as string ?? "").trim(),
        location: (job.location ?? "").trim(),
        jd_text: jdText,
        source_file: source.tag,
      };

      if (job.source_url) clean.source_url = job.source_url as string;
      if (job.augmentation_type) clean.augmentation_type = job.augmentation_type;
      if (job.source_job_id) clean.source_job_id = job.source_job_id;

      pool.push(clean);
      added++;
    }

    stats[source.tag] = { total: jobs.length, deduped: added };
  }

  // Write output
  const outStream = fs.createWriteStream(OUTPUT);
  for (const job of pool) {
    outStream.write(JSON.stringify(job) + "\n");
  }
  outStream.end();

  // Report
  console.log("═".repeat(60));
  console.log("V5 INPUT POOL BUILT");
  console.log("═".repeat(60));
  console.log();

  for (const [tag, s] of Object.entries(stats)) {
    console.log(`  ${tag.padEnd(25)} ${String(s.total).padStart(4)} input → ${String(s.deduped).padStart(4)} added`);
  }
  console.log("  " + "─".repeat(45));
  console.log(`  ${"TOTAL".padEnd(25)} ${String(pool.length).padStart(4)} unique jobs`);
  console.log();

  // Source file distribution
  const bySource: Record<string, number> = {};
  for (const j of pool) {
    bySource[j.source_file] = (bySource[j.source_file] ?? 0) + 1;
  }
  console.log("Source distribution:");
  for (const [src, count] of Object.entries(bySource).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${src.padEnd(25)} ${count}`);
  }

  console.log(`\nOutput: ${OUTPUT}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

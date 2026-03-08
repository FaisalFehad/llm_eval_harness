/**
 * Assemble a curated, balanced training set for the student model.
 *
 * Merges all labeled sources → deduplicates → excludes eval set →
 * stratified samples to ~450 curated examples with balanced label/comp distributions.
 *
 * Usage:
 *   npx tsx src/cli/assemble-student-training.ts \
 *     --eval-set "data/Student Training Data/clean_eval.jsonl" \
 *     --output "data/Student Training Data/curated_training_set.jsonl" \
 *     --target-size 450 \
 *     --augmented-cap 30 \
 *     --seed 42
 */

import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { mulberry32, shuffleArray } from "../lib/sampling.js";

// ── Types ────────────────────────────────────────────────────────────────────

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  label: string;
  score: number;
  loc: number;
  role: number;
  tech: number;
  comp: number;
  reasoning?: string;
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
};

type PoolEntry = LabeledJob & {
  source_file: string;
  is_augmented: boolean;
};

// ── Data sources ─────────────────────────────────────────────────────────────

// Teacher-labeled (from pipeline steps 2 and 4)
const TEACHER_LABELED_FILES = [
  "data/Student Training Data/teacher_labeled_500.jsonl",
  "data/Student Training Data/salary_augmented_labeled.jsonl",
  "data/Student Training Data/contrastive_pairs_labeled.jsonl",
  "data/Student Training Data/location_variants_labeled.jsonl",
  "data/Student Training Data/truncated_jds_labeled.jsonl",
];

// Existing labeled pool (already scored/labeled in earlier work)
const EXISTING_LABELED_FILES = [
  "data/distillation/combined.jsonl",
  "data/distillation/location_diversity_supplement.jsonl",
  "data/linkedin_teacher_v2_scored.jsonl",
  "data/golden_jobs_scored.jsonl",
  "data/new_uk_jobs_golden.jsonl",
];

// Files that contain augmented data (for capping augmented %)
const AUGMENTED_FILE_NAMES = new Set([
  "salary_augmented_labeled.jsonl",
  "contrastive_pairs_labeled.jsonl",
  "location_variants_labeled.jsonl",
  "truncated_jds_labeled.jsonl",
]);

// ── Helpers (duplicated from build-clean-eval-set.ts to avoid coupling) ─────

function normalizeText(value: string | undefined): string {
  if (!value) return "";
  return value.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

function normalizeCompany(raw: string): string {
  return raw
    .toLowerCase()
    .replace(
      /\b(ltd|llc|inc|plc|limited|corporation|corp|group|gmbh)\.?\b/g,
      "",
    )
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function jobDedupeKey(title: string, company: string): string {
  return `${title.toLowerCase().trim()}||${normalizeCompany(company)}`;
}

// ── Load & tag ───────────────────────────────────────────────────────────────

async function loadPool(): Promise<PoolEntry[]> {
  const pool: PoolEntry[] = [];

  // Load existing labeled files FIRST (they get dedup priority — "real" > "augmented")
  for (const file of EXISTING_LABELED_FILES) {
    try {
      const rows = await readJsonlFile<LabeledJob>(file);
      const shortName = file.split("/").pop()!;
      for (const row of rows) {
        pool.push({
          ...row,
          title: normalizeText(row.title),
          company: normalizeText(row.company),
          location: normalizeText(row.location),
          source_file: shortName,
          is_augmented: false,
        });
      }
      console.log(`  ${file}: ${rows.length} records`);
    } catch {
      console.warn(`  ${file}: skipped (not found)`);
    }
  }

  // Then load teacher-labeled files
  for (const file of TEACHER_LABELED_FILES) {
    try {
      const rows = await readJsonlFile<LabeledJob>(file);
      const shortName = file.split("/").pop()!;
      const isAug = AUGMENTED_FILE_NAMES.has(shortName);
      for (const row of rows) {
        pool.push({
          ...row,
          title: normalizeText(row.title),
          company: normalizeText(row.company),
          location: normalizeText(row.location),
          source_file: shortName,
          is_augmented: isAug,
        });
      }
      console.log(`  ${file}: ${rows.length} records (${isAug ? "augmented" : "real"})`);
    } catch {
      console.warn(`  ${file}: skipped (not found)`);
    }
  }

  return pool;
}

// ── Eval exclusion ───────────────────────────────────────────────────────────

async function buildEvalExclusionSet(
  evalPath: string,
): Promise<{ byId: Set<string>; byKey: Set<string> }> {
  const byId = new Set<string>();
  const byKey = new Set<string>();

  try {
    const evalJobs = await readJsonlFile<LabeledJob>(evalPath);
    for (const job of evalJobs) {
      byId.add(String(job.job_id));
      byKey.add(jobDedupeKey(job.title, job.company));
    }
    console.log(`\nEval exclusion set: ${evalJobs.length} jobs (${byId.size} IDs, ${byKey.size} unique keys)`);
  } catch {
    console.warn(`\nWARNING: Eval set ${evalPath} not found — no exclusion applied!`);
  }

  return { byId, byKey };
}

// ── Deduplication ────────────────────────────────────────────────────────────

function deduplicatePool(pool: PoolEntry[]): PoolEntry[] {
  const seen = new Set<string>();
  const result: PoolEntry[] = [];
  let dupes = 0;

  for (const entry of pool) {
    const key = jobDedupeKey(entry.title, entry.company);
    if (seen.has(key)) {
      dupes++;
      continue;
    }
    seen.add(key);
    result.push(entry);
  }

  console.log(`\nDeduplication: ${pool.length} → ${result.length} (${dupes} duplicates removed)`);
  return result;
}

// ── Stratified sampling ──────────────────────────────────────────────────────

const BAD_FIT_NON_UK_CAP = 50;

function stratifiedSample(
  pool: PoolEntry[],
  targetSize: number,
  augmentedCapPct: number,
  rng: () => number,
): PoolEntry[] {
  // Target label distribution: 25% good_fit, 38% maybe, 37% bad_fit
  const labelTargets: Record<string, number> = {
    good_fit: Math.round(targetSize * 0.25),
    maybe: Math.round(targetSize * 0.38),
    bad_fit: targetSize - Math.round(targetSize * 0.25) - Math.round(targetSize * 0.38),
  };

  // Split pool by label → real vs augmented
  const buckets: Record<string, { real: PoolEntry[]; augmented: PoolEntry[] }> = {
    good_fit: { real: [], augmented: [] },
    maybe: { real: [], augmented: [] },
    bad_fit: { real: [], augmented: [] },
  };

  for (const entry of pool) {
    const label = entry.label;
    if (!buckets[label]) continue;
    if (entry.is_augmented) {
      buckets[label].augmented.push(entry);
    } else {
      buckets[label].real.push(entry);
    }
  }

  // Shuffle each sub-bucket for reproducible random selection
  for (const label of Object.keys(buckets)) {
    buckets[label].real = shuffleArray(buckets[label].real, rng);
    buckets[label].augmented = shuffleArray(buckets[label].augmented, rng);
  }

  const maxAugmented = Math.floor(targetSize * (augmentedCapPct / 100));
  let totalAugmented = 0;
  const result: PoolEntry[] = [];

  for (const label of ["good_fit", "maybe", "bad_fit"]) {
    const target = labelTargets[label]!;
    let { real, augmented } = buckets[label];
    const selected: PoolEntry[] = [];

    // For bad_fit: cap non-UK (loc=-50) at BAD_FIT_NON_UK_CAP, then UK fills the rest.
    // This prevents the training set from being flooded with "wrong country" examples
    // which are trivially learnable and don't teach nuanced decision boundaries.
    if (label === "bad_fit") {
      const nonUkReal = real.filter((j) => j.loc === -50);
      const ukReal = real.filter((j) => j.loc !== -50);
      const cappedNonUk = nonUkReal.slice(0, BAD_FIT_NON_UK_CAP);
      // Put UK first (more valuable — wrong role/tech/comp signals), then capped non-UK
      real = [...ukReal, ...cappedNonUk];

      const nonUkAug = augmented.filter((j) => j.loc === -50);
      const ukAug = augmented.filter((j) => j.loc !== -50);
      const cappedNonUkAug = nonUkAug.slice(0, Math.max(0, BAD_FIT_NON_UK_CAP - cappedNonUk.length));
      augmented = [...ukAug, ...cappedNonUkAug];
    }

    // Phase 1: Take real jobs first (always preferred over augmented)
    const realTake = Math.min(real.length, target);
    selected.push(...real.slice(0, realTake));

    // Phase 2: Fill remaining from augmented (respecting global cap)
    const remaining = target - selected.length;
    if (remaining > 0) {
      const augBudget = Math.min(remaining, maxAugmented - totalAugmented);
      const augTake = Math.min(augmented.length, augBudget);
      if (augTake > 0) {
        selected.push(...augmented.slice(0, augTake));
        totalAugmented += augTake;
      }
    }

    // Phase 3: If still short, oversample existing selected entries.
    // Spread duplicates evenly across unique jobs rather than repeating the
    // same job many times — distributes the gradient signal more evenly.
    const stillNeeded = target - selected.length;
    if (stillNeeded > 0 && selected.length > 0) {
      console.log(
        `  ${label}: oversampling ${stillNeeded} from ${selected.length} unique`,
      );
      const uniquePool = [...selected]; // copy the unique entries to sample from
      for (let d = 0; d < stillNeeded; d++) {
        // Round-robin through unique jobs so each gets duplicated roughly equally
        const sourceIdx = d % uniquePool.length;
        const source = uniquePool[sourceIdx]!;
        selected.push({
          ...source,
          job_id: `${source.job_id}_dup_${d}`,
        });
      }
    }

    console.log(
      `  ${label}: target=${target}, selected=${selected.length} ` +
      `(${real.slice(0, realTake).length} real, ` +
      `${selected.filter((s) => s.is_augmented).length} aug, ` +
      `${Math.max(0, stillNeeded)} oversampled)`,
    );

    result.push(...selected);
  }

  return shuffleArray(result, rng);
}

// ── Stats printing ───────────────────────────────────────────────────────────

function printStats(jobs: PoolEntry[]): void {
  // Label distribution
  const labels: Record<string, number> = {};
  const comps: Record<number, number> = {};
  const locs: Record<number, number> = {};
  let augCount = 0;
  const sources: Record<string, number> = {};

  for (const job of jobs) {
    labels[job.label] = (labels[job.label] ?? 0) + 1;
    comps[job.comp] = (comps[job.comp] ?? 0) + 1;
    locs[job.loc] = (locs[job.loc] ?? 0) + 1;
    if (job.is_augmented) augCount++;
    sources[job.source_file] = (sources[job.source_file] ?? 0) + 1;
  }

  console.log("\n═══════════════════════════════════════════════════");
  console.log("  CURATED TRAINING SET STATS");
  console.log("═══════════════════════════════════════════════════");

  console.log(`\n  Total: ${jobs.length}`);
  console.log(`  Augmented: ${augCount} (${((augCount / jobs.length) * 100).toFixed(1)}%)`);

  console.log("\n  Labels:");
  for (const [label, count] of Object.entries(labels).sort()) {
    console.log(`    ${label}: ${count} (${((count / jobs.length) * 100).toFixed(1)}%)`);
  }

  console.log("\n  Comp scores:");
  for (const [comp, count] of Object.entries(comps).sort((a, b) => Number(a[0]) - Number(b[0]))) {
    console.log(`    comp=${comp}: ${count} (${((count / jobs.length) * 100).toFixed(1)}%)`);
  }

  const compNonZero = jobs.filter((j) => j.comp !== 0).length;
  console.log(`    comp≠0 total: ${compNonZero} (${((compNonZero / jobs.length) * 100).toFixed(1)}%)`);

  console.log("\n  Location scores:");
  for (const [loc, count] of Object.entries(locs).sort((a, b) => Number(a[0]) - Number(b[0]))) {
    console.log(`    loc=${loc}: ${count} (${((count / jobs.length) * 100).toFixed(1)}%)`);
  }

  console.log("\n  Sources:");
  for (const [src, count] of Object.entries(sources).sort((a, b) => b[1] - a[1])) {
    console.log(`    ${src}: ${count}`);
  }

  // Label × Comp cross-tab
  console.log("\n  Label × Comp:");
  const compVals = Array.from(new Set(jobs.map((j) => j.comp))).sort((a, b) => a - b);
  const header = "           " + compVals.map((c) => `comp=${c}`.padStart(8)).join("") + "   TOTAL";
  console.log(`  ${header}`);
  for (const label of ["good_fit", "maybe", "bad_fit"]) {
    const row = label.padEnd(10);
    const cells = compVals.map((c) => {
      const count = jobs.filter((j) => j.label === label && j.comp === c).length;
      return String(count).padStart(8);
    });
    const total = jobs.filter((j) => j.label === label).length;
    console.log(`  ${row} ${cells.join("")}   ${total}`);
  }

  console.log("═══════════════════════════════════════════════════\n");
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const evalSetPath =
    getStringArg(args, "eval-set") ??
    "data/Student Training Data/clean_eval.jsonl";
  const outputPath =
    getStringArg(args, "output") ??
    "data/Student Training Data/curated_training_set.jsonl";
  const targetSize = getNumberArg(args, "target-size") ?? 450;
  const augmentedCap = getNumberArg(args, "augmented-cap") ?? 30;
  const seed = getNumberArg(args, "seed") ?? 42;

  console.log("Loading labeled data sources...\n");
  const rawPool = await loadPool();
  console.log(`\nTotal loaded: ${rawPool.length} records`);

  // Exclude eval set
  const evalExclusion = await buildEvalExclusionSet(evalSetPath);
  const nonEval = rawPool.filter((entry) => {
    const id = String(entry.job_id);
    const key = jobDedupeKey(entry.title, entry.company);
    return !evalExclusion.byId.has(id) && !evalExclusion.byKey.has(key);
  });
  console.log(`After eval exclusion: ${nonEval.length} (removed ${rawPool.length - nonEval.length})`);

  // Deduplicate
  const deduped = deduplicatePool(nonEval);

  // Print pool stats before sampling
  console.log("\n── Pool before sampling ──");
  const poolLabels: Record<string, number> = {};
  for (const e of deduped) poolLabels[e.label] = (poolLabels[e.label] ?? 0) + 1;
  for (const [l, c] of Object.entries(poolLabels).sort()) {
    console.log(`  ${l}: ${c}`);
  }
  console.log(`  Total: ${deduped.length}`);

  // Stratified sample
  const rng = mulberry32(seed);
  const curated = stratifiedSample(deduped, targetSize, augmentedCap, rng);

  // Print final stats
  printStats(curated);

  // Write output
  await writeJsonlFile(outputPath, curated);
  console.log(`Written ${curated.length} curated examples to ${outputPath}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

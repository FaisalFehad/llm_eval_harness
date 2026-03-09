/**
 * Build V5 Eval Set + Training Set from labeled pool.
 *
 * Step 2 of TRAINING_PLAN_V5:
 *   2A: Select 150 eval jobs (50 good_fit, 50 maybe, 50 bad_fit)
 *   2B: Assemble ~800 training jobs from remaining pool
 *   2C: Enforce distribution minimums, cap synthetic ≤25%
 *   2D: Deduplicate, verify zero eval overlap
 *
 * Usage:
 *   npx tsx src/cli/build-datasets.ts \
 *     --input data/v5/all_labeled_pool.jsonl \
 *     --eval-output data/v5/eval_150_golden.jsonl \
 *     --train-output data/v5/train_800.jsonl \
 *     --seed 42 \
 *     --target-train 800
 */

import * as fs from "node:fs";
import * as crypto from "node:crypto";
import { execSync } from "node:child_process";
import { parseArgs, getStringArg, getNumberArg, getBooleanArg } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  loc_reason: string;
  loc: string;
  role_reason: string;
  role: string;
  tech_reason: string;
  tech: string;
  comp_reason: string;
  comp: string;
  loc_score: number;
  role_score: number;
  tech_score: number;
  comp_score: number;
  score: number;
  label: string;
  source_file?: string;
  augmentation_type?: string;
  source_job_id?: string;
  synthetic_target?: string;
};

// ── Seeded RNG ──────────────────────────────────────────────────────────

function seededRng(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) & 0xffffffff;
    return (state >>> 0) / 0x100000000;
  };
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}

// ── Family ID extraction ─────────────────────────────────────────────────
// Jobs can be suffixed: _dup_N, _con_title, _con_loc, _sal_N, _locvar_N, _trunc_N
// Extract the base "family" ID so all variants of the same job are treated as one unit.
const SUFFIX_PATTERN = /(_dup_\d+|_con_\w+|_sal_\d+|_locvar_\d+|_trunc_\d+)$/;

function familyId(jobId: string): string {
  return jobId.replace(SUFFIX_PATTERN, "");
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v5/all_labeled_pool.jsonl";
  const evalOutput = getStringArg(args, "eval-output") ?? "data/v5/eval_150_golden.jsonl";
  const trainOutput = getStringArg(args, "train-output") ?? "data/v5/train_800.jsonl";
  const seed = getNumberArg(args, "seed") ?? 42;
  const targetTrain = getNumberArg(args, "target-train") ?? 800;
  const skipAudit = getBooleanArg(args, "skip-audit");
  const force = getBooleanArg(args, "force");

  // ── Overwrite protection ───────────────────────────────────────────────
  // Refuse to overwrite non-empty output files without --force.
  // This prevents accidental data loss (e.g. overwriting the locked eval set).
  for (const [label, filePath] of [["eval", evalOutput], ["train", trainOutput]] as const) {
    if (fs.existsSync(filePath)) {
      const stat = fs.statSync(filePath);
      if (stat.size > 0 && !force) {
        const readOnly = (stat.mode & 0o200) === 0;
        console.error(`\n══════════════════════════════════════════════════════`);
        console.error(`  REFUSED: ${label} output file already exists and is non-empty:`);
        console.error(`  ${filePath} (${stat.size} bytes${readOnly ? ", READ-ONLY" : ""})`);
        console.error(`  Use --force to overwrite, or specify a different --${label}-output path.`);
        console.error(`══════════════════════════════════════════════════════\n`);
        process.exit(1);
      }
      // If --force, unlock read-only files before writing
      if (force && (stat.mode & 0o200) === 0) {
        fs.chmodSync(filePath, 0o644);
        console.log(`  Unlocked ${label} file for overwrite: ${filePath}`);
      }
    }
  }

  // Also refuse if input path equals any output path (would destroy source data)
  if (inputPath === evalOutput || inputPath === trainOutput) {
    console.error(`\n══════════════════════════════════════════════════════`);
    console.error(`  REFUSED: input path cannot be the same as an output path.`);
    console.error(`  Input: ${inputPath}`);
    console.error(`  This would destroy the source data.`);
    console.error(`══════════════════════════════════════════════════════\n`);
    process.exit(1);
  }

  // ── Pre-flight audit gate ──────────────────────────────────────────────
  if (!skipAudit) {
    console.log("── PRE-FLIGHT AUDIT ───────────────────────────────────");
    console.log(`Running audit on ${inputPath}...`);
    try {
      // Only pass --eval-set if the file exists (it may not yet during initial V6 bootstrap)
      const evalSetFlag = fs.existsSync(evalOutput) && fs.statSync(evalOutput).size > 0
        ? ` --eval-set "${evalOutput}"`
        : "";
      const auditCmd = `npx tsx src/cli/audit-training-data.ts --input "${inputPath}"${evalSetFlag} --dry-run`;
      execSync(auditCmd, { stdio: "inherit" });
      console.log("Audit passed.\n");
    } catch {
      console.error("\n══════════════════════════════════════════════════════");
      console.error("  AUDIT FAILED — refusing to build datasets.");
      console.error("  Fix critical issues first, or use --skip-audit to bypass.");
      console.error("  Run audit separately:  npx tsx src/cli/audit-training-data.ts --input " + inputPath);
      console.error("══════════════════════════════════════════════════════\n");
      process.exit(1);
    }
  } else {
    console.log("⚠  Audit skipped (--skip-audit)\n");
  }

  const allJobs = await readJsonlFile<LabeledJob>(inputPath);
  console.log(`Loaded ${allJobs.length} labeled jobs from ${inputPath}`);

  const rng = seededRng(seed);

  // ── Classify jobs ──────────────────────────────────────────────────────

  const isSynthetic = (j: LabeledJob) =>
    j.source_file === "synthetic_v5" || (j.augmentation_type ?? "").length > 0;

  const byLabel: Record<string, LabeledJob[]> = { good_fit: [], maybe: [], bad_fit: [] };
  for (const j of allJobs) {
    if (byLabel[j.label]) byLabel[j.label]!.push(j);
  }

  console.log(`\nLabel distribution:`);
  for (const [lbl, jobs] of Object.entries(byLabel)) {
    const real = jobs.filter((j) => !isSynthetic(j)).length;
    const synth = jobs.filter((j) => isSynthetic(j)).length;
    console.log(`  ${lbl.padEnd(10)}: ${jobs.length} (${real} real, ${synth} synthetic)`);
  }

  // ── Step 2A: Select 150 eval jobs ──────────────────────────────────────

  console.log("\n── BUILDING EVAL SET (150 jobs) ────────────────────────");

  const evalJobs: LabeledJob[] = [];
  const evalIds = new Set<string>();

  for (const label of ["good_fit", "maybe", "bad_fit"] as const) {
    const candidates = shuffle(byLabel[label]!, rng);

    // Prefer real jobs for eval
    const real = candidates.filter((j) => !isSynthetic(j));
    const synth = candidates.filter((j) => isSynthetic(j));
    const ordered = [...real, ...synth];

    // Select up to 50, but cap at 40% of available to leave enough for training
    const evalTarget = Math.min(50, Math.floor(candidates.length * 0.4));
    const selected: LabeledJob[] = [];
    const usedCompanies = new Set<string>();

    for (const j of ordered) {
      if (selected.length >= evalTarget) break;

      // Avoid duplicate companies in eval
      const compKey = (j.company ?? "").toLowerCase().trim();
      if (compKey && usedCompanies.has(compKey)) continue;

      selected.push(j);
      evalIds.add(j.job_id);
      if (compKey) usedCompanies.add(compKey);
    }

    evalJobs.push(...selected);
    console.log(`  ${label}: selected ${selected.length} (${selected.filter((j) => !isSynthetic(j)).length} real)`);
  }

  // Check eval coverage requirements
  const evalOutsideUk = evalJobs.filter((j) => j.loc === "OUTSIDE_UK").length;
  const evalCompHardNeg = evalJobs.filter((j) =>
    j.comp === "UP_TO_ONLY" || j.comp === "NO_GBP",
  ).length;
  const evalNodeTech = evalJobs.filter((j) =>
    j.tech === "NODE" || j.tech === "NODE_JS_TS",
  ).length;
  const evalBorderline = evalJobs.filter((j) => j.score >= 45 && j.score <= 75).length;

  console.log(`\n  Coverage checks:`);
  console.log(`    OUTSIDE_UK:     ${evalOutsideUk} (target: 15+)`);
  console.log(`    Comp hard neg:  ${evalCompHardNeg} (target: 10+)`);
  console.log(`    NODE/NODE_JS_TS: ${evalNodeTech} (target: 10+)`);
  console.log(`    Borderline:     ${evalBorderline} (target: 20+)`);

  // ── Step 2B: Assemble training jobs ────────────────────────────────────

  console.log("\n── BUILDING TRAINING SET ────────────────────────────────");

  // Remaining pool — exclude eval jobs and ALL relatives (dup, augmented, etc.)
  // Build a set of "family IDs" from eval: if job 4372742578 is in eval,
  // then 4372742578_dup_33, 4372742578_con_title, etc. must all be excluded.
  const evalFamilyIds = new Set<string>();
  for (const j of evalJobs) {
    evalFamilyIds.add(familyId(j.job_id));
    if (j.source_job_id) evalFamilyIds.add(familyId(j.source_job_id));
  }
  // Also build JD text fingerprints from eval to catch cross-posting duplicates
  // (different companies/IDs posting identical JD text)
  const evalJdFingerprints = new Set<string>();
  for (const j of evalJobs) {
    evalJdFingerprints.add(
      crypto.createHash("sha256").update(j.jd_text.slice(0, 500)).digest("hex").slice(0, 16),
    );
  }
  let familyExcluded = 0;
  let jdExcluded = 0;
  const trainPool = allJobs.filter((j) => {
    if (evalIds.has(j.job_id)) return false;
    // Exclude any job in the same family as an eval job
    if (evalFamilyIds.has(familyId(j.job_id))) { familyExcluded++; return false; }
    // Exclude by source_job_id too
    if (j.source_job_id && evalFamilyIds.has(familyId(j.source_job_id))) { familyExcluded++; return false; }
    // Exclude jobs with identical JD text as any eval job (catches cross-posting duplicates)
    const jdFp = crypto.createHash("sha256").update(j.jd_text.slice(0, 500)).digest("hex").slice(0, 16);
    if (evalJdFingerprints.has(jdFp)) { jdExcluded++; return false; }
    return true;
  });
  console.log(`  Pool after eval removal: ${trainPool.length} (${familyExcluded} family, ${jdExcluded} JD-text excluded)`);

  // Deduplicate by job_id (preserves augmented variants with different JD content)
  const seen = new Set<string>();
  const deduped: LabeledJob[] = [];
  for (const j of trainPool) {
    if (seen.has(j.job_id)) continue;
    seen.add(j.job_id);
    deduped.push(j);
  }
  console.log(`  After deduplication: ${deduped.length}`);

  // Separate real and synthetic
  const realPool = shuffle(deduped.filter((j) => !isSynthetic(j)), rng);
  const synthPool = shuffle(deduped.filter((j) => isSynthetic(j)), rng);

  // Build training set: stratified by label
  const trainJobs: LabeledJob[] = [];
  const synthCap = Math.floor(targetTrain * 0.25);
  let synthCount = 0;

  const trainByLabel: Record<string, { target: number; real: LabeledJob[]; synth: LabeledJob[] }> = {
    good_fit: { target: Math.round(targetTrain * 0.18), real: [], synth: [] },  // ~145
    maybe: { target: Math.round(targetTrain * 0.25), real: [], synth: [] },      // ~200
    bad_fit: { target: Math.round(targetTrain * 0.57), real: [], synth: [] },     // ~455
  };

  // Sort into label buckets
  for (const j of realPool) {
    if (trainByLabel[j.label]) trainByLabel[j.label]!.real.push(j);
  }
  for (const j of synthPool) {
    if (trainByLabel[j.label]) trainByLabel[j.label]!.synth.push(j);
  }

  // Fill each label bucket: real first, then synthetic
  for (const [label, bucket] of Object.entries(trainByLabel)) {
    let added = 0;

    // Add real jobs first
    for (const j of bucket.real) {
      if (added >= bucket.target) break;
      trainJobs.push(j);
      added++;
    }

    // Fill remainder with synthetic (up to cap)
    for (const j of bucket.synth) {
      if (added >= bucket.target) break;
      if (synthCount >= synthCap) break;
      trainJobs.push(j);
      synthCount++;
      added++;
    }

    console.log(`  ${label.padEnd(10)}: ${added} (target ${bucket.target})`);
  }

  console.log(`\n  Total training: ${trainJobs.length}`);
  console.log(`  Synthetic in training: ${synthCount} (${((synthCount / trainJobs.length) * 100).toFixed(1)}%, cap ${synthCap})`);

  // Verify zero overlap (by job_id, family_id, source_job_id, AND JD text)
  const trainIds = new Set(trainJobs.map((j) => j.job_id));
  const overlap = [...evalIds].filter((id) => trainIds.has(id));
  if (overlap.length > 0) {
    console.error(`\n❌ FATAL: ${overlap.length} eval/train job_id overlap!`);
    process.exit(1);
  }
  // Check for family-level contamination
  const familyOverlap: string[] = [];
  for (const tj of trainJobs) {
    if (evalFamilyIds.has(familyId(tj.job_id))) {
      familyOverlap.push(`${tj.job_id} (family: ${familyId(tj.job_id)})`);
    }
    if (tj.source_job_id && evalFamilyIds.has(familyId(tj.source_job_id))) {
      familyOverlap.push(`${tj.job_id} (source family: ${familyId(tj.source_job_id)})`);
    }
  }
  if (familyOverlap.length > 0) {
    console.error(`\n❌ FATAL: ${familyOverlap.length} family-level contamination leaks!`);
    for (const s of familyOverlap.slice(0, 10)) console.error(`  - ${s}`);
    if (familyOverlap.length > 10) console.error(`  ... and ${familyOverlap.length - 10} more`);
    process.exit(1);
  }
  // Check for JD text content overlap
  const jdOverlap: string[] = [];
  for (const tj of trainJobs) {
    const fp = crypto.createHash("sha256").update(tj.jd_text.slice(0, 500)).digest("hex").slice(0, 16);
    if (evalJdFingerprints.has(fp)) {
      jdOverlap.push(tj.job_id);
    }
  }
  if (jdOverlap.length > 0) {
    console.error(`\n❌ FATAL: ${jdOverlap.length} JD text content leaks!`);
    for (const s of jdOverlap.slice(0, 10)) console.error(`  - ${s}`);
    process.exit(1);
  }
  console.log("  ✅ Zero eval/train overlap verified (job_id + family_id + source_job_id + JD text)");

  // ── Write outputs ──────────────────────────────────────────────────────

  // Write eval set
  const evalContent = evalJobs.map((j) => JSON.stringify(j)).join("\n") + "\n";
  fs.writeFileSync(evalOutput, evalContent);

  // Lock eval file as read-only (chmod 444) to prevent accidental overwrites
  fs.chmodSync(evalOutput, 0o444);
  console.log(`  Locked eval file as read-only (chmod 444): ${evalOutput}`);

  // Compute SHA-256 of eval set
  const evalHash = crypto.createHash("sha256").update(evalContent).digest("hex");

  // Write training set
  const trainContent = shuffle(trainJobs, rng).map((j) => JSON.stringify(j)).join("\n") + "\n";
  fs.writeFileSync(trainOutput, trainContent);

  // Write manifest
  const manifest = {
    seed,
    timestamp: new Date().toISOString(),
    eval: {
      path: evalOutput,
      count: evalJobs.length,
      sha256: evalHash,
      by_label: {
        good_fit: evalJobs.filter((j) => j.label === "good_fit").length,
        maybe: evalJobs.filter((j) => j.label === "maybe").length,
        bad_fit: evalJobs.filter((j) => j.label === "bad_fit").length,
      },
    },
    train: {
      path: trainOutput,
      count: trainJobs.length,
      synthetic_count: synthCount,
      synthetic_pct: Number(((synthCount / trainJobs.length) * 100).toFixed(1)),
      by_label: {
        good_fit: trainJobs.filter((j) => j.label === "good_fit").length,
        maybe: trainJobs.filter((j) => j.label === "maybe").length,
        bad_fit: trainJobs.filter((j) => j.label === "bad_fit").length,
      },
    },
  };
  const manifestPath = trainOutput.replace(/[^/]+$/, "split_manifest.json");
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));

  console.log("\n" + "═".repeat(60));
  console.log("DATASETS BUILT");
  console.log("═".repeat(60));
  console.log(`Eval:     ${evalOutput} (${evalJobs.length} jobs, SHA-256: ${evalHash.slice(0, 16)}...)`);
  console.log(`Train:    ${trainOutput} (${trainJobs.length} jobs)`);
  console.log(`Manifest: ${manifestPath}`);
  console.log("\nNext steps:");
  console.log("  1. Manually verify all 50 'maybe' eval labels");
  console.log("  2. Spot-check 20 good_fit + 20 bad_fit eval labels");
  console.log("  3. Format for MLX: npx tsx src/cli/format-for-mlx.ts");
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

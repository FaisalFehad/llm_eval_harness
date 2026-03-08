/**
 * Check distributions of V5 semantic token labeled data against plan minimums.
 *
 * Reads the labeled pool, counts each token, and compares against
 * the minimums specified in TRAINING_PLAN_V5.md. Reports gaps that
 * need synthetic data.
 *
 * Also checks:
 *   - Location diversity (unique cities/countries)
 *   - Comp sub-categories (USD visible, EUR visible, daily rates, etc.)
 *   - Label balance (good_fit, maybe, bad_fit)
 *   - Borderline scores (45-75)
 *   - Edge cases
 *
 * Usage:
 *   npx tsx src/cli/check-distribution.ts \
 *     --input data/v5/all_labeled_pool.jsonl
 */

import * as fs from "node:fs";
import { parseArgs, getStringArg } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  loc: string;
  role: string;
  tech: string;
  comp: string;
  reasoning: string;
  loc_score: number;
  role_score: number;
  tech_score: number;
  comp_score: number;
  score: number;
  label: string;
  augmentation_type?: string;
};

// ── Target minimums from TRAINING_PLAN_V5.md ─────────────────────────────

const LOC_MINIMUMS: Record<string, number> = {
  OUTSIDE_UK: 100,
  UK_OTHER: 80,
  LONDON_OR_REMOTE: 250,
  MISSING: 15,
};

const TECH_MINIMUMS: Record<string, number> = {
  NONE: 150,
  JS_TS: 100,
  NODE: 80,
  NODE_JS_TS: 120,
  AI_ML: 20,
  JS_TS_AI_ML: 20,
  NODE_AI_ML: 20,
  NODE_JS_TS_AI_ML: 30,
};

const COMP_MINIMUMS: Record<string, number> = {
  // Aggregated minimums for comp tokens
  NO_GBP: 205, // 40+20+15+120+10 (various sub-categories)
  UP_TO_ONLY: 40,
  BELOW_45K: 20,
  RANGE_55_74K: 50,
  RANGE_75_99K: 80,
  ABOVE_100K: 100,
};

const ROLE_MINIMUMS: Record<string, number> = {
  // SENIOR_PLUS: 250 + 40 edge = 290
  SENIOR_PLUS: 290,
  MID_LEVEL: 120,
  NO_SENIORITY: 120,
};

const LABEL_MINIMUMS: Record<string, number> = {
  good_fit: 150,
  maybe: 200,
  bad_fit: 300,
};

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v5/all_labeled_pool.jsonl";

  if (!fs.existsSync(inputPath)) {
    console.error(`File not found: ${inputPath}`);
    process.exit(1);
  }

  const jobs = await readJsonlFile<LabeledJob>(inputPath);
  console.log(`Loaded ${jobs.length} labeled jobs from ${inputPath}\n`);

  // ── Count token distributions ──────────────────────────────────────────

  const locDist: Record<string, number> = {};
  const roleDist: Record<string, number> = {};
  const techDist: Record<string, number> = {};
  const compDist: Record<string, number> = {};
  const labelDist: Record<string, number> = {};
  let borderlineCount = 0;

  for (const j of jobs) {
    locDist[j.loc] = (locDist[j.loc] ?? 0) + 1;
    roleDist[j.role] = (roleDist[j.role] ?? 0) + 1;
    techDist[j.tech] = (techDist[j.tech] ?? 0) + 1;
    compDist[j.comp] = (compDist[j.comp] ?? 0) + 1;
    labelDist[j.label] = (labelDist[j.label] ?? 0) + 1;
    if (j.score >= 45 && j.score <= 75) borderlineCount++;
  }

  // ── Location diversity ─────────────────────────────────────────────────

  const outsideUkLocations = new Set<string>();
  const ukOtherLocations = new Set<string>();
  const londonFormats = new Set<string>();

  for (const j of jobs) {
    const loc = j.location.trim();
    if (j.loc === "OUTSIDE_UK" && loc) outsideUkLocations.add(loc);
    if (j.loc === "UK_OTHER" && loc) ukOtherLocations.add(loc);
    if (j.loc === "LONDON_OR_REMOTE" && loc) londonFormats.add(loc);
  }

  // ── Comp sub-categories ────────────────────────────────────────────────

  let noGbpWithUsd = 0;
  let noGbpWithEur = 0;
  let noGbpDailyRate = 0;
  let noGbpNoSalary = 0;
  let noGbpMidpoint45_54 = 0;

  for (const j of jobs) {
    if (j.comp !== "NO_GBP") continue;
    const reason = (j.reasoning ?? "").toLowerCase();
    const jd = j.jd_text.toLowerCase();

    if (reason.includes("usd") || reason.includes("$") || jd.includes("$")) {
      noGbpWithUsd++;
    } else if (reason.includes("eur") || reason.includes("€") || jd.includes("€")) {
      noGbpWithEur++;
    } else if (reason.includes("daily rate") || reason.includes("/day") || reason.includes("per day") || reason.includes("p/d")) {
      noGbpDailyRate++;
    } else if (reason.includes("£45") || reason.includes("£46") || reason.includes("£47") ||
               reason.includes("£48") || reason.includes("£49") || reason.includes("£50") ||
               reason.includes("£51") || reason.includes("£52") || reason.includes("£53") ||
               reason.includes("£54")) {
      noGbpMidpoint45_54++;
    } else {
      noGbpNoSalary++;
    }
  }

  // ── Role edge cases ────────────────────────────────────────────────────

  const roleEdgeKeywords = [
    "snr", "sr.", "engineer iii", "swe iii", "vp", "founding",
    "distinguished", "head of", "tech lead", "principal",
  ];
  let seniorEdgeCount = 0;
  for (const j of jobs) {
    if (j.role !== "SENIOR_PLUS") continue;
    const title = j.title.toLowerCase();
    if (roleEdgeKeywords.some((kw) => title.includes(kw))) {
      seniorEdgeCount++;
    }
  }

  // ── Report ─────────────────────────────────────────────────────────────

  console.log("═".repeat(70));
  console.log("V5 DISTRIBUTION REPORT");
  console.log("═".repeat(70));

  function reportField(
    name: string,
    dist: Record<string, number>,
    minimums: Record<string, number>,
  ): number {
    console.log(`\n── ${name} ${"─".repeat(60 - name.length)}`);
    let totalGap = 0;
    for (const [token, min] of Object.entries(minimums).sort(
      (a, b) => b[1] - a[1],
    )) {
      const actual = dist[token] ?? 0;
      const gap = Math.max(0, min - actual);
      const status = gap === 0 ? "✅" : "❌";
      const gapStr = gap > 0 ? ` (need ${gap} more)` : "";
      console.log(
        `  ${status} ${token.padEnd(22)} ${String(actual).padStart(4)} / ${String(min).padStart(4)}${gapStr}`,
      );
      totalGap += gap;
    }
    // Show any tokens not in minimums
    for (const [token, count] of Object.entries(dist)) {
      if (!(token in minimums)) {
        console.log(`  ℹ️  ${token.padEnd(22)} ${String(count).padStart(4)} (no minimum)`);
      }
    }
    return totalGap;
  }

  let totalGaps = 0;
  totalGaps += reportField("LOCATION", locDist, LOC_MINIMUMS);
  totalGaps += reportField("ROLE", roleDist, ROLE_MINIMUMS);
  totalGaps += reportField("TECH", techDist, TECH_MINIMUMS);
  totalGaps += reportField("COMP", compDist, COMP_MINIMUMS);
  totalGaps += reportField("LABEL", labelDist, LABEL_MINIMUMS);

  // Borderline
  console.log(`\n── BORDERLINE (score 45-75) ${"─".repeat(40)}`);
  const borderlineMin = 150;
  const borderlineGap = Math.max(0, borderlineMin - borderlineCount);
  const bStatus = borderlineGap === 0 ? "✅" : "❌";
  console.log(`  ${bStatus} borderline (45-75)      ${String(borderlineCount).padStart(4)} / ${borderlineMin}${borderlineGap > 0 ? ` (need ${borderlineGap} more)` : ""}`);
  totalGaps += borderlineGap;

  // Location diversity
  console.log(`\n── LOCATION DIVERSITY ${"─".repeat(47)}`);
  console.log(`  OUTSIDE_UK unique locations: ${outsideUkLocations.size} (target: 30+)`);
  console.log(`  UK_OTHER unique locations:   ${ukOtherLocations.size} (target: 15+)`);
  console.log(`  LONDON_OR_REMOTE formats:    ${londonFormats.size}`);

  // Comp sub-categories
  console.log(`\n── COMP SUB-CATEGORIES ${"─".repeat(46)}`);
  console.log(`  NO_GBP with USD visible:     ${noGbpWithUsd} (target: 40)`);
  console.log(`  NO_GBP with EUR/other:       ${noGbpWithEur} (target: 20)`);
  console.log(`  NO_GBP daily rate:           ${noGbpDailyRate} (target: 15)`);
  console.log(`  NO_GBP no salary at all:     ${noGbpNoSalary} (target: 120)`);
  console.log(`  NO_GBP midpoint £45k-54k:    ${noGbpMidpoint45_54} (target: 10)`);

  // Role edge
  console.log(`\n── ROLE EDGE CASES ${"─".repeat(49)}`);
  console.log(`  SENIOR_PLUS with edge titles: ${seniorEdgeCount} (target: 40)`);

  // Summary
  console.log(`\n${"═".repeat(70)}`);
  console.log(`TOTAL GAPS: ${totalGaps} jobs needed across all categories`);
  if (totalGaps === 0) {
    console.log("All distribution minimums met! ✅");
  } else {
    console.log("Synthetic data generation needed for gap categories. ❌");
  }
  console.log("═".repeat(70));

  // Write machine-readable report
  const reportPath = inputPath.replace(/\.jsonl$/, ".distribution.json");
  const report = {
    total_jobs: jobs.length,
    loc: locDist,
    role: roleDist,
    tech: techDist,
    comp: compDist,
    label: labelDist,
    borderline_count: borderlineCount,
    total_gaps: totalGaps,
    location_diversity: {
      outside_uk_unique: outsideUkLocations.size,
      uk_other_unique: ukOtherLocations.size,
      london_formats: londonFormats.size,
    },
    comp_sub: {
      no_gbp_with_usd: noGbpWithUsd,
      no_gbp_with_eur: noGbpWithEur,
      no_gbp_daily_rate: noGbpDailyRate,
      no_gbp_no_salary: noGbpNoSalary,
      no_gbp_midpoint_45_54: noGbpMidpoint45_54,
    },
    role_edge: {
      senior_plus_edge: seniorEdgeCount,
    },
  };
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(`\nMachine-readable report: ${reportPath}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

/**
 * Augment teacher-labeled jobs to improve training distribution.
 *
 * Produces 4 output files (unlabeled, ready for teacher labeling):
 *   - salary_augmented.jsonl    (~200 records)
 *   - contrastive_pairs.jsonl   (~90  records)
 *   - location_variants.jsonl   (~50  records)
 *   - truncated_jds.jsonl       (~100 records)
 *
 * Usage:
 *   npx tsx src/cli/augment-training-data.ts \
 *     --input "data/Student Training Data/teacher_labeled_500.jsonl" \
 *     --output-dir "data/Student Training Data" \
 *     --seed 42
 */

import * as path from "node:path";
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
};

type AugmentedJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  source_url?: string;
  augmentation_type: string;
  source_job_id: string;
};

// ── Seeded RNG helper ────────────────────────────────────────────────────────

function pickRandom<T>(arr: T[], rng: () => number): T {
  return arr[Math.floor(rng() * arr.length)]!;
}

function pickN<T>(arr: T[], n: number, rng: () => number): T[] {
  const shuffled = shuffleArray(arr, rng);
  return shuffled.slice(0, n);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SALARY INJECTION
// ═══════════════════════════════════════════════════════════════════════════════

// Salary ranges per tier [loK, hiK] (in thousands)
const SALARY_TIERS: { name: string; ranges: [number, number][] }[] = [
  { name: "high",     ranges: [[100, 130], [105, 125], [110, 140], [115, 145]] },
  { name: "mid_high", ranges: [[75, 90], [78, 92], [80, 95], [85, 98]] },
  { name: "mid",      ranges: [[55, 68], [58, 70], [60, 72], [62, 74]] },
  { name: "low",      ranges: [[28, 35], [30, 38], [32, 40], [25, 33]] },
];

// Formats that take (lo, hi) in full pounds (e.g. 60000, 80000)
type RangeFormatter = (lo: number, hi: number) => string;

const RANGE_FORMATS: RangeFormatter[] = [
  // User-specified core formats
  (lo, hi) => `£${lo.toLocaleString("en-GB")} - £${hi.toLocaleString("en-GB")}`,
  (lo, hi) => `£${lo / 1000}k - £${hi / 1000}k`,
  (lo, hi) => `£${lo / 1000}K\u2013£${hi / 1000}K`,
  (lo, hi) => `£${lo / 1000}k to £${hi / 1000}k`,
  (lo, hi) => `£${lo.toLocaleString("en-GB")} to £${hi.toLocaleString("en-GB")}`,
  (lo, hi) => `£${lo.toLocaleString("en-GB")}\u2013£${hi.toLocaleString("en-GB")} per year`,
  (lo, hi) => `£${lo / 1000}k\u2013${hi / 1000}k`,
  (lo, hi) => `${lo / 1000}k\u2013${hi / 1000}k`,
  (lo, hi) => `£${lo.toLocaleString("en-GB")}-£${hi.toLocaleString("en-GB")} per annum`,
  (lo, hi) => `£${lo.toLocaleString("en-GB")} - £${hi.toLocaleString("en-GB")} p.a.`,
  // Single value formats (use hi as the displayed value)
  (_lo, hi) => `£${hi.toLocaleString("en-GB")}`,
  (_lo, hi) => `£${hi / 1000}k`,
  // "From £X" / "£X+" formats (use lo as the displayed value)
  (lo, _hi) => `From £${lo.toLocaleString("en-GB")}`,
  (lo, _hi) => `£${lo / 1000}k+`,
  (lo, _hi) => `£${lo.toLocaleString("en-GB")}+`,
  // "DOE" suffix
  (lo, hi) => `£${lo / 1000}k\u2013£${hi / 1000}k DOE`,
  (lo, hi) => `£${lo.toLocaleString("en-GB")} - £${hi.toLocaleString("en-GB")} DOE`,
  // "OTE" prefix (on-target earnings)
  (_lo, hi) => `OTE £${hi.toLocaleString("en-GB")}`,
  (_lo, hi) => `OTE £${hi / 1000}k`,
];

// Static salary strings (no range needed)
const CEILING_FORMATS = [
  "Up to £90,000", "up to £80k", "Up to £110K", "Up to £95,000",
  "up to £120k", "Up to £75K", "Up to £100,000",
];

const USD_FORMATS = [
  "$100,000 - $150,000", "$120,000 - $180,000", "$80,000 - $120,000",
  "$90,000 - $140,000", "$110,000 - $160,000",
];

const EUR_FORMATS = [
  "€70,000 - €90,000", "€60,000 - €80,000", "€80,000 - €110,000",
];

const DAY_RATE_FORMATS = [
  "£500/day", "£600/day", "£450/day", "£700/day",
  "£550 per day", "£650 per day",
];

const VAGUE_FORMATS = [
  "Competitive salary", "Competitive package", "Market rate",
  "Competitive base + equity", "Attractive compensation package",
];

const SALARY_WRAPPERS = [
  (s: string) => `\nSalary: ${s}\n`,
  (s: string) => `\nCompensation: ${s}\n`,
  (s: string) => `\nThe salary for this role is ${s}.\n`,
  (s: string) => `\nWe offer a salary of ${s}.\n`,
  (s: string) => `\n${s}\n`,
];

/**
 * Find paragraph break positions in JD text for salary injection.
 */
function findInjectionPositions(jdText: string): number[] {
  const positions: number[] = [0]; // top
  let idx = 0;
  while (idx < jdText.length) {
    const next = jdText.indexOf("\n\n", idx);
    if (next === -1) break;
    positions.push(next);
    idx = next + 2;
  }
  positions.push(jdText.length); // bottom
  return positions;
}

function injectSalary(
  jdText: string,
  salaryLine: string,
  position: "top" | "mid" | "bottom",
): string {
  const breaks = findInjectionPositions(jdText);

  if (position === "top") {
    // After first paragraph break, or at position 0
    const pos = breaks.length > 1 ? breaks[1]! : 0;
    return jdText.slice(0, pos) + salaryLine + jdText.slice(pos);
  }

  if (position === "mid") {
    // Around 40-60% into the text
    const midBreaks = breaks.filter(
      (p) => p > jdText.length * 0.3 && p < jdText.length * 0.7,
    );
    if (midBreaks.length > 0) {
      const pos = midBreaks[Math.floor(midBreaks.length / 2)]!;
      return jdText.slice(0, pos) + salaryLine + jdText.slice(pos);
    }
    // Fallback: insert at 50%
    const half = Math.floor(jdText.length / 2);
    return jdText.slice(0, half) + salaryLine + jdText.slice(half);
  }

  // bottom: before last paragraph
  const pos = breaks.length > 1 ? breaks[breaks.length - 2]! : jdText.length;
  return jdText.slice(0, pos) + salaryLine + jdText.slice(pos);
}

function generateSalaryAugmentations(
  jobs: LabeledJob[],
  rng: () => number,
  maxJobs: number,
): AugmentedJob[] {
  // Select UK jobs with comp=0
  const candidates = jobs.filter((j) => j.loc >= 10 && j.comp === 0);
  const selected = pickN(candidates, maxJobs, rng);

  console.log(
    `  Salary injection: ${selected.length} source jobs (from ${candidates.length} candidates)`,
  );

  const results: AugmentedJob[] = [];
  const positions: Array<"top" | "mid" | "bottom"> = ["top", "mid", "bottom"];

  // Each job gets 3-4 salary variants
  const tierNames = ["high", "mid_high", "mid", "low"];
  // Also add some comp=0 variants (ceiling, USD, vague)
  const comp0Types = ["ceiling", "usd", "eur", "vague", "dayrate"];

  for (const job of selected) {
    // Pick 3-4 distinct tiers per job
    const numVariants = rng() > 0.5 ? 4 : 3;
    const chosenTiers = pickN(tierNames, Math.min(numVariants, tierNames.length), rng);

    // Occasionally add a comp=0 variant too
    const addComp0 = rng() > 0.6;
    if (addComp0) {
      chosenTiers.push(pickRandom(comp0Types, rng));
    }

    for (let i = 0; i < chosenTiers.length; i++) {
      const tierName = chosenTiers[i]!;
      let salaryString: string;

      if (tierName === "ceiling") {
        salaryString = pickRandom(CEILING_FORMATS, rng);
      } else if (tierName === "usd") {
        salaryString = pickRandom(USD_FORMATS, rng);
      } else if (tierName === "eur") {
        salaryString = pickRandom(EUR_FORMATS, rng);
      } else if (tierName === "vague") {
        salaryString = pickRandom(VAGUE_FORMATS, rng);
      } else if (tierName === "dayrate") {
        salaryString = pickRandom(DAY_RATE_FORMATS, rng);
      } else {
        // Range tier
        const tier = SALARY_TIERS.find((t) => t.name === tierName)!;
        const [loK, hiK] = pickRandom(tier.ranges, rng);
        const lo = loK * 1000;
        const hi = hiK * 1000;
        const formatter = pickRandom(RANGE_FORMATS, rng);
        salaryString = formatter(lo, hi);
      }

      const wrapper = pickRandom(SALARY_WRAPPERS, rng);
      const salaryLine = wrapper(salaryString);
      const position = pickRandom(positions, rng);
      const augmentedJd = injectSalary(job.jd_text, salaryLine, position);

      results.push({
        job_id: `${job.job_id}_sal_${i}`,
        title: job.title,
        company: job.company,
        location: job.location,
        jd_text: augmentedJd,
        source_url: job.source_url,
        augmentation_type: `salary_${tierName}`,
        source_job_id: job.job_id,
      });
    }
  }

  return results;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTRASTIVE PAIRS
// ═══════════════════════════════════════════════════════════════════════════════

const NON_UK_LOCATIONS = [
  "Paris, Île-de-France, France",
  "Berlin, Germany",
  "San Francisco Bay Area",
  "New York, New York, United States",
  "Toronto, Ontario, Canada",
  "Amsterdam, North Holland, Netherlands",
  "Singapore",
  "Sydney, New South Wales, Australia",
  "Bangalore, Karnataka, India",
  "Dublin, County Dublin, Ireland",
];

const TITLE_DOWNGRADES: [RegExp, string][] = [
  [/\bSenior\b/i, "Junior"],
  [/\bStaff\b/i, ""],
  [/\bPrincipal\b/i, ""],
  [/\bLead\b/i, ""],
  [/\bTech Lead\b/i, ""],
  [/\bHead of\b/i, ""],
  [/\bSr\.?\s/i, "Jr "],
  [/\bSnr\.?\s/i, "Jr "],
  [/\bDistinguished\b/i, ""],
  [/\bFounding\b/i, ""],
];

const TECH_REPLACEMENTS: [RegExp, string][] = [
  [/\bNode\.?js\b/gi, "Spring Boot"],
  [/\bNodeJS\b/gi, "Spring Boot"],
  [/\bTypeScript\b/gi, "Kotlin"],
  [/\bJavaScript\b/gi, "Java"],
];

function generateContrastivePairs(
  jobs: LabeledJob[],
  rng: () => number,
  maxJobs: number,
): AugmentedJob[] {
  // Select UK jobs with decent role AND tech scores
  const candidates = jobs.filter(
    (j) => j.loc >= 10 && j.role >= 15 && j.tech >= 10,
  );
  const selected = pickN(candidates, maxJobs, rng);

  console.log(
    `  Contrastive pairs: ${selected.length} source jobs (from ${candidates.length} candidates)`,
  );

  const results: AugmentedJob[] = [];

  for (const job of selected) {
    // Variant 1: Location flip (UK → non-UK)
    results.push({
      job_id: `${job.job_id}_con_loc`,
      title: job.title,
      company: job.company,
      location: pickRandom(NON_UK_LOCATIONS, rng),
      jd_text: job.jd_text,
      source_url: job.source_url,
      augmentation_type: "contrastive_loc",
      source_job_id: job.job_id,
    });

    // Variant 2: Title downgrade (senior → junior)
    let downgraded = job.title;
    for (const [pattern, replacement] of TITLE_DOWNGRADES) {
      if (pattern.test(downgraded)) {
        downgraded = downgraded.replace(pattern, replacement).replace(/\s+/g, " ").trim();
        break; // Only apply first match
      }
    }
    if (downgraded !== job.title) {
      results.push({
        job_id: `${job.job_id}_con_title`,
        title: downgraded,
        company: job.company,
        location: job.location,
        jd_text: job.jd_text,
        source_url: job.source_url,
        augmentation_type: "contrastive_title",
        source_job_id: job.job_id,
      });
    }

    // Variant 3: Tech swap (Node.js → Spring Boot, TypeScript → Kotlin)
    let swappedJd = job.jd_text;
    for (const [pattern, replacement] of TECH_REPLACEMENTS) {
      swappedJd = swappedJd.replace(pattern, replacement);
    }
    if (swappedJd !== job.jd_text) {
      results.push({
        job_id: `${job.job_id}_con_tech`,
        title: job.title,
        company: job.company,
        location: job.location,
        jd_text: swappedJd,
        source_url: job.source_url,
        augmentation_type: "contrastive_tech",
        source_job_id: job.job_id,
      });
    }
  }

  return results;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOCATION FORMAT VARIATION
// ═══════════════════════════════════════════════════════════════════════════════

const LONDON_VARIANTS = [
  "London, England, United Kingdom",
  "London, UK",
  "London",
  "Greater London Area",
  "London (Hybrid)",
  "City Of London, England, United Kingdom",
  "Greater London, England, United Kingdom",
  "London, England, United Kingdom (Hybrid)",
  "London Area, United Kingdom",
];

const UK_CITY_VARIANTS: Record<string, string[]> = {
  manchester: [
    "Manchester, England, United Kingdom",
    "Manchester, UK",
    "Greater Manchester",
    "Manchester (Hybrid)",
    "Manchester Area, United Kingdom",
  ],
  edinburgh: [
    "Edinburgh, Scotland, United Kingdom",
    "Edinburgh, UK",
    "Edinburgh",
    "Edinburgh (Hybrid)",
  ],
  bristol: [
    "Bristol, England, United Kingdom",
    "Bristol, UK",
    "Bristol",
    "Greater Bristol Area, United Kingdom",
  ],
  birmingham: [
    "Birmingham, England, United Kingdom",
    "Birmingham, UK",
    "Birmingham",
    "Birmingham (Hybrid)",
  ],
  leeds: [
    "Leeds, England, United Kingdom",
    "Leeds, UK",
    "Leeds",
    "Leeds (Hybrid)",
  ],
  glasgow: [
    "Glasgow, Scotland, United Kingdom",
    "Glasgow, UK",
    "Glasgow",
    "Glasgow City, Scotland, United Kingdom",
  ],
  cambridge: [
    "Cambridge, England, United Kingdom",
    "Cambridge, UK",
    "Cambridge",
  ],
  cardiff: [
    "Cardiff, Wales, United Kingdom",
    "Cardiff, UK",
    "Cardiff",
  ],
};

function findCityGroup(location: string): string[] | null {
  const lower = location.toLowerCase();
  if (lower.includes("london")) return LONDON_VARIANTS;
  for (const [city, variants] of Object.entries(UK_CITY_VARIANTS)) {
    if (lower.includes(city)) return variants;
  }
  return null;
}

function generateLocationVariants(
  jobs: LabeledJob[],
  rng: () => number,
  targetCount: number,
): AugmentedJob[] {
  // Split by location type
  const london = jobs.filter((j) => j.loc === 25 && j.location.toLowerCase().includes("london"));
  const ukNonLondon = jobs.filter((j) => j.loc === 10);
  const nonUk = jobs.filter((j) => j.loc === -50);

  const selectedLondon = pickN(london, Math.min(25, Math.ceil(targetCount * 0.5)), rng);
  const selectedUk = pickN(ukNonLondon, Math.min(15, Math.ceil(targetCount * 0.3)), rng);
  const selectedNonUk = pickN(nonUk, Math.min(10, Math.ceil(targetCount * 0.2)), rng);
  const allSelected = [...selectedLondon, ...selectedUk, ...selectedNonUk];

  console.log(
    `  Location variants: ${allSelected.length} source jobs ` +
    `(${selectedLondon.length} London, ${selectedUk.length} UK, ${selectedNonUk.length} non-UK)`,
  );

  const results: AugmentedJob[] = [];

  for (const job of allSelected) {
    const variants = findCityGroup(job.location);
    if (!variants) {
      // Non-UK or unrecognised city: just reformat slightly
      // Add "(On-site)" or "(Remote)" suffix variation
      const suffix = rng() > 0.5 ? " (On-site)" : " (Remote)";
      const newLoc = job.location.includes("(")
        ? job.location.replace(/\s*\([^)]+\)/, suffix)
        : job.location + suffix;

      if (newLoc !== job.location) {
        results.push({
          job_id: `${job.job_id}_locvar_0`,
          title: job.title,
          company: job.company,
          location: newLoc,
          jd_text: job.jd_text,
          source_url: job.source_url,
          augmentation_type: "location_format",
          source_job_id: job.job_id,
        });
      }
      continue;
    }

    // Pick a variant different from current
    const currentLower = job.location.toLowerCase().trim();
    const alternatives = variants.filter(
      (v) => v.toLowerCase().trim() !== currentLower,
    );

    if (alternatives.length > 0) {
      const newLoc = pickRandom(alternatives, rng);
      results.push({
        job_id: `${job.job_id}_locvar_0`,
        title: job.title,
        company: job.company,
        location: newLoc,
        jd_text: job.jd_text,
        source_url: job.source_url,
        augmentation_type: "location_format",
        source_job_id: job.job_id,
      });
    }
  }

  return results;
}

// ═══════════════════════════════════════════════════════════════════════════════
// JD TRUNCATION
// ═══════════════════════════════════════════════════════════════════════════════

function generateTruncatedJds(
  jobs: LabeledJob[],
  rng: () => number,
  maxJobs: number,
): AugmentedJob[] {
  const candidates = jobs.filter((j) => j.jd_text.length > 500);
  const selected = pickN(candidates, maxJobs, rng);

  console.log(
    `  JD truncation: ${selected.length} source jobs (from ${candidates.length} candidates with JD > 500 chars)`,
  );

  const results: AugmentedJob[] = [];

  for (const job of selected) {
    // 50% truncation
    const half = Math.floor(job.jd_text.length * 0.5);
    results.push({
      job_id: `${job.job_id}_trunc50`,
      title: job.title,
      company: job.company,
      location: job.location,
      jd_text: job.jd_text.slice(0, half) + " [truncated]",
      source_url: job.source_url,
      augmentation_type: "truncation_50",
      source_job_id: job.job_id,
    });

    // 25% truncation
    const quarter = Math.floor(job.jd_text.length * 0.25);
    results.push({
      job_id: `${job.job_id}_trunc25`,
      title: job.title,
      company: job.company,
      location: job.location,
      jd_text: job.jd_text.slice(0, quarter) + " [truncated]",
      source_url: job.source_url,
      augmentation_type: "truncation_25",
      source_job_id: job.job_id,
    });
  }

  return results;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ??
    "data/Student Training Data/teacher_labeled_500.jsonl";
  const outputDir =
    getStringArg(args, "output-dir") ?? "data/Student Training Data";
  const seed = getNumberArg(args, "seed") ?? 42;

  // Augmentation limits
  const maxSalaryJobs = getNumberArg(args, "max-salary") ?? 60;
  const maxContrastiveJobs = getNumberArg(args, "max-contrastive") ?? 30;
  const locationTargetCount = getNumberArg(args, "max-location") ?? 50;
  const maxTruncationJobs = getNumberArg(args, "max-truncation") ?? 50;

  const rng = mulberry32(seed);

  const jobs = await readJsonlFile<LabeledJob>(inputPath);
  console.log(`Loaded ${jobs.length} labeled jobs from ${inputPath}`);
  console.log(`Seed: ${seed}\n`);

  // Generate all augmentation types
  console.log("Generating augmented data...\n");

  const salary = generateSalaryAugmentations(jobs, rng, maxSalaryJobs);
  const contrastive = generateContrastivePairs(jobs, rng, maxContrastiveJobs);
  const location = generateLocationVariants(jobs, rng, locationTargetCount);
  const truncated = generateTruncatedJds(jobs, rng, maxTruncationJobs);

  // Write output files
  const outputs = [
    { name: "salary_augmented.jsonl", data: salary },
    { name: "contrastive_pairs.jsonl", data: contrastive },
    { name: "location_variants.jsonl", data: location },
    { name: "truncated_jds.jsonl", data: truncated },
  ];

  console.log("\nWriting output files...\n");

  for (const { name, data } of outputs) {
    const filePath = path.join(outputDir, name);
    await writeJsonlFile(filePath, data);
    console.log(`  ${name}: ${data.length} records`);
  }

  const total = salary.length + contrastive.length + location.length + truncated.length;
  console.log(`\n  Total augmented: ${total} records (ready for teacher labeling)`);

  // Print augmentation type breakdown for salary
  const salaryTypes: Record<string, number> = {};
  for (const j of salary) {
    const t = j.augmentation_type;
    salaryTypes[t] = (salaryTypes[t] ?? 0) + 1;
  }
  console.log("\nSalary injection breakdown:");
  for (const [type, count] of Object.entries(salaryTypes).sort()) {
    console.log(`  ${type}: ${count}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

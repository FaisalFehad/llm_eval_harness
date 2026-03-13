/**
 * Build V10 training data from existing labeled pools (no manual edits).
 *
 * Goals:
 * - Reduce bad_fit prior drift seen in v9
 * - Increase good_fit/maybe coverage to match eval prior more closely
 * - Keep hard bad_fit examples, down-weight trivially-negative source buckets
 * - Add v7 supplemental examples deterministically when v9 is under target
 *
 * Usage:
 *   npx tsx src/cli/build-v10-training-set.ts \
 *     --v9-input data/v9/train_labeled.jsonl \
 *     --v7-input data/v7/train_labeled.jsonl \
 *     --output data/v10/train_labeled.jsonl \
 *     --report data/v10/build_report.json \
 *     --target-size 1000 \
 *     --seed 42
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type Label = "good_fit" | "maybe" | "bad_fit";
type SourceBucket = "it_support_scrape" | "low_salary" | "patch_scrape" | "node_variants" | "other";

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  job_location: string;
  jd_text: string;
  loc: string;
  arr: string;
  sen: string;
  tech: string[];
  comp: string;
  score: number;
  label: string;
  source_file?: string;
  [key: string]: unknown;
};

const GOOD_TARGET_RATIO = 0.238;
const MAYBE_TARGET_RATIO = 0.238;

const BAD_SOURCE_CAPS: Partial<Record<SourceBucket, number>> = {
  it_support_scrape: 40,
  low_salary: 25,
  patch_scrape: 20,
  node_variants: 25,
};

function normalizeText(value: string | null | undefined): string {
  return (value ?? "").toLowerCase().replace(/\s+/g, " ").trim();
}

function contentKey(job: LabeledJob): string {
  const jdHead = normalizeText(job.jd_text).slice(0, 500);
  return [
    normalizeText(job.title),
    normalizeText(job.company),
    normalizeText(job.job_location),
    jdHead,
  ].join("|");
}

function hashString(input: string): number {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededShuffle<T>(rows: T[], seed: number): T[] {
  const out = [...rows];
  let state = seed | 0;
  const rand = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

function rankAndSort(
  rows: LabeledJob[],
  scorer: (job: LabeledJob) => number,
  seed: number,
): LabeledJob[] {
  return [...rows]
    .map((job) => ({
      job,
      score: scorer(job),
      tie: hashString(`${seed}|${job.job_id}|${job.title}|${job.company}`),
    }))
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.tie - b.tie;
    })
    .map((x) => x.job);
}

function toLabel(value: string): Label | null {
  if (value === "good_fit" || value === "maybe" || value === "bad_fit") return value;
  return null;
}

function splitByLabel(rows: LabeledJob[]): Record<Label, LabeledJob[]> {
  const grouped: Record<Label, LabeledJob[]> = {
    good_fit: [],
    maybe: [],
    bad_fit: [],
  };
  for (const row of rows) {
    const label = toLabel(row.label);
    if (label) grouped[label].push(row);
  }
  return grouped;
}

function sourceBucket(sourceFile: string | undefined): SourceBucket {
  const source = sourceFile ?? "";
  if (source === "it_support_scrape") return "it_support_scrape";
  if (source.startsWith("new_jobs_run_low_salary")) return "low_salary";
  if (source === "patch_scrape") return "patch_scrape";
  if (source.startsWith("node_variants_v")) return "node_variants";
  return "other";
}

function hasTrackedTech(job: LabeledJob): boolean {
  return Array.isArray(job.tech) && job.tech.some((token) => token !== "OOS");
}

function hasAiMl(job: LabeledJob): boolean {
  return Array.isArray(job.tech) && job.tech.includes("AI_ML");
}

function badPriority(job: LabeledJob): number {
  let score = 0;
  const numericScore = Number.isFinite(job.score) ? job.score : 0;
  if (numericScore >= 30) score += 4;
  if (numericScore >= 40) score += 2;
  if (job.comp === "RANGE_55_74K") score += 3;
  if (job.comp === "RANGE_75_99K") score += 1;
  if (hasTrackedTech(job)) score += 2;
  if (hasAiMl(job)) score += 2;
  if (job.loc === "IN_LONDON") score += 1;
  if (job.arr === "HYBRID" || job.arr === "REMOTE") score += 1;
  return score;
}

function supplementPriority(job: LabeledJob): number {
  let score = 0;
  const numericScore = Number.isFinite(job.score) ? job.score : 0;
  if (job.label === "good_fit") score += 3;
  if (job.comp === "RANGE_55_74K") score += 3;
  if (job.loc === "IN_LONDON") score += 2;
  if (hasAiMl(job)) score += 3;
  if (hasTrackedTech(job)) score += 1;
  if (numericScore >= 55) score += 1;
  return score;
}

function cloneForOversample(job: LabeledJob, suffix: string, sourceTag: string): LabeledJob {
  const copy: LabeledJob = { ...job };
  copy.job_id = `${job.job_id}__${suffix}`;
  copy.source_file = copy.source_file ? `${copy.source_file}__${sourceTag}` : sourceTag;
  return copy;
}

function countBy(rows: LabeledJob[], keyFn: (row: LabeledJob) => string): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const row of rows) {
    const key = keyFn(row);
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}

function summarize(rows: LabeledJob[]): {
  size: number;
  labels: Record<Label, number>;
  comp: Record<string, number>;
  loc: Record<string, number>;
  source: Record<string, number>;
  tech: Record<string, number>;
} {
  const labels: Record<Label, number> = {
    good_fit: 0,
    maybe: 0,
    bad_fit: 0,
  };
  const comp = countBy(rows, (r) => r.comp || "UNK");
  const loc = countBy(rows, (r) => r.loc || "UNK");
  const source = countBy(rows, (r) => r.source_file ?? "unknown");
  const tech: Record<string, number> = {};

  for (const row of rows) {
    const label = toLabel(row.label);
    if (label) labels[label] += 1;
    for (const token of row.tech ?? []) {
      tech[token] = (tech[token] ?? 0) + 1;
    }
  }

  return { size: rows.length, labels, comp, loc, source, tech };
}

function topEntries(counts: Record<string, number>, limit: number): Record<string, number> {
  return Object.fromEntries(
    Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit),
  );
}

function pct(part: number, total: number): number {
  if (total === 0) return 0;
  return Number(((part / total) * 100).toFixed(1));
}

async function main(): Promise<void> {
  const args = parseArgs();
  const v9Input = getStringArg(args, "v9-input") ?? "data/v9/train_labeled.jsonl";
  const v7Input = getStringArg(args, "v7-input") ?? "data/v7/train_labeled.jsonl";
  const outputPath = getStringArg(args, "output") ?? "data/v10/train_labeled.jsonl";
  const reportPath = getStringArg(args, "report") ?? "data/v10/build_report.json";
  const seed = getNumberArg(args, "seed") ?? 42;
  const targetSize = getNumberArg(args, "target-size") ?? 1000;

  const v9Rows = await readJsonlFile<LabeledJob>(v9Input);
  const v7Rows = await readJsonlFile<LabeledJob>(v7Input);
  if (v9Rows.length === 0) throw new Error(`No rows in ${v9Input}`);
  if (v7Rows.length === 0) throw new Error(`No rows in ${v7Input}`);

  const v9ByKey = new Map<string, LabeledJob>();
  let dedupedV9 = 0;
  for (const row of v9Rows) {
    const key = contentKey(row);
    if (v9ByKey.has(key)) {
      dedupedV9 += 1;
      continue;
    }
    v9ByKey.set(key, row);
  }
  const v9Unique = [...v9ByKey.values()];
  const v9Keys = new Set(v9ByKey.keys());

  const v7CandidateByKey = new Map<string, LabeledJob>();
  let overlapWithV9 = 0;
  for (const row of v7Rows) {
    const key = contentKey(row);
    if (v9Keys.has(key)) {
      overlapWithV9 += 1;
      continue;
    }
    if (!v7CandidateByKey.has(key)) {
      v7CandidateByKey.set(key, row);
    }
  }
  const v7Candidates = [...v7CandidateByKey.values()];

  const v9ByLabel = splitByLabel(v9Unique);
  const v7ByLabel = splitByLabel(v7Candidates);

  const v9BadBuckets: Record<SourceBucket, LabeledJob[]> = {
    it_support_scrape: [],
    low_salary: [],
    patch_scrape: [],
    node_variants: [],
    other: [],
  };

  for (const row of v9ByLabel.bad_fit) {
    v9BadBuckets[sourceBucket(row.source_file)].push(row);
  }

  const cappedBadV9: LabeledJob[] = [];
  const cappedDropCounts: Record<SourceBucket, number> = {
    it_support_scrape: 0,
    low_salary: 0,
    patch_scrape: 0,
    node_variants: 0,
    other: 0,
  };

  for (const bucket of Object.keys(v9BadBuckets) as SourceBucket[]) {
    const pool = v9BadBuckets[bucket];
    const cap = BAD_SOURCE_CAPS[bucket];
    if (cap === undefined || pool.length <= cap) {
      cappedBadV9.push(...pool);
      continue;
    }
    const ranked = rankAndSort(pool, badPriority, seed + hashString(bucket));
    cappedBadV9.push(...ranked.slice(0, cap));
    cappedDropCounts[bucket] = pool.length - cap;
  }

  const rankedBadV9 = rankAndSort(cappedBadV9, badPriority, seed + 13);

  const targetGood = Math.round(targetSize * GOOD_TARGET_RATIO);
  const targetMaybe = Math.round(targetSize * MAYBE_TARGET_RATIO);
  const targetBad = Math.max(0, targetSize - targetGood - targetMaybe);

  let selectedGood = [...v9ByLabel.good_fit];
  if (selectedGood.length < targetGood) {
    const needed = targetGood - selectedGood.length;
    const extras = rankAndSort(v7ByLabel.good_fit, supplementPriority, seed + 101).slice(0, needed);
    selectedGood.push(...extras);
  }
  if (selectedGood.length > targetGood) {
    selectedGood = rankAndSort(selectedGood, supplementPriority, seed + 108).slice(0, targetGood);
  }
  if (selectedGood.length < targetGood) {
    const needed = targetGood - selectedGood.length;
    const pool = rankAndSort(
      [...selectedGood, ...v9ByLabel.good_fit, ...v7ByLabel.good_fit],
      supplementPriority,
      seed + 109,
    );
    if (pool.length > 0) {
      for (let i = 0; i < needed; i += 1) {
        const base = pool[i % pool.length]!;
        selectedGood.push(cloneForOversample(base, `v10os_good_${i + 1}`, "v10_oversample_good"));
      }
    }
  }

  let selectedMaybe = [...v9ByLabel.maybe];
  if (selectedMaybe.length < targetMaybe) {
    const needed = targetMaybe - selectedMaybe.length;
    const extras = rankAndSort(v7ByLabel.maybe, supplementPriority, seed + 202).slice(0, needed);
    selectedMaybe.push(...extras);
  }
  if (selectedMaybe.length > targetMaybe) {
    selectedMaybe = rankAndSort(selectedMaybe, supplementPriority, seed + 209).slice(0, targetMaybe);
  }
  if (selectedMaybe.length < targetMaybe) {
    const needed = targetMaybe - selectedMaybe.length;
    const pool = rankAndSort(
      [...selectedMaybe, ...v9ByLabel.maybe, ...v7ByLabel.maybe],
      supplementPriority,
      seed + 210,
    );
    if (pool.length > 0) {
      for (let i = 0; i < needed; i += 1) {
        const base = pool[i % pool.length]!;
        selectedMaybe.push(cloneForOversample(base, `v10os_maybe_${i + 1}`, "v10_oversample_maybe"));
      }
    }
  }

  let selectedBad = [...rankedBadV9];
  if (selectedBad.length > targetBad) {
    selectedBad = selectedBad.slice(0, targetBad);
  } else if (selectedBad.length < targetBad) {
    const needed = targetBad - selectedBad.length;
    const extras = rankAndSort(v7ByLabel.bad_fit, badPriority, seed + 303).slice(0, needed);
    selectedBad.push(...extras);
  }

  const selected = [...selectedGood, ...selectedMaybe, ...selectedBad];
  const selectedKeys = new Set(selected.map(contentKey));

  if (selected.length < targetSize) {
    const remainingSupp = rankAndSort(
      v7Candidates.filter((row) => !selectedKeys.has(contentKey(row))),
      supplementPriority,
      seed + 404,
    );
    for (const row of remainingSupp) {
      if (selected.length >= targetSize) break;
      const key = contentKey(row);
      if (selectedKeys.has(key)) continue;
      selected.push(row);
      selectedKeys.add(key);
    }
  }

  if (selected.length < targetSize) {
    const remainingV9 = rankAndSort(
      v9Unique.filter((row) => !selectedKeys.has(contentKey(row))),
      supplementPriority,
      seed + 505,
    );
    for (const row of remainingV9) {
      if (selected.length >= targetSize) break;
      const key = contentKey(row);
      if (selectedKeys.has(key)) continue;
      selected.push(row);
      selectedKeys.add(key);
    }
  }

  const idCounts = new Map<string, number>();
  const finalized: LabeledJob[] = selected.map((row, i) => {
    const copy = { ...row };
    const baseId = (copy.job_id ?? "").trim() || `auto_v10_${i + 1}`;
    const seen = idCounts.get(baseId) ?? 0;
    if (seen === 0) {
      copy.job_id = baseId;
    } else {
      copy.job_id = `${baseId}__v10_${seen + 1}`;
    }
    idCounts.set(baseId, seen + 1);
    return copy;
  });

  const finalShuffled = seededShuffle(finalized, seed);

  await writeJsonlFile(outputPath, finalShuffled);

  const reportDir = path.dirname(reportPath);
  fs.mkdirSync(reportDir, { recursive: true });

  const finalSummary = summarize(finalShuffled);
  const report = {
    timestamp: new Date().toISOString(),
    seed,
    target_size: targetSize,
    selected_size: finalShuffled.length,
    target_counts: {
      good_fit: targetGood,
      maybe: targetMaybe,
      bad_fit: targetBad,
    },
    input_counts: {
      v9_raw: v9Rows.length,
      v9_unique: v9Unique.length,
      v9_deduped: dedupedV9,
      v7_raw: v7Rows.length,
      v7_overlap_with_v9: overlapWithV9,
      v7_candidates: v7Candidates.length,
    },
    bad_source_cap_drops: cappedDropCounts,
    label_distribution: {
      good_fit: {
        count: finalSummary.labels.good_fit,
        pct: pct(finalSummary.labels.good_fit, finalSummary.size),
      },
      maybe: {
        count: finalSummary.labels.maybe,
        pct: pct(finalSummary.labels.maybe, finalSummary.size),
      },
      bad_fit: {
        count: finalSummary.labels.bad_fit,
        pct: pct(finalSummary.labels.bad_fit, finalSummary.size),
      },
    },
    top_source_distribution: topEntries(finalSummary.source, 12),
    top_comp_distribution: topEntries(finalSummary.comp, 10),
    top_tech_distribution: topEntries(finalSummary.tech, 10),
    top_loc_distribution: topEntries(finalSummary.loc, 10),
  };

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");

  console.log(`Wrote v10 training set: ${outputPath}`);
  console.log(`Wrote report: ${reportPath}`);
  console.log(
    `Final label mix: good_fit=${report.label_distribution.good_fit.count} (${report.label_distribution.good_fit.pct}%), ` +
      `maybe=${report.label_distribution.maybe.count} (${report.label_distribution.maybe.pct}%), ` +
      `bad_fit=${report.label_distribution.bad_fit.count} (${report.label_distribution.bad_fit.pct}%)`,
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

/**
 * Build V11 training data with teacher-filtered v9, sentence-level cleaning,
 * balanced label/feature distributions, and dual schema outputs.
 *
 * Usage:
 *   npx tsx src/cli/build-v11.ts \
 *     --v7 data/v7/train_labeled.jsonl \
 *     --v9 data/v9/train_labeled.jsonl \
 *     --v9-teacher-preds data/v9/train_labeled_v7preds.jsonl \
 *     --out-dir data/v11 \
 *     --target-size 1000 \
 *     --seed 42
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs, getNumberArg, getStringArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type Label = "good_fit" | "maybe" | "bad_fit";

type LabeledRow = {
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
  label: Label;
  source_file?: string;
  loc_raw?: string | null;
  arr_raw?: string | null;
  tech_raw?: string | null;
  comp_raw?: string | null;
};

type TeacherPred = {
  job_index: number;
  golden_label: Label;
  golden_score: number;
  pred_label: Label | null;
  pred_score: number | null;
  parse_fail?: boolean;
  invalid_token?: boolean;
};

const DROP_PATTERNS = [
  /equal opportunity/i,
  /diversity/i,
  /inclusion/i,
  /privacy policy/i,
  /cookies?/i,
  /terms and conditions/i,
  /how to apply/i,
  /apply now/i,
  /about (the )?company/i,
  /benefits include/i,
  /\bperks\b/i,
  /vision and values/i,
];

const FACT_PATTERNS = [
  /£|€|\$|salary|compensation|per annum|p\.?a\.?/i,
  /\bremote\b|\bhybrid\b|\bon[- ]?site\b|\bin[- ]?office\b/i,
  /\blondon\b|\buk\b|\bunited kingdom\b|\bbased in\b/i,
  /\bnode(\.js|js)?\b|\breact(\.js|js)?\b|\btypescript\b|\bjavascript\b|\bai\b|\bmachine learning\b|\bml\b/i,
  /\bjunior\b|\bgraduate\b|\bentry[- ]?level\b|\bsenior\b|\blead\b|\bstaff\b|\bprincipal\b|\bmanager\b|\bintern(ship)?\b/i,
  /\baws\b|\bpython\b|\bjava\b|\bdotnet\b|\bc\+\+\b|\bphp\b|\bruby\b|\bsnowflake\b|\bgo(lang)?\b/i, // keep non-target tech as evidence
];

const NON_TARGET_TECH = /\bpython\b|\bjava\b|\bdotnet\b|\bc#\b|\bc\+\+\b|\bphp\b|\bruby\b|\bgo(lang)?\b|\bscala\b|\bkotlin\b|\bswift\b|\bvue\b|\bangular\b|\bterraform\b|\bdocker\b|\bkubernetes\b|\bsnowflake\b|\bsql\b/i;

const CURRENCY_GBP = /£|\bgbp\b/i;
const CURRENCY_NON_GBP = /\busd\b|\beur\b|\$|€/i;

function splitSentences(text: string): string[] {
  const normalized = text.replace(/\r/g, "\n").replace(/\n+/g, "\n").trim();
  if (!normalized) return [];
  // split on newlines first, else on punctuation.
  const parts = normalized
    .split(/\n/)
    .flatMap((line) => line.split(/(?<=[.!?])\s+/))
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  return Array.from(new Set(parts));
}

function keepSentence(s: string): boolean {
  if (DROP_PATTERNS.some((p) => p.test(s))) return false;
  return FACT_PATTERNS.some((p) => p.test(s));
}

function firstSentenceContaining(sentences: string[], pattern: RegExp): string | null {
  for (const s of sentences) {
    if (pattern.test(s)) return s;
  }
  return null;
}

function buildRawFields(sentences: string[]): {
  loc_raw: string | null;
  arr_raw: string | null;
  tech_raw: string | null;
  comp_raw: string | null;
} {
  const loc_raw =
    firstSentenceContaining(sentences, /\blondon\b|\buk\b|\bunited kingdom\b|\bremote\b|\bhybrid\b|\bon[- ]?site\b|\bin[- ]?office\b/i) ?? null;
  const arr_raw =
    firstSentenceContaining(sentences, /\bremote\b|\bhybrid\b|\bon[- ]?site\b|\bin[- ]?office\b/i) ?? null;
  const tech_raw =
    firstSentenceContaining(sentences, /\bnode(\.js|js)?\b|\breact(\.js|js)?\b|\btypescript\b|\bjavascript\b|\bai\b|\bmachine learning\b|\bml\b/i) ??
    firstSentenceContaining(sentences, NON_TARGET_TECH) ??
    null;
  const comp_raw = firstSentenceContaining(sentences, /£|\$|€|salary|compensation|per annum|p\.?a\.?|k\b/i) ?? null;
  return { loc_raw, arr_raw, tech_raw, comp_raw };
}

function dedupSentences(sentences: string[]): string[] {
  return Array.from(new Set(sentences));
}

function cleanText(jd: string): { sentences: string[]; text: string } {
  const sentences = splitSentences(jd);
  const kept = sentences.filter(keepSentence);
  const clipped = kept.slice(0, 28); // ~2200-2600 chars typically
  const uniq = dedupSentences(clipped);
  return { sentences: uniq, text: uniq.join("\n") };
}

async function readTeacherPreds(pathStr: string): Promise<Map<number, TeacherPred>> {
  const preds = await readJsonlFile<TeacherPred>(pathStr);
  const map = new Map<number, TeacherPred>();
  preds.forEach((p) => map.set(p.job_index, p));
  return map;
}

function teacherPass(row: LabeledRow, pred: TeacherPred | undefined): boolean {
  if (!pred) return false;
  if (pred.parse_fail || pred.invalid_token || pred.pred_label === null || pred.pred_score === null) return false;
  if (row.label === pred.pred_label) return true;
  if (row.label !== "bad_fit" && pred.pred_label !== "bad_fit") {
    const delta = Math.abs(row.score - pred.pred_score);
    return delta <= 15;
  }
  return false;
}

function contentKey(row: LabeledRow): string {
  const head = row.jd_text.toLowerCase().replace(/\s+/g, " ").slice(0, 500);
  return [row.title, row.company, row.job_location, head].map((x) => x.toLowerCase()).join("|");
}

function bucketCounts(rows: LabeledRow[], keyFn: (r: LabeledRow) => string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const k = keyFn(r);
    out[k] = (out[k] ?? 0) + 1;
  }
  return out;
}

function matchesSourceCap(row: LabeledRow, caps: Record<string, number>, current: Record<string, number>): boolean {
  const src = row.source_file ?? "other";
  const key =
    src === "it_support_scrape"
      ? "it_support_scrape"
      : src.startsWith("new_jobs_run_low_salary")
        ? "low_salary"
        : src === "patch_scrape"
          ? "patch_scrape"
          : src.startsWith("node_variants")
            ? "node_variants"
            : "other";
  if (!(key in caps)) return true;
  return (current[key] ?? 0) + 1 <= caps[key]!;
}

function enforceCurrencyRule(row: LabeledRow): boolean {
  const text = row.jd_text;
  const hasGbp = CURRENCY_GBP.test(text);
  const hasNonGbp = CURRENCY_NON_GBP.test(text);
  if (hasNonGbp && !hasGbp) {
    row.comp = "NO_GBP";
    if (row.loc === "IN_LONDON") return false;
  }
  return true;
}

function withinDistribution(row: LabeledRow): boolean {
  const loc = row.loc;
  if (!["IN_LONDON", "REMOTE", "UK_OTHER", "OUTSIDE_UK", "UNK"].includes(loc)) return false;
  if (!Array.isArray(row.tech) || row.tech.length === 0) return false;
  return true;
}

function buildTargets(targetSize: number): Record<Label, number> {
  return {
    good_fit: Math.round(targetSize * 0.30),
    maybe: Math.round(targetSize * 0.30),
    bad_fit: targetSize - Math.round(targetSize * 0.30) - Math.round(targetSize * 0.30),
  };
}

function adjustTargets(
  targets: Record<Label, number>,
  available: Record<string, number>,
): Record<Label, number> {
  const adj: Record<Label, number> = { ...targets };
  let deficit = 0;
  (['good_fit', 'maybe', 'bad_fit'] as Label[]).forEach((lbl) => {
    const avail = available[lbl] ?? 0;
    if (avail < adj[lbl]) {
      deficit += adj[lbl] - avail;
      adj[lbl] = avail;
    }
  });
  if (deficit > 0) {
    const order: Label[] = ['maybe', 'bad_fit', 'good_fit'];
    let remaining = deficit;
    while (remaining > 0) {
      let filled = false;
      for (const lbl of order) {
        const avail = available[lbl] ?? 0;
        if (adj[lbl] < avail) {
          adj[lbl] += 1;
          remaining -= 1;
          filled = true;
          if (remaining === 0) break;
        }
      }
      if (!filled) break;
    }
  }
  return adj;
}

function selectWithCaps(
  rows: LabeledRow[],
  labelTargets: Record<Label, number>,
  targetSize: number,
  seed: number,
): LabeledRow[] {
  const locCaps = { OUTSIDE_UK: Math.floor(targetSize * 0.2), UNK: Math.floor(targetSize * 0.08) };
  const compCaps: Record<string, number> = {
    NO_GBP: Math.floor(targetSize * 0.15),
    BELOW_45K: Math.floor(targetSize * 0.18),
    RANGE_55_74K: Math.floor(targetSize * 0.35),
    RANGE_75_99K: Math.floor(targetSize * 0.35),
    ABOVE_100K: Math.floor(targetSize * 0.18),
  };
  const locCounts: Record<string, number> = {};
  const compCounts: Record<string, number> = {};
  const labelCounts: Record<Label, number> = { good_fit: 0, maybe: 0, bad_fit: 0 };
  const pool = seededShuffle(rows, seed);
  const picked: LabeledRow[] = [];

  const tryAdd = (r: LabeledRow, enforceCaps: boolean): boolean => {
    const lbl = r.label as Label;
    if (labelCounts[lbl] >= labelTargets[lbl]) return false;
    const locOk = !enforceCaps || !(r.loc in locCaps) || (locCounts[r.loc] ?? 0) < locCaps[r.loc]!;
    const compOk = !enforceCaps || !(r.comp in compCaps) || (compCounts[r.comp] ?? 0) < compCaps[r.comp]!;
    if (!locOk || !compOk) return false;
    picked.push(r);
    labelCounts[lbl] += 1;
    locCounts[r.loc] = (locCounts[r.loc] ?? 0) + 1;
    compCounts[r.comp] = (compCounts[r.comp] ?? 0) + 1;
    return true;
  };

  for (const r of pool) {
    if (picked.length >= targetSize) break;
    tryAdd(r, true);
  }
  if (picked.length < targetSize) {
    for (const r of pool) {
      if (picked.length >= targetSize) break;
      tryAdd(r, true);
    }
  }
  return picked;
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

async function main(): Promise<void> {
  const args = parseArgs();
  const v7Path = getStringArg(args, "v7") ?? "data/v7/train_labeled.jsonl";
  const v9Path = getStringArg(args, "v9") ?? "data/v9/train_labeled.jsonl";
  const v9TeacherPath = getStringArg(args, "v9-teacher-preds");
  const outDir = getStringArg(args, "out-dir") ?? "data/v11";
  const targetSize = getNumberArg(args, "target-size") ?? 1000;
  const seed = getNumberArg(args, "seed") ?? 42;

  if (!v9TeacherPath) throw new Error("Missing --v9-teacher-preds");

  fs.mkdirSync(outDir, { recursive: true });

  const v7Rows = await readJsonlFile<LabeledRow>(v7Path);
  const v9Rows = await readJsonlFile<LabeledRow>(v9Path);
  const teacherMap = await readTeacherPreds(v9TeacherPath);

  const all: LabeledRow[] = [];
  const dupKeys = new Set<string>();
  let addedV7 = 0;
  let addedV9 = 0;

  const tryAdd = (row: LabeledRow, preferV7: boolean) => {
    if (!withinDistribution(row)) return;
    if (!enforceCurrencyRule(row)) return;
    const key = contentKey(row);
    if (dupKeys.has(key)) return;
    const { sentences, text } = cleanText(row.jd_text);
    if (sentences.length === 0) return;
    const raw = buildRawFields(sentences);
    const cleaned: LabeledRow = {
      ...row,
      jd_text: text,
      loc_raw: raw.loc_raw,
      arr_raw: raw.arr_raw,
      tech_raw: raw.tech_raw,
      comp_raw: raw.comp_raw,
    };
    all.push({ ...cleaned, source_file: preferV7 ? row.source_file ?? "v7" : row.source_file ?? "v9" });
    if (preferV7) addedV7 += 1; else addedV9 += 1;
    dupKeys.add(key);
  };

  v7Rows.forEach((r) => tryAdd(r, true));
  v9Rows.forEach((r, idx) => {
    const pred = teacherMap.get(idx + 1);
    if (teacherPass(r, pred)) tryAdd(r, false);
  });
  console.log({ addedV7, addedV9, totalCandidates: all.length, labels: bucketCounts(all, (r) => r.label) });

  const targets = buildTargets(targetSize);
  const availLabels = bucketCounts(all, (r) => r.label);
  const adjustedTargets = adjustTargets(targets, availLabels);
  console.log({ targets, adjustedTargets });
  const selected = selectWithCaps(all, adjustedTargets, targetSize, seed);
  console.log({ selected: selected.length, selectedLoc: bucketCounts(selected, (r) => r.loc), selectedComp: bucketCounts(selected, (r) => r.comp) });

  // Stratified dev split 90/10
  const byLabel: Record<Label, LabeledRow[]> = { good_fit: [], maybe: [], bad_fit: [] };
  selected.forEach((r) => byLabel[r.label].push(r));
  Object.values(byLabel).forEach((arr) => seededShuffle(arr, seed));
  const train: LabeledRow[] = [];
  const dev: LabeledRow[] = [];
  for (const lbl of ["good_fit", "maybe", "bad_fit"] as Label[]) {
    const rows = byLabel[lbl];
    const devCount = Math.max(1, Math.round(rows.length * 0.1));
    dev.push(...rows.slice(0, devCount));
    train.push(...rows.slice(devCount));
  }

  // Outputs
  fs.mkdirSync(path.join(outDir, "mlx"), { recursive: true });
  fs.mkdirSync(path.join(outDir, "mlx_tokens"), { recursive: true });

  await writeJsonlFile(path.join(outDir, "train_labeled.jsonl"), train);
  await writeJsonlFile(path.join(outDir, "valid_labeled.jsonl"), dev);

  const dropRaw = (rows: LabeledRow[]) =>
    rows.map((r) => {
      const { loc_raw, arr_raw, tech_raw, comp_raw, ...rest } = r;
      return rest;
    });

  await writeJsonlFile(path.join(outDir, "mlx", "train.jsonl"), train);
  await writeJsonlFile(path.join(outDir, "mlx", "valid.jsonl"), dev);
  await writeJsonlFile(path.join(outDir, "mlx_tokens", "train.jsonl"), dropRaw(train));
  await writeJsonlFile(path.join(outDir, "mlx_tokens", "valid.jsonl"), dropRaw(dev));

  const report = {
    total: selected.length,
    labels: bucketCounts(selected, (r) => r.label),
    comp: bucketCounts(selected, (r) => r.comp),
    loc: bucketCounts(selected, (r) => r.loc),
    tech_oos: selected.filter((r) => r.tech.includes("OOS")).length,
    source: bucketCounts(selected, (r) => r.source_file ?? "unknown"),
  };
  fs.writeFileSync(path.join(outDir, "build_report.json"), JSON.stringify(report, null, 2));
  console.log("Built v11 dataset");
  console.log(report);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

/**
 * Clean V10 *_raw fields to concise evidence snippets.
 *
 * Why:
 * - Teacher raw fields can include long noisy prose that adds weak supervision.
 * - We want short, token-aligned evidence strings for loc/arr/sen/tech/comp.
 *
 * Usage:
 *   npx tsx src/cli/clean-raw-fields-v10.ts \
 *     --input data/v10/train_labeled.jsonl \
 *     --output data/v10/train_labeled.clean_raw.jsonl \
 *     --report data/v10/raw_clean_report.json
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { filterTechRaw } from "../lib/filter-tech-raw.js";

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
  loc_raw?: string | null;
  arr_raw?: string | null;
  sen_raw?: string | null;
  tech_raw?: string | null;
  comp_raw?: string | null;
  [key: string]: unknown;
};

type RawField = "loc_raw" | "arr_raw" | "sen_raw" | "tech_raw" | "comp_raw";

const MAX_LEN: Record<RawField, number> = {
  loc_raw: 48,
  arr_raw: 40,
  sen_raw: 24,
  tech_raw: 72,
  comp_raw: 48,
};

const TECH_NAME: Record<string, string> = {
  NODE: "Node.js",
  REACT: "React",
  JS_TS: "TypeScript/JavaScript",
  AI_ML: "AI/ML",
};

const COMP_CANONICAL: Record<string, string | null> = {
  NO_GBP: null,
  UP_TO_ONLY: "Up to salary",
  BELOW_45K: "Below £45k",
  RANGE_45_54K: "£45k-£54k",
  RANGE_55_74K: "£55k-£74k",
  RANGE_75_99K: "£75k-£99k",
  ABOVE_100K: "£100k+",
};

const ARR_CANONICAL: Record<string, string | null> = {
  REMOTE: "Remote",
  HYBRID: "Hybrid",
  IN_OFFICE: "On-site",
  UNK: null,
};

function norm(input: string | null | undefined): string {
  return (input ?? "").replace(/\s+/g, " ").trim();
}

function cut(input: string | null | undefined, maxLen: number): string | null {
  const text = norm(input);
  if (!text) return null;
  if (text.length <= maxLen) return text;
  if (maxLen <= 3) return text.slice(0, maxLen);
  return `${text.slice(0, Math.max(0, maxLen - 3)).trimEnd()}...`;
}

function firstMatch(text: string, patterns: RegExp[]): string | null {
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match && match[0]) return norm(match[0]);
  }
  return null;
}

function cleanLocRaw(job: LabeledJob): string | null {
  const combined = norm([job.loc_raw, job.job_location, job.jd_text.slice(0, 400)].filter(Boolean).join(" | "));
  if (!combined) {
    if (job.loc === "UNK") return null;
    if (job.loc === "REMOTE") return "Remote";
    if (job.loc === "IN_LONDON") return "London";
    if (job.loc === "UK_OTHER") return "UK";
    if (job.loc === "OUTSIDE_UK") return "Outside UK";
    return null;
  }

  if (job.loc === "REMOTE") {
    const hit = firstMatch(combined, [
      /\bfully remote(?:\s*\([^)]*\))?/i,
      /\bremote(?:\s*-\s*[a-z ]+)?(?:\s*\([^)]*\))?/i,
      /\bwork from home\b/i,
    ]);
    return cut(hit ?? "Remote", MAX_LEN.loc_raw);
  }

  if (job.loc === "IN_LONDON") {
    const hit = firstMatch(combined, [/\blondon(?:,?\s*uk)?\b/i]);
    return cut(hit ?? "London", MAX_LEN.loc_raw);
  }

  if (job.loc === "UK_OTHER") {
    const first = norm((job.job_location ?? "").split(/[|,]/)[0]);
    if (first && !/london/i.test(first)) return cut(first, MAX_LEN.loc_raw);
    const hit = firstMatch(combined, [/\b(uk|united kingdom|england|scotland|wales|northern ireland)\b/i]);
    return cut(hit ?? "UK (non-London)", MAX_LEN.loc_raw);
  }

  if (job.loc === "OUTSIDE_UK") {
    const first = norm((job.job_location ?? "").split(/[|,]/)[0]);
    if (first && !/\b(uk|united kingdom|england|scotland|wales|northern ireland|london)\b/i.test(first)) {
      return cut(first, MAX_LEN.loc_raw);
    }
    return cut("Outside UK", MAX_LEN.loc_raw);
  }

  return null;
}

function cleanArrRaw(job: LabeledJob): string | null {
  const canonical = ARR_CANONICAL[job.arr] ?? null;
  if (!canonical) return null;

  const source = norm([job.arr_raw, job.job_location, job.jd_text.slice(0, 400)].filter(Boolean).join(" | "));
  if (!source) return canonical;

  if (job.arr === "REMOTE") {
    const hit = firstMatch(source, [/\bfully remote\b/i, /\bremote\b/i, /\bwork from home\b/i]);
    return cut(hit ?? canonical, MAX_LEN.arr_raw);
  }
  if (job.arr === "HYBRID") {
    const hit = firstMatch(source, [/\bhybrid\b/i, /\b\d+\s+days?\s+in\s+office\b/i]);
    return cut(hit ?? canonical, MAX_LEN.arr_raw);
  }
  const hit = firstMatch(source, [/\bon[-\s]?site\b/i, /\bin[-\s]?office\b/i, /\bin office\b/i]);
  return cut(hit ?? canonical, MAX_LEN.arr_raw);
}

function cleanSenRaw(job: LabeledJob): string | null {
  const source = norm([job.sen_raw, job.title].filter(Boolean).join(" | "));
  if (!source) {
    if (job.sen === "LEVEL_1") return "Junior";
    if (job.sen === "LEVEL_2") return "Mid-level";
    if (job.sen === "LEVEL_3") return "Senior";
    return null;
  }

  if (job.sen === "LEVEL_1") {
    const hit = firstMatch(source, [/\b(junior|graduate|entry[\s-]?level|intern|trainee|apprentice)\b/i]);
    return cut(hit ?? "Junior", MAX_LEN.sen_raw);
  }
  if (job.sen === "LEVEL_3") {
    const hit = firstMatch(source, [/\b(senior|staff|principal|lead|head|director|architect|manager|founding)\b/i]);
    return cut(hit ?? "Senior", MAX_LEN.sen_raw);
  }
  const hit = firstMatch(source, [/\b(mid(?:dle)?[\s-]?level|intermediate)\b/i]);
  return cut(hit ?? "Mid-level", MAX_LEN.sen_raw);
}

function cleanTechRaw(job: LabeledJob): string | null {
  const filtered = filterTechRaw(job.tech_raw, job.tech ?? []);
  if (filtered) return cut(filtered, MAX_LEN.tech_raw);

  const tech = Array.isArray(job.tech) ? job.tech : [];
  const real = tech.filter((t) => t !== "OOS");
  if (real.length === 0) return null;

  const canonical = real
    .map((t) => TECH_NAME[t] ?? t)
    .slice(0, 4)
    .join(", ");
  return cut(canonical, MAX_LEN.tech_raw);
}

function cleanCompRaw(job: LabeledJob): string | null {
  if (job.comp === "NO_GBP") return null;

  const source = norm([job.comp_raw, job.jd_text.slice(0, 700)].filter(Boolean).join(" | "));
  if (!source) return cut(COMP_CANONICAL[job.comp] ?? null, MAX_LEN.comp_raw);

  const salaryHit = firstMatch(source, [
    /£\s?\d[\d,]*(?:\s?[kK])?(?:\s?(?:-|–|—|to)\s?£?\s?\d[\d,]*(?:\s?[kK])?)?/i,
    /\bup to\s*£\s?\d[\d,]*(?:\s?[kK])?/i,
    /\b£\s?\d[\d,]*\s?(?:per year|p\.a\.|pa|annum)\b/i,
  ]);
  if (salaryHit) return cut(salaryHit, MAX_LEN.comp_raw);

  return cut(COMP_CANONICAL[job.comp] ?? null, MAX_LEN.comp_raw);
}

function cleanJobRaw(job: LabeledJob): LabeledJob {
  return {
    ...job,
    loc_raw: cleanLocRaw(job),
    arr_raw: cleanArrRaw(job),
    sen_raw: cleanSenRaw(job),
    tech_raw: cleanTechRaw(job),
    comp_raw: cleanCompRaw(job),
  };
}

function summarizeLengths(rows: LabeledJob[], fields: RawField[]): Record<string, unknown> {
  const summary: Record<string, unknown> = {};
  for (const field of fields) {
    const vals = rows.map((r) => r[field]).filter((v): v is string => typeof v === "string" && v.trim().length > 0);
    const lengths = vals.map((v) => v.length).sort((a, b) => a - b);
    const q = (p: number): number => {
      if (lengths.length === 0) return 0;
      return lengths[Math.floor((lengths.length - 1) * p)] ?? 0;
    };
    summary[field] = {
      n_present: vals.length,
      avg_len: Number((lengths.reduce((a, b) => a + b, 0) / (lengths.length || 1)).toFixed(2)),
      p50: q(0.5),
      p90: q(0.9),
      p99: q(0.99),
      over_80: vals.filter((v) => v.length > 80).length,
      over_140: vals.filter((v) => v.length > 140).length,
    };
  }
  return summary;
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v10/train_labeled.jsonl";
  const outputPath = getStringArg(args, "output") ?? "data/v10/train_labeled.clean_raw.jsonl";
  const reportPath = getStringArg(args, "report") ?? "data/v10/raw_clean_report.json";
  const sampleCount = getNumberArg(args, "sample-count") ?? 5;

  const rows = await readJsonlFile<LabeledJob>(inputPath);
  if (rows.length === 0) throw new Error(`No rows found in ${inputPath}`);

  const fields: RawField[] = ["loc_raw", "arr_raw", "sen_raw", "tech_raw", "comp_raw"];
  const before = summarizeLengths(rows, fields);
  const cleaned = rows.map(cleanJobRaw);
  const after = summarizeLengths(cleaned, fields);

  await writeJsonlFile(outputPath, cleaned);
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });

  const changedByField: Record<RawField, number> = {
    loc_raw: 0,
    arr_raw: 0,
    sen_raw: 0,
    tech_raw: 0,
    comp_raw: 0,
  };
  const samples: Record<RawField, Array<{ job_id: string; before: string | null; after: string | null }>> = {
    loc_raw: [],
    arr_raw: [],
    sen_raw: [],
    tech_raw: [],
    comp_raw: [],
  };

  for (let i = 0; i < rows.length; i += 1) {
    const prev = rows[i]!;
    const next = cleaned[i]!;
    for (const field of fields) {
      const b = (prev[field] ?? null) as string | null;
      const a = (next[field] ?? null) as string | null;
      if (b !== a) {
        changedByField[field] += 1;
        if (samples[field].length < sampleCount) {
          samples[field].push({
            job_id: prev.job_id,
            before: b,
            after: a,
          });
        }
      }
    }
  }

  const report = {
    timestamp: new Date().toISOString(),
    input: inputPath,
    output: outputPath,
    n_rows: rows.length,
    changed_by_field: changedByField,
    lengths_before: before,
    lengths_after: after,
    samples,
  };

  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), "utf8");

  console.log(`Cleaned ${rows.length} rows`);
  console.log(`Output: ${outputPath}`);
  console.log(`Report: ${reportPath}`);
  for (const field of fields) {
    console.log(`  ${field}: changed ${changedByField[field]}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

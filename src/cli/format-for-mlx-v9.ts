/**
 * Format V9 training data as MLX chat JSONL for LoRA fine-tuning.
 *
 * V9 version: all 10 fields in student output — 5 token fields + 5 _raw fields.
 * All _raw fields are hard-capped at 50 characters (chain-of-thought scaffolding
 * without the token budget of uncapped v7 _raw fields).
 *
 * Option B split: synthetic jobs (generated_v*, node_variants) go to training
 * only. Validation set is built exclusively from real scraped jobs.
 *
 * Usage:
 *   npx tsx src/cli/format-for-mlx-v9.ts \
 *     --input data/v9/train_labeled.jsonl \
 *     --output-dir data/v9/mlx \
 *     --prompt prompts/student_v9.txt \
 *     --valid-count 72 \
 *     --max-tokens 7500
 */

import * as fs from "node:fs";
import { readFile } from "node:fs/promises";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import { filterTechRaw } from "../lib/filter-tech-raw.js";

// ── Types ────────────────────────────────────────────────────────────────

type V9Job = {
  job_id: string;
  title: string;
  company: string;
  job_location: string;
  jd_text: string;
  // Token fields (required)
  loc: string;
  arr: string;
  sen: string;
  tech: string[];
  comp: string;
  // Raw fields (optional — present when labeled with teacher_v9)
  loc_raw?: string | null;
  arr_raw?: string | null;
  sen_raw?: string | null;
  tech_raw?: string | null;
  comp_raw?: string | null;
  // Computed
  score: number;
  label: string;
  // Source tracking
  source_file?: string;
  is_synthetic?: boolean;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type MLXExample = {
  messages: ChatMessage[];
};

// ── Synthetic job detection (Option B split) ─────────────────────────────

const SYNTHETIC_SOURCE_FILES = new Set([
  "generated_v7",
  "generated_v9",
]);

function isSyntheticJob(job: V9Job): boolean {
  if (job.is_synthetic === true) return true;
  if (job.is_synthetic === false) return false;
  // Fallback: check source_file
  const src = job.source_file ?? "";
  return (
    SYNTHETIC_SOURCE_FILES.has(src) ||
    src.startsWith("generated_v") ||
    src.includes("synthetic") ||
    (job.job_id ?? "").startsWith("gen_v")
  );
}

// ── 50-char cap on _raw fields ────────────────────────────────────────────

function cap50(s: string | null | undefined): string | null {
  if (!s) return null;
  return s.length > 50 ? s.slice(0, 50) : s;
}

// ── Seeded RNG ────────────────────────────────────────────────────────────

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const result = [...arr];
  let s = seed | 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}

// ── Smart Truncation ──────────────────────────────────────────────────────

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function smartTruncate(jdText: string, maxTokens: number): { text: string; truncated: boolean } {
  if (estimateTokens(jdText) <= maxTokens) {
    return { text: jdText, truncated: false };
  }

  const words = jdText.split(/\s+/);
  const totalWords = words.length;
  const protectedWords = new Set<number>();

  for (let i = 0; i < Math.min(300, totalWords); i++) protectedWords.add(i);
  for (let i = Math.max(0, totalWords - 200); i < totalWords; i++) protectedWords.add(i);

  const salaryPattern = /[£$€]|salary|compensation|pay|per annum|p\.a\./gi;
  let charIndex = 0;
  for (let i = 0; i < totalWords; i++) {
    const word = words[i]!;
    const wordStart = jdText.indexOf(word, charIndex);
    charIndex = wordStart + word.length;
    if (salaryPattern.test(word)) {
      for (let j = Math.max(0, i - 50); j < Math.min(totalWords, i + 50); j++) {
        protectedWords.add(j);
      }
    }
    salaryPattern.lastIndex = 0;
  }

  let bestStart = -1, bestLen = 0, currentStart = -1, currentLen = 0;
  for (let i = 0; i < totalWords; i++) {
    if (!protectedWords.has(i)) {
      if (currentStart === -1) currentStart = i;
      currentLen++;
    } else {
      if (currentLen > bestLen) { bestStart = currentStart; bestLen = currentLen; }
      currentStart = -1; currentLen = 0;
    }
  }
  if (currentLen > bestLen) { bestStart = currentStart; bestLen = currentLen; }

  if (bestLen === 0 || bestStart === -1) {
    return { text: jdText.slice(0, maxTokens * 4) + " [...]", truncated: true };
  }

  const targetWords = Math.floor(maxTokens * 4 / 6);
  const wordsToRemove = totalWords - targetWords;
  if (wordsToRemove <= 0) return { text: jdText, truncated: false };

  const removeStart = bestStart + Math.floor((bestLen - wordsToRemove) / 2);
  const removeEnd = Math.min(removeStart + wordsToRemove, bestStart + bestLen);
  const kept = [...words.slice(0, Math.max(0, removeStart)), "[...]", ...words.slice(removeEnd)];

  return { text: kept.join(" "), truncated: true };
}

// ── Build messages ────────────────────────────────────────────────────────

function buildUserMessage(promptTemplate: string, job: V9Job): string {
  return promptTemplate
    .replace(/\{\{job_title\}\}/g, job.title)
    .replace(/\{\{job_location\}\}/g, job.job_location ?? "")
    .replace(/\{\{jd_text\}\}/g, job.jd_text);
}

function buildAssistantResponse(job: V9Job): string {
  // 10 fields: all 5 _raw (capped at 50 chars) + 5 token fields
  const output = {
    loc_raw: cap50(job.loc_raw),
    loc: job.loc,
    arr_raw: cap50(job.arr_raw),
    arr: job.arr,
    sen_raw: cap50(job.sen_raw),
    sen: job.sen,
    tech_raw: filterTechRaw(job.tech_raw, job.tech),
    tech: job.tech,
    comp_raw: cap50(job.comp_raw),
    comp: job.comp,
  };
  return JSON.stringify(output);
}

// ── Main ──────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath  = getStringArg(args, "input")      ?? "data/v9/train_labeled.jsonl";
  const outputDir  = getStringArg(args, "output-dir") ?? "data/v9/mlx";
  const promptPath = getStringArg(args, "prompt")     ?? "prompts/student_v9.txt";
  const validCount = getNumberArg(args, "valid-count") ?? 72;
  const maxTokens  = getNumberArg(args, "max-tokens")  ?? 7500;

  fs.mkdirSync(outputDir, { recursive: true });

  const promptTemplate = await readFile(promptPath, "utf8");
  console.log(`Prompt: ${promptPath} (${promptTemplate.length} chars)`);

  const jobs = await readJsonlFile<V9Job>(inputPath);
  console.log(`Loaded ${jobs.length} jobs from ${inputPath}`);

  // Validate required token fields
  const tokenFields = ["loc", "arr", "sen", "tech", "comp"] as const;
  const missing = jobs.filter((j) =>
    tokenFields.some((f) => (j as Record<string, unknown>)[f] === undefined),
  );
  if (missing.length > 0) {
    console.error(`${missing.length} jobs missing token fields (first: ${missing[0]?.job_id})`);
    process.exit(1);
  }

  // Separate synthetic vs real (Option B split)
  const syntheticJobs = jobs.filter(isSyntheticJob);
  const realJobs = jobs.filter((j) => !isSyntheticJob(j));
  console.log(`\nOption B split: ${syntheticJobs.length} synthetic (train only) + ${realJobs.length} real`);

  // Smart truncation
  let truncatedCount = 0;
  for (const job of jobs) {
    const totalEstimate = estimateTokens(promptTemplate + job.jd_text + "x".repeat(200));
    if (totalEstimate > maxTokens) {
      const { text, truncated } = smartTruncate(job.jd_text, maxTokens - 500);
      if (truncated) { job.jd_text = text; truncatedCount++; }
    }
  }
  if (truncatedCount > 0) console.log(`Smart-truncated ${truncatedCount} long JDs`);

  const systemMsg = "Respond with JSON only.";

  // Build all examples
  const syntheticExamples: MLXExample[] = syntheticJobs.map((job) => ({
    messages: [
      { role: "system" as const, content: systemMsg },
      { role: "user" as const, content: buildUserMessage(promptTemplate, job) },
      { role: "assistant" as const, content: buildAssistantResponse(job) },
    ],
  }));

  const realExamples: MLXExample[] = realJobs.map((job) => ({
    messages: [
      { role: "system" as const, content: systemMsg },
      { role: "user" as const, content: buildUserMessage(promptTemplate, job) },
      { role: "assistant" as const, content: buildAssistantResponse(job) },
    ],
  }));

  // Stratified split on real jobs only — maintain label proportions
  const byLabel: Record<string, { ex: MLXExample; label: string }[]> = {};
  for (let i = 0; i < realJobs.length; i++) {
    const label = realJobs[i]!.label;
    if (!byLabel[label]) byLabel[label] = [];
    byLabel[label]!.push({ ex: realExamples[i]!, label });
  }

  // Distribute validCount proportionally across labels
  const totalReal = realJobs.length;
  let trainReal: MLXExample[] = [];
  let validExamples: MLXExample[] = [];

  console.log(`\nReal job split (${validCount} valid target):`);
  for (const [label, items] of Object.entries(byLabel)) {
    const shuffled = seededShuffle(items, 42);
    const labelValidCount = Math.max(1, Math.round(shuffled.length * validCount / totalReal));
    validExamples.push(...shuffled.slice(0, labelValidCount).map((x) => x.ex));
    trainReal.push(...shuffled.slice(labelValidCount).map((x) => x.ex));
    console.log(`  ${label}: ${shuffled.length - labelValidCount} train, ${labelValidCount} valid`);
  }

  // Combine: all synthetic + remaining real → training
  let trainExamples = [...syntheticExamples, ...trainReal];
  trainExamples = seededShuffle(trainExamples, 42);
  validExamples = seededShuffle(validExamples, 43);

  const trainPath = `${outputDir}/train.jsonl`;
  const validPath = `${outputDir}/valid.jsonl`;
  await writeJsonlFile(trainPath, trainExamples);
  await writeJsonlFile(validPath, validExamples);

  console.log(`\nWrote ${trainExamples.length} train → ${trainPath}`);
  console.log(`Wrote ${validExamples.length} valid → ${validPath}`);
  console.log(`  (${syntheticJobs.length} synthetic + ${trainReal.length} real in train)`);

  // Stats
  const allExamples = [...trainExamples, ...validExamples];
  const avgUserLen = Math.round(allExamples.reduce((s, e) => s + e.messages[1]!.content.length, 0) / allExamples.length);
  const avgAssistantLen = Math.round(allExamples.reduce((s, e) => s + e.messages[2]!.content.length, 0) / allExamples.length);
  const maxUserLen = Math.max(...allExamples.map((e) => e.messages[1]!.content.length));

  console.log(`\nAvg user message:       ${avgUserLen} chars (~${Math.round(avgUserLen / 4)} tokens)`);
  console.log(`Max user message:       ${maxUserLen} chars (~${Math.round(maxUserLen / 4)} tokens)`);
  console.log(`Avg assistant response: ${avgAssistantLen} chars (~${Math.round(avgAssistantLen / 4)} tokens)`);

  const sample = trainExamples[0];
  if (sample) {
    console.log("\n=== Sample training example ===");
    console.log(`User: ${sample.messages[1]!.content.slice(0, 150)}...`);
    console.log(`Assistant: ${sample.messages[2]!.content}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

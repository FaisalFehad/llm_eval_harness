/**
 * Format V7 training data as MLX chat JSONL for LoRA fine-tuning.
 *
 * V7 version: uses 5 fields (loc, arr, sen, tech, comp) with 10-field
 * interleaved output format. Tech is an array of individual tokens.
 *
 * Features:
 *   - Uses V7 semantic tokens (IN_LONDON, LEVEL_3, OOS, etc.)
 *   - Uses the minimal V7 student prompt
 *   - Smart truncation: protects salary windows and first/last paragraphs
 *   - Stratified train/valid split by computed label
 *
 * Usage:
 *   npx tsx src/cli/format-for-mlx-v7.ts \
 *     --input data/v7/labeled.jsonl \
 *     --output-dir data/v7/mlx \
 *     --prompt prompts/student_v7.txt \
 *     --valid-pct 10 \
 *     --max-tokens 7500
 */

import * as fs from "node:fs";
import { readFile } from "node:fs/promises";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type V7Job = {
  job_id: string;
  title: string;
  company: string;
  job_location: string; // raw location (renamed from "location" to avoid V7 token collision)
  jd_text: string;
  // V7 token fields (10-field interleaved)
  loc_raw: string | null;
  loc: string;
  arr_raw: string | null;
  arr: string;
  sen_raw: string | null;
  sen: string;
  tech_raw: string | null;
  tech: string[]; // array of individual tokens
  comp_raw: string | null;
  comp: string;
  // Computed
  score: number;
  label: string;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type MLXExample = {
  messages: ChatMessage[];
};

// ── Seeded RNG ──────────────────────────────────────────────────────────

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

// ── Smart Truncation ────────────────────────────────────────────────────

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

  // Protect first 300 words
  for (let i = 0; i < Math.min(300, totalWords); i++) {
    protectedWords.add(i);
  }

  // Protect last 200 words
  for (let i = Math.max(0, totalWords - 200); i < totalWords; i++) {
    protectedWords.add(i);
  }

  // Protect 100-word windows around salary-related positions
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

  // Find the largest contiguous unprotected region
  let bestStart = -1;
  let bestLen = 0;
  let currentStart = -1;
  let currentLen = 0;

  for (let i = 0; i < totalWords; i++) {
    if (!protectedWords.has(i)) {
      if (currentStart === -1) currentStart = i;
      currentLen++;
    } else {
      if (currentLen > bestLen) {
        bestStart = currentStart;
        bestLen = currentLen;
      }
      currentStart = -1;
      currentLen = 0;
    }
  }
  if (currentLen > bestLen) {
    bestStart = currentStart;
    bestLen = currentLen;
  }

  if (bestLen === 0 || bestStart === -1) {
    const maxChars = maxTokens * 4;
    return { text: jdText.slice(0, maxChars) + " [...]", truncated: true };
  }

  const targetWords = Math.floor(maxTokens * 4 / 6);
  const wordsToRemove = totalWords - targetWords;

  if (wordsToRemove <= 0) {
    return { text: jdText, truncated: false };
  }

  const removeStart = bestStart + Math.floor((bestLen - wordsToRemove) / 2);
  const removeEnd = Math.min(removeStart + wordsToRemove, bestStart + bestLen);

  const kept = [
    ...words.slice(0, Math.max(0, removeStart)),
    "[...]",
    ...words.slice(removeEnd),
  ];

  return { text: kept.join(" "), truncated: true };
}

// ── Build messages ──────────────────────────────────────────────────────

function buildUserMessage(promptTemplate: string, job: V7Job): string {
  return promptTemplate
    .replace(/\{\{job_title\}\}/g, job.title)
    .replace(/\{\{job_location\}\}/g, job.job_location ?? "")
    .replace(/\{\{jd_text\}\}/g, job.jd_text);
}

function buildAssistantResponse(job: V7Job): string {
  // Full 10-field interleaved format with _raw fields for chain-of-thought grounding
  const output = {
    loc_raw: job.loc_raw,
    loc: job.loc,
    arr_raw: job.arr_raw,
    arr: job.arr,
    sen_raw: job.sen_raw,
    sen: job.sen,
    tech_raw: job.tech_raw,
    tech: job.tech,
    comp_raw: job.comp_raw,
    comp: job.comp,
  };
  return JSON.stringify(output);
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v7/labeled.jsonl";
  const outputDir = getStringArg(args, "output-dir") ?? "data/v7/mlx";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/student_v7.txt";
  const validPct = getNumberArg(args, "valid-pct") ?? 10;
  const maxTokens = getNumberArg(args, "max-tokens") ?? 7500;

  fs.mkdirSync(outputDir, { recursive: true });

  const promptTemplate = await readFile(promptPath, "utf8");
  console.log(`Prompt: ${promptPath} (${promptTemplate.length} chars)`);

  const jobs = await readJsonlFile<V7Job>(inputPath);
  console.log(`Loaded ${jobs.length} training jobs from ${inputPath}`);

  // Validate required fields (10-field V7 format)
  const tokenFields = ["loc", "arr", "sen", "tech", "comp"] as const;
  const rawFields = tokenFields.map((f) => `${f}_raw`);

  const missing = jobs.filter((j) => {
    for (const f of tokenFields) {
      if ((j as Record<string, unknown>)[f] === undefined) return true;
    }
    for (const f of rawFields) {
      if (!((f in (j as Record<string, unknown>)))) return true;
    }
    return false;
  });
  if (missing.length > 0) {
    console.error(`${missing.length} jobs missing V7 semantic token fields.`);
    console.error(`First missing: ${missing[0]?.job_id} — check that input was labeled with teacher_v7.txt`);
    process.exit(1);
  }

  // Smart truncation
  let truncatedCount = 0;
  const truncatedJobs: string[] = [];

  for (const job of jobs) {
    const raws = (job.loc_raw ?? "") + (job.arr_raw ?? "") +
      (job.sen_raw ?? "") + (job.tech_raw ?? "") + (job.comp_raw ?? "");
    const totalEstimate = estimateTokens(
      promptTemplate + job.jd_text + raws + "x".repeat(100),
    );
    if (totalEstimate > maxTokens) {
      const { text, truncated } = smartTruncate(job.jd_text, maxTokens - 500);
      if (truncated) {
        job.jd_text = text;
        truncatedCount++;
        truncatedJobs.push(job.job_id);
      }
    }
  }

  if (truncatedCount > 0) {
    console.log(`Smart-truncated ${truncatedCount} long JDs`);
    fs.writeFileSync(
      `${outputDir}/truncated_ids.json`,
      JSON.stringify(truncatedJobs, null, 2),
    );
  }

  // System message for Qwen2.5
  const systemMsg = "Respond with JSON only.";
  console.log(`System message: "${systemMsg}"`);

  // Format as MLX chat examples (10-field V7 interleaved format)
  const examples: MLXExample[] = jobs.map((job) => ({
    messages: [
      { role: "system" as const, content: systemMsg },
      { role: "user" as const, content: buildUserMessage(promptTemplate, job) },
      { role: "assistant" as const, content: buildAssistantResponse(job) },
    ],
  }));

  // Stratified split: maintain label proportions in train/valid
  const byLabel: Record<string, MLXExample[]> = {};
  for (let i = 0; i < jobs.length; i++) {
    const label = jobs[i]!.label;
    if (!byLabel[label]) byLabel[label] = [];
    byLabel[label]!.push(examples[i]!);
  }

  let trainExamples: MLXExample[] = [];
  let validExamples: MLXExample[] = [];

  for (const [label, labelExamples] of Object.entries(byLabel)) {
    const shuffled = seededShuffle(labelExamples, 42);
    const validCount = Math.max(1, Math.round(shuffled.length * validPct / 100));
    validExamples.push(...shuffled.slice(0, validCount));
    trainExamples.push(...shuffled.slice(validCount));
    console.log(`  ${label}: ${shuffled.length - validCount} train, ${validCount} valid`);
  }

  // Final shuffle
  trainExamples = seededShuffle(trainExamples, 42);
  validExamples = seededShuffle(validExamples, 42);

  // Write files
  const trainPath = `${outputDir}/train.jsonl`;
  const validPath = `${outputDir}/valid.jsonl`;

  await writeJsonlFile(trainPath, trainExamples);
  await writeJsonlFile(validPath, validExamples);

  console.log(`\nWrote ${trainExamples.length} train to ${trainPath}`);
  console.log(`Wrote ${validExamples.length} valid to ${validPath}`);

  // Stats
  const avgUserLen = Math.round(
    examples.reduce((sum, e) => sum + e.messages[1]!.content.length, 0) / examples.length,
  );
  const avgAssistantLen = Math.round(
    examples.reduce((sum, e) => sum + e.messages[2]!.content.length, 0) / examples.length,
  );
  const maxUserLen = Math.max(...examples.map((e) => e.messages[1]!.content.length));

  console.log(`\nAvg user message:      ${avgUserLen} chars (~${Math.round(avgUserLen / 4)} tokens)`);
  console.log(`Max user message:      ${maxUserLen} chars (~${Math.round(maxUserLen / 4)} tokens)`);
  console.log(`Avg assistant response: ${avgAssistantLen} chars (~${Math.round(avgAssistantLen / 4)} tokens)`);

  // Sample
  const sample = trainExamples[0];
  if (sample) {
    console.log("\n=== Sample training example ===");
    console.log(`System: ${sample.messages[0]!.content}`);
    console.log(`User: ${sample.messages[1]!.content.slice(0, 150)}...`);
    console.log(`Assistant: ${sample.messages[2]!.content}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

/**
 * Format V8 training data as MLX chat JSONL for LoRA fine-tuning.
 *
 * V8 version: 5 token fields only in student output (loc, arr, sen, tech, comp).
 * No _raw fields — redirects gradient signal to classification decisions.
 * Teacher labels may include loc_raw, sen_raw, comp_raw but these are NOT
 * included in the assistant response the student is trained to produce.
 *
 * Usage:
 *   npx tsx src/cli/format-for-mlx-v8.ts \
 *     --input data/v8/train_labeled.jsonl \
 *     --output-dir data/v8/mlx \
 *     --prompt prompts/student_v8.txt \
 *     --valid-pct 10 \
 *     --max-tokens 7500
 */

import * as fs from "node:fs";
import { readFile } from "node:fs/promises";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type V8Job = {
  job_id: string;
  title: string;
  company: string;
  job_location: string;
  jd_text: string;
  // Token fields (required for training)
  loc: string;
  arr: string;
  sen: string;
  tech: string[];
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

function buildUserMessage(promptTemplate: string, job: V8Job): string {
  return promptTemplate
    .replace(/\{\{job_title\}\}/g, job.title)
    .replace(/\{\{job_location\}\}/g, job.job_location ?? "")
    .replace(/\{\{jd_text\}\}/g, job.jd_text);
}

function buildAssistantResponse(job: V8Job): string {
  // 5 token fields only — no _raw fields in student output
  const output = {
    loc: job.loc,
    arr: job.arr,
    sen: job.sen,
    tech: job.tech,
    comp: job.comp,
  };
  return JSON.stringify(output);
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v8/train_labeled.jsonl";
  const outputDir = getStringArg(args, "output-dir") ?? "data/v8/mlx";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/student_v8.txt";
  const validPct = getNumberArg(args, "valid-pct") ?? 10;
  const maxTokens = getNumberArg(args, "max-tokens") ?? 7500;

  fs.mkdirSync(outputDir, { recursive: true });

  const promptTemplate = await readFile(promptPath, "utf8");
  console.log(`Prompt: ${promptPath} (${promptTemplate.length} chars)`);

  const jobs = await readJsonlFile<V8Job>(inputPath);
  console.log(`Loaded ${jobs.length} training jobs from ${inputPath}`);

  // Validate required token fields (no raw fields required)
  const tokenFields = ["loc", "arr", "sen", "tech", "comp"] as const;

  const missing = jobs.filter((j) => {
    for (const f of tokenFields) {
      if ((j as Record<string, unknown>)[f] === undefined) return true;
    }
    return false;
  });
  if (missing.length > 0) {
    console.error(`${missing.length} jobs missing token fields.`);
    console.error(`First missing: ${missing[0]?.job_id} — check that input was labeled with teacher_v8.txt`);
    process.exit(1);
  }

  // Smart truncation
  let truncatedCount = 0;
  const truncatedJobs: string[] = [];

  for (const job of jobs) {
    const totalEstimate = estimateTokens(promptTemplate + job.jd_text + "x".repeat(100));
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

  // Format as MLX chat examples (5-field token-only format)
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

  console.log(`\nAvg user message:       ${avgUserLen} chars (~${Math.round(avgUserLen / 4)} tokens)`);
  console.log(`Max user message:       ${maxUserLen} chars (~${Math.round(maxUserLen / 4)} tokens)`);
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

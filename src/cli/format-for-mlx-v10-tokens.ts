/**
 * Format labeled jobs as MLX chat JSONL with token-only assistant output.
 *
 * Output JSON keys:
 *   loc, arr, sen, tech, comp
 *
 * Usage:
 *   npx tsx src/cli/format-for-mlx-v10-tokens.ts \
 *     --input data/v7/train_labeled.jsonl \
 *     --output-dir data/v10_tokens_v7/mlx \
 *     --prompt prompts/student_v10_tokens.txt \
 *     --valid-count 72 \
 *     --max-tokens 7500
 */

import * as fs from "node:fs";
import { readFile } from "node:fs/promises";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type Job = {
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
  label: string;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type MLXExample = {
  messages: ChatMessage[];
};

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

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function smartTruncate(jdText: string, maxTokens: number): { text: string; truncated: boolean } {
  if (estimateTokens(jdText) <= maxTokens) return { text: jdText, truncated: false };

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
    return { text: jdText.slice(0, maxTokens * 4) + " [...]", truncated: true };
  }

  const targetWords = Math.floor((maxTokens * 4) / 6);
  const wordsToRemove = totalWords - targetWords;
  if (wordsToRemove <= 0) return { text: jdText, truncated: false };

  const removeStart = bestStart + Math.floor((bestLen - wordsToRemove) / 2);
  const removeEnd = Math.min(removeStart + wordsToRemove, bestStart + bestLen);
  const kept = [...words.slice(0, Math.max(0, removeStart)), "[...]", ...words.slice(removeEnd)];
  return { text: kept.join(" "), truncated: true };
}

function buildUserMessage(promptTemplate: string, job: Job): string {
  return promptTemplate
    .replace(/\{\{job_title\}\}/g, job.title)
    .replace(/\{\{job_location\}\}/g, job.job_location ?? "")
    .replace(/\{\{jd_text\}\}/g, job.jd_text);
}

function buildAssistantResponse(job: Job): string {
  return JSON.stringify({
    loc: job.loc,
    arr: job.arr,
    sen: job.sen,
    tech: job.tech,
    comp: job.comp,
  });
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v7/train_labeled.jsonl";
  const outputDir = getStringArg(args, "output-dir") ?? "data/v10_tokens_v7/mlx";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/student_v10_tokens.txt";
  const validCount = getNumberArg(args, "valid-count") ?? 72;
  const maxTokens = getNumberArg(args, "max-tokens") ?? 7500;

  fs.mkdirSync(outputDir, { recursive: true });

  const promptTemplate = await readFile(promptPath, "utf8");
  console.log(`Prompt: ${promptPath} (${promptTemplate.length} chars)`);

  const jobs = await readJsonlFile<Job>(inputPath);
  console.log(`Loaded ${jobs.length} jobs from ${inputPath}`);

  const tokenFields = ["loc", "arr", "sen", "tech", "comp"] as const;
  const missing = jobs.filter((j) =>
    tokenFields.some((f) => (j as Record<string, unknown>)[f] === undefined),
  );
  if (missing.length > 0) {
    console.error(`${missing.length} jobs missing token fields (first: ${missing[0]?.job_id})`);
    process.exit(1);
  }

  let truncatedCount = 0;
  for (const job of jobs) {
    const totalEstimate = estimateTokens(promptTemplate + job.jd_text + "x".repeat(200));
    if (totalEstimate > maxTokens) {
      const { text, truncated } = smartTruncate(job.jd_text, maxTokens - 500);
      if (truncated) {
        job.jd_text = text;
        truncatedCount++;
      }
    }
  }
  if (truncatedCount > 0) console.log(`Smart-truncated ${truncatedCount} long JDs`);

  const systemMsg = "Respond with JSON only.";
  const examples = jobs.map((job) => ({
    label: job.label,
    ex: {
      messages: [
        { role: "system" as const, content: systemMsg },
        { role: "user" as const, content: buildUserMessage(promptTemplate, job) },
        { role: "assistant" as const, content: buildAssistantResponse(job) },
      ],
    } satisfies MLXExample,
  }));

  const byLabel: Record<string, MLXExample[]> = {};
  for (const row of examples) {
    if (!byLabel[row.label]) byLabel[row.label] = [];
    byLabel[row.label]!.push(row.ex);
  }

  let trainExamples: MLXExample[] = [];
  let validExamples: MLXExample[] = [];
  let remainingValid = Math.max(1, validCount);
  const labels = Object.keys(byLabel);
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i]!;
    const pool = seededShuffle(byLabel[label]!, 42 + i);
    const share = Math.max(1, Math.round((pool.length / jobs.length) * validCount));
    const take = Math.min(pool.length - 1, i === labels.length - 1 ? remainingValid : share);
    const clampedTake = Math.max(1, Math.max(0, take));
    validExamples.push(...pool.slice(0, clampedTake));
    trainExamples.push(...pool.slice(clampedTake));
    remainingValid -= clampedTake;
    console.log(`  ${label}: ${pool.length - clampedTake} train, ${clampedTake} valid`);
  }

  if (remainingValid > 0 && trainExamples.length > 0) {
    const shuffledTrain = seededShuffle(trainExamples, 31415);
    validExamples.push(...shuffledTrain.slice(0, remainingValid));
    trainExamples = shuffledTrain.slice(remainingValid);
  }

  trainExamples = seededShuffle(trainExamples, 99);
  validExamples = seededShuffle(validExamples, 100);

  const trainPath = `${outputDir}/train.jsonl`;
  const validPath = `${outputDir}/valid.jsonl`;
  await writeJsonlFile(trainPath, trainExamples);
  await writeJsonlFile(validPath, validExamples);

  console.log(`\nWrote ${trainExamples.length} train -> ${trainPath}`);
  console.log(`Wrote ${validExamples.length} valid -> ${validPath}`);

  const avgUserLen = Math.round(
    [...trainExamples, ...validExamples].reduce((sum, e) => sum + e.messages[1]!.content.length, 0) /
      (trainExamples.length + validExamples.length),
  );
  const avgAssistantLen = Math.round(
    [...trainExamples, ...validExamples].reduce((sum, e) => sum + e.messages[2]!.content.length, 0) /
      (trainExamples.length + validExamples.length),
  );

  console.log(`Avg user message:       ${avgUserLen} chars (~${Math.round(avgUserLen / 4)} tokens)`);
  console.log(`Avg assistant response: ${avgAssistantLen} chars (~${Math.round(avgAssistantLen / 4)} tokens)`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

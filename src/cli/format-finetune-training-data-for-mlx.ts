import { readFile } from "node:fs/promises";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";
import type { FitLabel } from "../schema.js";

type ScoredJob = {
  job_id: string;
  title: string;
  company: string;
  location?: string;
  jd_text: string;
  label: FitLabel;
  score: number;
  reasoning: string;
  loc: number;
  role: number;
  tech: number;
  comp: number;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type MLXTrainingExample = {
  messages: ChatMessage[];
};

/**
 * Seeded shuffle (Fisher-Yates) for deterministic train/valid splits.
 */
function seededShuffle<T>(arr: T[], seed: number): T[] {
  const result = [...arr];
  // Simple mulberry32 PRNG
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

/**
 * Build the user message by filling the prompt template with job variables.
 */
function buildUserMessage(
  promptTemplate: string,
  job: ScoredJob,
): string {
  return promptTemplate
    .replace(/\{\{job_title\}\}/g, job.title)
    .replace(/\{\{job_location\}\}/g, job.location ?? "")
    .replace(/\{\{jd_text\}\}/g, job.jd_text);
}

/**
 * Build the assistant response — the target JSON the model should output.
 * Uses golden score/label (ground truth) with computed category breakdowns.
 */
function buildAssistantResponse(job: ScoredJob): string {
  // Generate a brief reasoning string from the breakdown
  const parts: string[] = [];

  if (job.loc === 25) parts.push("London/Remote UK (+25)");
  else if (job.loc === 10) parts.push("UK outside London (+10)");
  else if (job.loc === -50) parts.push("Outside UK (-50)");
  else if (job.loc === 0) parts.push("Location unclear (0)");

  if (job.role === 25) parts.push("senior-level role (+25)");
  else if (job.role === 15) parts.push("mid-level role (+15)");
  else if (job.role === 0) parts.push("no seniority keyword (0)");

  if (job.tech > 0) parts.push(`tech stack (${job.tech})`);
  else parts.push("no matching tech (0)");

  if (job.comp === 25) parts.push("salary ≥£100k (+25)");
  else if (job.comp === 15) parts.push("salary £75-99k (+15)");
  else if (job.comp === 5) parts.push("salary £55-74k (+5)");
  else if (job.comp === -30) parts.push("salary <£45k (-30)");
  else parts.push("no GBP salary (0)");

  const reasoning = parts.join(", ") + `.`;

  const output = {
    loc: job.loc,
    role: job.role,
    tech: job.tech,
    comp: job.comp,
    score: job.score,
    label: job.label,
    reasoning,
  };

  return JSON.stringify(output);
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ?? "data/finetune/train.jsonl";
  const outputDir =
    getStringArg(args, "output-dir") ?? "data/finetune/mlx";
  const promptPath =
    getStringArg(args, "prompt") ?? "prompts/scorer_v9.txt";
  const validPct = getNumberArg(args, "valid-pct") ?? 0;
  // Default to Qwen3 for backwards compatibility (existing teacher training)
  const modelName = getStringArg(args, "model") ?? "qwen3";

  // Read the prompt template
  const promptTemplate = await readFile(promptPath, "utf8");
  console.log(
    `Loaded prompt template from ${promptPath} (${promptTemplate.length} chars)`,
  );

  // Read scored training jobs
  const jobs = await readJsonlFile<ScoredJob>(inputPath);
  console.log(`Loaded ${jobs.length} training jobs from ${inputPath}`);

  // Validate that jobs have breakdown fields
  const missingBreakdowns = jobs.filter(
    (j) =>
      j.loc === undefined ||
      j.role === undefined ||
      j.tech === undefined ||
      j.comp === undefined,
  );
  if (missingBreakdowns.length > 0) {
    console.error(
      `${missingBreakdowns.length} jobs missing breakdown fields (loc/role/tech/comp).`,
    );
    console.error("Run golden:breakdowns first to compute these.");
    process.exit(1);
  }

  // /no_think is Qwen3-specific — other models get a neutral system message
  const systemMsg = modelName.toLowerCase().includes("qwen3")
    ? "/no_think"
    : "Respond with JSON only.";
  console.log(`System message: "${systemMsg}"`);

  // Format as MLX chat training examples
  const examples: MLXTrainingExample[] = jobs.map((job) => ({
    messages: [
      { role: "system" as const, content: systemMsg },
      { role: "user" as const, content: buildUserMessage(promptTemplate, job) },
      { role: "assistant" as const, content: buildAssistantResponse(job) },
    ],
  }));

  // Split into train/valid if --valid-pct is set
  let trainExamples = examples;
  let validExamples: MLXTrainingExample[] = [];

  if (validPct > 0 && validPct < 100) {
    const shuffled = seededShuffle(examples, 42);
    const validCount = Math.max(1, Math.round(shuffled.length * validPct / 100));
    validExamples = shuffled.slice(0, validCount);
    trainExamples = shuffled.slice(validCount);
    console.log(
      `\nSplit: ${trainExamples.length} train, ${validExamples.length} valid (${validPct}%, seed=42)`,
    );
  }

  // Write training file
  const trainPath = `${outputDir}/train.jsonl`;
  await writeJsonlFile(trainPath, trainExamples);
  console.log(`\nWrote ${trainExamples.length} training examples to ${trainPath}`);

  // Write validation file if split was requested
  if (validExamples.length > 0) {
    const validPath = `${outputDir}/valid.jsonl`;
    await writeJsonlFile(validPath, validExamples);
    console.log(`Wrote ${validExamples.length} validation examples to ${validPath}`);
  }

  // Print a sample
  const sample = examples[0];
  if (sample) {
    console.log("\n=== Sample training example ===");
    console.log(`System: ${sample.messages[0]!.content}`);
    console.log(
      `User: ${sample.messages[1]!.content.slice(0, 100)}...`,
    );
    console.log(`Assistant: ${sample.messages[2]!.content}`);
  }

  // Print stats
  const avgUserLen = Math.round(
    examples.reduce(
      (sum, e) => sum + e.messages[1]!.content.length,
      0,
    ) / examples.length,
  );
  const avgAssistantLen = Math.round(
    examples.reduce(
      (sum, e) => sum + e.messages[2]!.content.length,
      0,
    ) / examples.length,
  );
  console.log(
    `\nAvg user message: ${avgUserLen} chars, avg assistant response: ${avgAssistantLen} chars`,
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

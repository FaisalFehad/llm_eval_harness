import { readFile } from "node:fs/promises";

import { getStringArg, parseArgs } from "../lib/args.js";
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
 * Build the user message by filling the v9 prompt template with job variables.
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

  // Format as MLX chat training examples
  const examples: MLXTrainingExample[] = jobs.map((job) => ({
    messages: [
      { role: "system" as const, content: "/no_think" },
      { role: "user" as const, content: buildUserMessage(promptTemplate, job) },
      { role: "assistant" as const, content: buildAssistantResponse(job) },
    ],
  }));

  // Write training file
  const trainPath = `${outputDir}/train.jsonl`;
  await writeJsonlFile(trainPath, examples);
  console.log(`\nWrote ${examples.length} training examples to ${trainPath}`);

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

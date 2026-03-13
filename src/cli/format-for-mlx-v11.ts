/**
 * Convert v11 labeled data into MLX chat format for training.
 *
 * Usage:
 *   npx tsx src/cli/format-for-mlx-v11.ts \
 *     --train data/v11/train_labeled.jsonl \
 *     --valid data/v11/valid_labeled.jsonl \
 *     --prompt prompts/student_v11.txt \
 *     --prompt-tokens prompts/student_v11_tokens.txt \
 *     --out-dir data/v11
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { parseArgs, getStringArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

// Types

type LabeledRow = {
  job_id: string;
  title: string;
  company: string;
  job_location: string;
  jd_text: string;
  loc_raw?: string | null;
  loc: string;
  arr_raw?: string | null;
  arr: string;
  sen: string;
  tech_raw?: string | null;
  tech: string[];
  comp_raw?: string | null;
  comp: string;
  label: string;
};

type ChatMessage = { role: "system" | "user" | "assistant"; content: string };
type MLXExample = { messages: ChatMessage[] };

function buildUserMessage(template: string, job: LabeledRow): string {
  return template
    .replace(/\{\{job_title\}\}/g, job.title)
    .replace(/\{\{job_location\}\}/g, job.job_location ?? "")
    .replace(/\{\{jd_text\}\}/g, job.jd_text);
}

function buildAssistantFull(job: LabeledRow): string {
  const payload = {
    loc_raw: job.loc_raw ?? null,
    loc: job.loc,
    arr_raw: job.arr_raw ?? null,
    arr: job.arr,
    sen: job.sen,
    tech_raw: job.tech_raw ?? null,
    tech: job.tech,
    comp_raw: job.comp_raw ?? null,
    comp: job.comp,
  };
  return JSON.stringify(payload);
}

function buildAssistantTokens(job: LabeledRow): string {
  const payload = {
    loc: job.loc,
    arr: job.arr,
    sen: job.sen,
    tech: job.tech,
    comp: job.comp,
  };
  return JSON.stringify(payload);
}

async function main(): Promise<void> {
  const args = parseArgs();
  const trainPath = getStringArg(args, "train") ?? "data/v11/train_labeled.jsonl";
  const validPath = getStringArg(args, "valid") ?? "data/v11/valid_labeled.jsonl";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/student_v11.txt";
  const promptTokensPath = getStringArg(args, "prompt-tokens") ?? "prompts/student_v11_tokens.txt";
  const outDir = getStringArg(args, "out-dir") ?? "data/v11";

  const [trainRows, validRows] = await Promise.all([
    readJsonlFile<LabeledRow>(trainPath),
    readJsonlFile<LabeledRow>(validPath),
  ]);
  const promptFull = fs.readFileSync(promptPath, "utf8");
  const promptTokens = fs.readFileSync(promptTokensPath, "utf8");

  fs.mkdirSync(path.join(outDir, "mlx"), { recursive: true });
  fs.mkdirSync(path.join(outDir, "mlx_tokens"), { recursive: true });

  const toExamples = (rows: LabeledRow[], prompt: string, builder: (job: LabeledRow) => string): MLXExample[] =>
    rows.map((job) => ({
      messages: [
        { role: "system", content: "Respond with JSON only." },
        { role: "user", content: buildUserMessage(prompt, job) },
        { role: "assistant", content: builder(job) },
      ],
    }));

  await writeJsonlFile(path.join(outDir, "mlx", "train.jsonl"), toExamples(trainRows, promptFull, buildAssistantFull));
  await writeJsonlFile(path.join(outDir, "mlx", "valid.jsonl"), toExamples(validRows, promptFull, buildAssistantFull));
  await writeJsonlFile(path.join(outDir, "mlx_tokens", "train.jsonl"), toExamples(trainRows, promptTokens, buildAssistantTokens));
  await writeJsonlFile(path.join(outDir, "mlx_tokens", "valid.jsonl"), toExamples(validRows, promptTokens, buildAssistantTokens));

  console.log(`Wrote MLX chat data to ${path.join(outDir, "mlx")}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

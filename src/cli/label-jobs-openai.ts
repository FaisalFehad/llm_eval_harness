/**
 * Label jobs using OpenAI API (GPT-4o mini by default).
 *
 * Reads preprocessed JSONL jobs, sends each through the scoring prompt,
 * parses JSON responses, and writes labeled output compatible with
 * the rest of the student training pipeline.
 *
 * Usage:
 *   npx tsx src/cli/label-jobs-openai.ts \
 *     --input "data/Student Training Data/preprocessed_500.jsonl" \
 *     --output "data/Student Training Data/teacher_labeled_500.jsonl" \
 *     --prompt prompts/scorer_v9.8.txt \
 *     --model gpt-4o-mini \
 *     --concurrency 10
 *
 * Requires OPENAI_API_KEY environment variable.
 */

import * as fs from "node:fs";
import OpenAI from "openai";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";

// ── Types ────────────────────────────────────────────────────────────────────

type InputJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  source_url?: string;
  salary_field?: string;
  augmentation_type?: string;
  source_job_id?: string;
};

type ScoredOutput = {
  loc: number;
  role: number;
  tech: number;
  comp: number;
  score: number;
  label: "good_fit" | "maybe" | "bad_fit";
  reasoning: string;
  loc_reason?: string;
  role_reason?: string;
  tech_reason?: string;
  comp_reason?: string;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseJsonOutput(text: string): ScoredOutput | null {
  text = text.trim();

  // Try direct parse
  try {
    const parsed = JSON.parse(text);
    if (validateLabel(parsed)) return parsed as ScoredOutput;
  } catch {
    // fall through
  }

  // Try extracting JSON from markdown code blocks or surrounding text
  const match = text.match(/\{[^{}]+\}/);
  if (match) {
    try {
      const parsed = JSON.parse(match[0]);
      if (validateLabel(parsed)) return parsed as ScoredOutput;
    } catch {
      // fall through
    }
  }

  return null;
}

function validateLabel(parsed: Record<string, unknown>): boolean {
  const required = ["loc", "role", "tech", "comp", "score", "label"];
  if (!required.every((k) => k in parsed)) return false;
  if (!["good_fit", "maybe", "bad_fit"].includes(parsed.label as string)) return false;
  return true;
}

/**
 * Process a batch of jobs with concurrency limit.
 */
async function processWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      results[index] = await fn(items[index]!, index);
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, () => worker());
  await Promise.all(workers);
  return results;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ??
    "data/Student Training Data/preprocessed_500.jsonl";
  const outputPath =
    getStringArg(args, "output") ??
    "data/Student Training Data/teacher_labeled_500.jsonl";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/scorer_v9.8.txt";
  const modelId = getStringArg(args, "model") ?? "gpt-4o-mini";
  const concurrency = getNumberArg(args, "concurrency") ?? 5;

  // Check API key
  if (!process.env.OPENAI_API_KEY) {
    console.error("ERROR: OPENAI_API_KEY environment variable is required");
    process.exit(1);
  }

  const client = new OpenAI();

  // Load prompt template
  const promptTemplate = fs.readFileSync(promptPath, "utf-8");
  console.log(`Prompt: ${promptPath} (${Math.round(promptTemplate.length / 4)} est. tokens)`);
  console.log(`Model: ${modelId}`);
  console.log(`Concurrency: ${concurrency}`);

  // Load input jobs
  const jobs = await readJsonlFile<InputJob>(inputPath);
  console.log(`Loaded ${jobs.length} jobs from ${inputPath}`);
  console.log("─".repeat(60));

  // Track results
  const labeled: Array<Record<string, unknown>> = [];
  const failures: Array<{ index: number; title: string; raw: string }> = [];
  let completed = 0;

  const startTime = Date.now();

  await processWithConcurrency(jobs, concurrency, async (job, index) => {
    const promptText = promptTemplate
      .replace(/\{\{job_title\}\}/g, job.title)
      .replace(/\{\{job_location\}\}/g, job.location ?? "")
      .replace(/\{\{jd_text\}\}/g, job.jd_text);

    try {
      // Retry with short backoff for rate limits
      let content = "";
      const maxRetries = 3;
      for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
          const isO = modelId.startsWith("o");
          const response = await client.chat.completions.create({
            model: modelId,
            messages: [
              { role: isO ? "developer" : "system", content: "Respond with JSON only." },
              { role: "user", content: promptText },
            ],
            ...(isO
              ? { max_completion_tokens: 8000 }
              : { max_tokens: 350, temperature: 0 }),
          });
          const choice = response.choices[0];
          content = choice?.message?.content ?? "";
          // Retry if empty response (content filter or transient issue)
          if (!content && attempt < maxRetries) {
            const reason = choice?.finish_reason ?? "unknown";
            console.warn(`  [${index + 1}] Empty response (finish_reason=${reason}), retrying...`);
            await new Promise((r) => setTimeout(r, 2000));
            continue;
          }
          break;
        } catch (retryErr) {
          const msg = retryErr instanceof Error ? retryErr.message : String(retryErr);
          if (msg.includes("429") && attempt < maxRetries) {
            await new Promise((r) => setTimeout(r, 1500));
            continue;
          }
          throw retryErr;
        }
      }

      const parsed = parseJsonOutput(content);

      if (!parsed) {
        failures.push({ index: index + 1, title: job.title, raw: content.slice(0, 200) });
      } else {
        const scoredJob: Record<string, unknown> = {
          job_id: job.job_id,
          title: job.title,
          company: job.company,
          location: job.location,
          jd_text: job.jd_text,
          label: parsed.label,
          score: parsed.score,
          loc: parsed.loc,
          role: parsed.role,
          tech: parsed.tech,
          comp: parsed.comp,
          reasoning: parsed.reasoning ?? "",
          loc_reason: parsed.loc_reason ?? "",
          role_reason: parsed.role_reason ?? "",
          tech_reason: parsed.tech_reason ?? "",
          comp_reason: parsed.comp_reason ?? "",
        };

        if (job.source_url) scoredJob.source_url = job.source_url;
        if (job.augmentation_type) scoredJob.augmentation_type = job.augmentation_type;
        if (job.source_job_id) scoredJob.source_job_id = job.source_job_id;

        labeled.push(scoredJob);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      failures.push({ index: index + 1, title: job.title, raw: `API ERROR: ${msg}` });
    }

    completed++;
    if (completed % 50 === 0 || completed === jobs.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(
        `  [${completed}/${jobs.length}] ${labeled.length} labeled, ` +
        `${failures.length} failures (${elapsed}s)`,
      );
    }
  });

  // Write output — streaming write to preserve order by job_id
  labeled.sort((a, b) => String(a.job_id).localeCompare(String(b.job_id)));
  const outStream = fs.createWriteStream(outputPath);
  for (const job of labeled) {
    outStream.write(JSON.stringify(job) + "\n");
  }
  outStream.end();

  // Write failures log
  if (failures.length > 0) {
    const failPath = outputPath.replace(/\.jsonl$/, ".failures.jsonl");
    const failStream = fs.createWriteStream(failPath);
    for (const f of failures) {
      failStream.write(JSON.stringify(f) + "\n");
    }
    failStream.end();
    console.log(`Failures log: ${failPath}`);
  }

  // Summary
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log("\n" + "═".repeat(60));
  console.log("LABELING COMPLETE");
  console.log("═".repeat(60));
  console.log(`Input:          ${jobs.length} jobs`);
  console.log(`Labeled:        ${labeled.length}`);
  console.log(`Parse failures: ${failures.length}`);
  console.log(`Time:           ${elapsed}s`);
  console.log(`Output:         ${outputPath}`);

  // Label distribution
  const dist: Record<string, number> = {};
  for (const j of labeled) {
    const l = j.label as string;
    dist[l] = (dist[l] ?? 0) + 1;
  }
  console.log("\nLabel distribution:");
  for (const lbl of ["good_fit", "maybe", "bad_fit"]) {
    console.log(`  ${lbl.padEnd(10)}: ${dist[lbl] ?? 0}`);
  }

  // Comp distribution
  const compDist: Record<number, number> = {};
  for (const j of labeled) {
    const c = j.comp as number;
    compDist[c] = (compDist[c] ?? 0) + 1;
  }
  console.log("\nComp distribution:");
  for (const [c, n] of Object.entries(compDist).sort((a, b) => Number(a[0]) - Number(b[0]))) {
    console.log(`  comp=${c}: ${n}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

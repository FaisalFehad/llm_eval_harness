/**
 * Label jobs using OpenAI API with V5 semantic token teacher prompt.
 *
 * Reads input JSONL, sends each through the teacher prompt,
 * validates semantic tokens, cross-checks reasoning, and writes
 * labeled output with both tokens and computed scores.
 *
 * Usage:
 *   npx tsx src/cli/label-jobs.ts \
 *     --input data/v5/all_input_pool.jsonl \
 *     --output data/v5/all_labeled_pool.jsonl \
 *     --prompt prompts/teacher_v5.txt \
 *     --model gpt-4o-mini \
 *     --concurrency 10
 *
 * Requires OPENAI_API_KEY environment variable.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import OpenAI from "openai";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";
import {
  validateSemanticPrediction,
  computeFromTokens,
  crossCheckReasoning,
  type SemanticPrediction,
} from "../lib/semantic-tokens.js";

// ── Types ────────────────────────────────────────────────────────────────────

type InputJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
  source_file?: string;
};

type LabeledJob = InputJob & {
  // Semantic tokens
  loc: string;
  role: string;
  tech: string;
  comp: string;
  reasoning: string;
  // Computed scores (from code layer)
  loc_score: number;
  role_score: number;
  tech_score: number;
  comp_score: number;
  score: number;
  label: string;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseJsonOutput(text: string): Record<string, unknown> | null {
  text = text.trim();

  // Try direct parse
  try {
    return JSON.parse(text);
  } catch {
    // fall through
  }

  // Try extracting JSON from markdown code blocks or surrounding text
  const match = text.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      return JSON.parse(match[0]);
    } catch {
      // fall through
    }
  }

  return null;
}

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

  const workers = Array.from(
    { length: Math.min(concurrency, items.length) },
    () => worker(),
  );
  await Promise.all(workers);
  return results;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/v5/all_input_pool.jsonl";
  const outputPath = getStringArg(args, "output") ?? "data/v5/all_labeled_pool.jsonl";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/teacher_v5.txt";
  const modelId = getStringArg(args, "model") ?? "gpt-4o-mini";
  const concurrency = getNumberArg(args, "concurrency") ?? 10;

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
  const labeled: LabeledJob[] = [];
  const failures: Array<{ index: number; job_id: string; title: string; error: string; raw?: string }> = [];
  const inconsistencies: Array<{ job_id: string; title: string; issues: string[] }> = [];
  const fuzzyFixes: Array<{ job_id: string; corrections: string[] }> = [];
  let completed = 0;

  const startTime = Date.now();

  // ── Per-run log file ────────────────────────────────────────────────
  const logDir = path.join(path.dirname(outputPath), "labeling_runs");
  if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const logPath = path.join(logDir, `${timestamp}.log.jsonl`);
  const logStream = fs.createWriteStream(logPath, { flags: "a" });
  // Write header entry
  logStream.write(JSON.stringify({
    _type: "run_start",
    timestamp: new Date().toISOString(),
    input: inputPath,
    output: outputPath,
    prompt: promptPath,
    model: modelId,
    concurrency,
    total_jobs: jobs.length,
  }) + "\n");
  console.log(`Run log: ${logPath}`);

  await processWithConcurrency(jobs, concurrency, async (job, index) => {
    const promptText = promptTemplate
      .replace(/\{\{job_title\}\}/g, job.title)
      .replace(/\{\{job_location\}\}/g, job.location ?? "")
      .replace(/\{\{jd_text\}\}/g, job.jd_text);

    const jobStartMs = Date.now();
    let promptTokens = 0;
    let completionTokens = 0;

    try {
      let content = "";
      const maxRetries = 8;
      for (let attempt = 0; attempt <= maxRetries; attempt++) {
        try {
          const response = await client.chat.completions.create({
            model: modelId,
            messages: [
              { role: "system", content: "Respond with JSON only." },
              { role: "user", content: promptText },
            ],
            max_tokens: 500,
            temperature: 0,
          });
          const choice = response.choices[0];
          content = choice?.message?.content ?? "";
          promptTokens = response.usage?.prompt_tokens ?? 0;
          completionTokens = response.usage?.completion_tokens ?? 0;
          if (!content && attempt < maxRetries) {
            const reason = choice?.finish_reason ?? "unknown";
            console.warn(`  [${index + 1}] Empty response (${reason}), retrying...`);
            await new Promise((r) => setTimeout(r, 2000));
            continue;
          }
          break;
        } catch (retryErr) {
          const msg = retryErr instanceof Error ? retryErr.message : String(retryErr);
          if (msg.includes("429") && attempt < maxRetries) {
            // Exponential backoff: 2s, 4s, 8s, 16s, 32s...
            const waitMs = 2000 * Math.pow(2, attempt);
            if (attempt >= 2) {
              console.warn(`  [${index + 1}] Rate limited (attempt ${attempt + 1}), waiting ${(waitMs / 1000).toFixed(0)}s...`);
            }
            await new Promise((r) => setTimeout(r, waitMs));
            continue;
          }
          throw retryErr;
        }
      }

      // Parse JSON
      const parsed = parseJsonOutput(content);
      if (!parsed) {
        const err = "JSON parse failure";
        failures.push({ index: index + 1, job_id: job.job_id, title: job.title, error: err, raw: content.slice(0, 300) });
        logStream.write(JSON.stringify({
          _type: "job_fail", timestamp: new Date().toISOString(), index: index + 1,
          job_id: job.job_id, title: job.title, duration_ms: Date.now() - jobStartMs,
          prompt_tokens: promptTokens, completion_tokens: completionTokens, error: err,
        }) + "\n");
        return;
      }

      // Validate semantic tokens (with fuzzy matching)
      const validation = validateSemanticPrediction(parsed);
      if (!validation.valid) {
        const err = `Invalid tokens: ${validation.errors.join(", ")}`;
        failures.push({ index: index + 1, job_id: job.job_id, title: job.title, error: err, raw: content.slice(0, 300) });
        logStream.write(JSON.stringify({
          _type: "job_fail", timestamp: new Date().toISOString(), index: index + 1,
          job_id: job.job_id, title: job.title, duration_ms: Date.now() - jobStartMs,
          prompt_tokens: promptTokens, completion_tokens: completionTokens, error: err,
        }) + "\n");
        return;
      }

      const pred = validation.corrected!;

      // Track fuzzy corrections
      if (validation.fuzzyCorrections.length > 0) {
        fuzzyFixes.push({
          job_id: job.job_id,
          corrections: validation.fuzzyCorrections,
        });
      }

      // Cross-check reasoning vs tokens
      const consistency = crossCheckReasoning(pred);
      if (!consistency.consistent) {
        inconsistencies.push({
          job_id: job.job_id,
          title: job.title,
          issues: consistency.issues,
        });
      }

      // Compute scores from tokens
      const computed = computeFromTokens(pred);

      // Build labeled output
      const labeledJob: LabeledJob = {
        job_id: job.job_id,
        title: job.title,
        company: job.company,
        location: job.location,
        jd_text: job.jd_text,
        // Semantic tokens
        loc: pred.loc,
        role: pred.role,
        tech: pred.tech,
        comp: pred.comp,
        reasoning: pred.reasoning,
        // Computed scores
        loc_score: computed.loc_score,
        role_score: computed.role_score,
        tech_score: computed.tech_score,
        comp_score: computed.comp_score,
        score: computed.score,
        label: computed.label,
      };

      if (job.source_url) (labeledJob as Record<string, unknown>).source_url = job.source_url;
      if (job.augmentation_type) (labeledJob as Record<string, unknown>).augmentation_type = job.augmentation_type;
      if (job.source_job_id) (labeledJob as Record<string, unknown>).source_job_id = job.source_job_id;
      if (job.source_file) (labeledJob as Record<string, unknown>).source_file = job.source_file;

      labeled.push(labeledJob);

      // Log successful job
      logStream.write(JSON.stringify({
        _type: "job_ok",
        timestamp: new Date().toISOString(),
        index: index + 1,
        job_id: job.job_id,
        title: job.title,
        duration_ms: Date.now() - jobStartMs,
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        loc: pred.loc,
        role: pred.role,
        tech: pred.tech,
        comp: pred.comp,
        label: computed.label,
        score: computed.score,
        fuzzy_corrections: validation.fuzzyCorrections.length > 0 ? validation.fuzzyCorrections : undefined,
        inconsistencies: consistency.consistent ? undefined : consistency.issues,
      }) + "\n");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      failures.push({
        index: index + 1,
        job_id: job.job_id,
        title: job.title,
        error: `API ERROR: ${msg}`,
      });
      logStream.write(JSON.stringify({
        _type: "job_error",
        timestamp: new Date().toISOString(),
        index: index + 1,
        job_id: job.job_id,
        title: job.title,
        duration_ms: Date.now() - jobStartMs,
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        error: msg,
      }) + "\n");
    }

    completed++;
    if (completed % 100 === 0 || completed === jobs.length) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(
        `  [${completed}/${jobs.length}] ${labeled.length} labeled, ` +
        `${failures.length} failures (${elapsed}s)`,
      );
    }
  });

  // Sort and write output
  labeled.sort((a, b) => a.job_id.localeCompare(b.job_id));
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
    console.log(`\nFailures log: ${failPath}`);
  }

  // Write inconsistencies log
  if (inconsistencies.length > 0) {
    const incPath = outputPath.replace(/\.jsonl$/, ".inconsistencies.jsonl");
    const incStream = fs.createWriteStream(incPath);
    for (const inc of inconsistencies) {
      incStream.write(JSON.stringify(inc) + "\n");
    }
    incStream.end();
    console.log(`Inconsistencies log: ${incPath} (${inconsistencies.length} jobs)`);
  }

  // Write run summary to log
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  logStream.write(JSON.stringify({
    _type: "run_end",
    timestamp: new Date().toISOString(),
    total_jobs: jobs.length,
    labeled: labeled.length,
    failures: failures.length,
    inconsistencies: inconsistencies.length,
    fuzzy_corrections: fuzzyFixes.length,
    elapsed_s: parseFloat(elapsed),
  }) + "\n");
  await new Promise<void>((resolve) => logStream.end(resolve));

  // Summary
  console.log("\n" + "═".repeat(60));
  console.log("LABELING COMPLETE");
  console.log("═".repeat(60));
  console.log(`Input:              ${jobs.length} jobs`);
  console.log(`Labeled:            ${labeled.length}`);
  console.log(`Parse failures:     ${failures.length}`);
  console.log(`Inconsistencies:    ${inconsistencies.length}`);
  console.log(`Fuzzy corrections:  ${fuzzyFixes.length}`);
  console.log(`Time:               ${elapsed}s`);
  console.log(`Output:             ${outputPath}`);
  console.log(`Run log:            ${logPath}`);

  // Label distribution
  const labelDist: Record<string, number> = {};
  for (const j of labeled) {
    labelDist[j.label] = (labelDist[j.label] ?? 0) + 1;
  }
  console.log("\nLabel distribution (computed from tokens):");
  for (const lbl of ["good_fit", "maybe", "bad_fit"]) {
    console.log(`  ${lbl.padEnd(10)}: ${labelDist[lbl] ?? 0}`);
  }

  // Semantic token distributions
  console.log("\nSemantic token distributions:");
  for (const field of ["loc", "role", "tech", "comp"] as const) {
    const dist: Record<string, number> = {};
    for (const j of labeled) {
      const token = j[field];
      dist[token] = (dist[token] ?? 0) + 1;
    }
    console.log(`  ${field}:`);
    for (const [token, count] of Object.entries(dist).sort((a, b) => b[1] - a[1])) {
      console.log(`    ${token.padEnd(20)}: ${count}`);
    }
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

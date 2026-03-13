/**
 * Label jobs using OpenAI API with V7 semantic token teacher prompt.
 *
 * V7 version: uses 5 fields (loc, arr, sen, tech, comp) with 10 JSON keys.
 * Tech is an array of individual tokens. Scores are backward-compatible
 * with V6 (loc_score, role_score, tech_score, comp_score).
 *
 * Usage:
 *   npx tsx src/cli/label-jobs-v7.ts \
 *     --input data/v6/scraped_clean_for_labeling.jsonl \
 *     --output data/v7/labeled.jsonl \
 *     --prompt prompts/teacher_v7.txt \
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
  V7_TOKEN_FIELDS,
  type V7SemanticPrediction,
} from "../lib/semantic-tokens-v7.js";

// ── Types ────────────────────────────────────────────────────────────────────

type InputJob = {
  job_id: string;
  title: string;
  company: string;
  job_location: string;
  jd_text: string;
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
  source_file?: string;
};

// V7 output uses short field names (loc, arr, sen, tech, comp).
// Raw location string stored as "job_location" to avoid collision with "loc" token.
type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  job_location: string; // raw location string from input
  jd_text: string;
  // Per-field raw text + tokens (10-field interleaved format)
  loc_raw: string | null;
  loc: string; // V7 token: IN_LONDON | REMOTE | UK_OTHER | OUTSIDE_UK | UNK
  arr_raw: string | null;
  arr: string; // V7 token: REMOTE | HYBRID | IN_OFFICE | UNK
  sen_raw: string | null;
  sen: string; // V7 token: LEVEL_3 | LEVEL_2 | LEVEL_1
  tech_raw: string | null;
  tech: string[]; // V7 token array: ["NODE", "REACT", "JS_TS", "AI_ML"] or ["OOS"]
  comp_raw: string | null;
  comp: string; // V7 token: NO_GBP | UP_TO_ONLY | BELOW_45K | ...
  // Computed scores (from code layer, backward-compatible with V6)
  loc_score: number;
  role_score: number;
  tech_score: number;
  comp_score: number;
  score: number;
  label: string;
  // Optional fields passed through
  source_url?: string;
  augmentation_type?: string;
  source_job_id?: string;
  source_file?: string;
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
  const inputPath = getStringArg(args, "input") ?? "data/v6/scraped_clean_for_labeling.jsonl";
  const outputPath = getStringArg(args, "output") ?? "data/v7/labeled.jsonl";
  const promptPath = getStringArg(args, "prompt") ?? "prompts/teacher_v7.txt";
  const modelId = getStringArg(args, "model") ?? "gpt-4.1-mini";
  const temperature = getNumberArg(args, "temperature") ?? 0;
  const maxCompletionTokens = getNumberArg(args, "max-tokens") ?? 1200;
  const concurrency = getNumberArg(args, "concurrency") ?? 10;
  const maxFailures = getNumberArg(args, "max-failures") ?? 0; // 0 = no limit

  if (!process.env.OPENAI_API_KEY) {
    console.error("ERROR: OPENAI_API_KEY environment variable is required");
    process.exit(1);
  }

  // ── Pre-run validation ──────────────────────────────────────────────
  // Catch configuration errors before making any API calls.

  // 1. Input file must exist and be non-empty
  if (!fs.existsSync(inputPath)) {
    console.error(`ERROR: Input file not found: ${inputPath}`);
    process.exit(1);
  }
  const inputStat = fs.statSync(inputPath);
  if (inputStat.size === 0) {
    console.error(`ERROR: Input file is empty (0 bytes): ${inputPath}`);
    process.exit(1);
  }

  // 2. Prompt file must exist and contain expected placeholders
  if (!fs.existsSync(promptPath)) {
    console.error(`ERROR: Prompt file not found: ${promptPath}`);
    process.exit(1);
  }
  const promptTemplate = fs.readFileSync(promptPath, "utf-8");
  const missingPlaceholders = ["{{job_title}}", "{{job_location}}", "{{jd_text}}"]
    .filter((p) => !promptTemplate.includes(p));
  if (missingPlaceholders.length > 0) {
    console.error(`ERROR: Prompt missing placeholders: ${missingPlaceholders.join(", ")}`);
    console.error(`File: ${promptPath}`);
    process.exit(1);
  }

  // 3. Output file — warn if it already has data (will be overwritten)
  if (fs.existsSync(outputPath)) {
    const existingStat = fs.statSync(outputPath);
    if (existingStat.size > 0) {
      console.warn(`WARNING: Output file already exists (${Math.round(existingStat.size / 1024)}KB) and will be overwritten: ${outputPath}`);
    }
  }

  const client = new OpenAI();

  console.log(`Prompt: ${promptPath} (${Math.round(promptTemplate.length / 4)} est. tokens)`);
  console.log(`Model: ${modelId}`);
  console.log(`Concurrency: ${concurrency}`);
  if (maxFailures > 0) console.log(`Max failures: ${maxFailures} (will abort early)`);

  // 4. Load and validate input jobs
  const jobs = await readJsonlFile<InputJob>(inputPath);
  if (jobs.length === 0) {
    console.error(`ERROR: No jobs found in ${inputPath}`);
    process.exit(1);
  }

  // 5. Check required fields on every input job
  const hardBad = jobs.filter((j) => !j.title || !j.jd_text || j.jd_text.length < 20);
  if (hardBad.length > 0) {
    console.error(`ERROR: ${hardBad.length} jobs missing title or jd_text (< 20 chars)`);
    console.error(`First bad job: ${JSON.stringify(hardBad[0]).slice(0, 200)}`);
    process.exit(1);
  }
  // Auto-fix empty job_ids with a hash
  let autoIdCount = 0;
  for (const job of jobs) {
    if (!job.job_id) {
      job.job_id = `auto_${Buffer.from(job.title + "|" + job.jd_text.slice(0, 200)).toString("base64url").slice(0, 16)}`;
      autoIdCount++;
    }
  }
  if (autoIdCount > 0) {
    console.warn(`WARNING: Generated job_id for ${autoIdCount} job(s) with empty IDs`);
  }


  console.log(`Loaded ${jobs.length} jobs from ${inputPath}`);
  console.log("\u2500".repeat(60));

  // Track results
  const labeled: LabeledJob[] = [];
  const failures: Array<{ index: number; job_id: string; title: string; error: string; raw?: string }> = [];
  const inconsistencies: Array<{ job_id: string; title: string; issues: string[] }> = [];
  const fuzzyFixes: Array<{ job_id: string; corrections: string[] }> = [];
  let completed = 0;
  let aborted = false;

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
    version: "v7",
  }) + "\n");
  console.log(`Run log: ${logPath}`);

  // ── Preflight: validate model access with first job ──────────────────
  {
    const testJob = jobs[0]!;
    const testPrompt = promptTemplate
      .replace(/\{\{job_title\}\}/g, testJob.title)
      .replace(/\{\{job_location\}\}/g, testJob.job_location ?? "")
      .replace(/\{\{jd_text\}\}/g, testJob.jd_text);
    try {
      const testResp = await client.chat.completions.create({
        model: modelId,
        messages: [
          { role: "system", content: "Respond with JSON only." },
          { role: "user", content: testPrompt },
        ],
        max_completion_tokens: maxCompletionTokens,
        temperature: temperature,
      });
      const testContent = testResp.choices[0]?.message?.content ?? "";
      if (!testContent) {
        console.error("PREFLIGHT FAIL: First job returned empty response. Aborting.");
        process.exit(1);
      }
      const testParsed = parseJsonOutput(testContent);
      if (!testParsed) {
        console.error(`PREFLIGHT FAIL: First job returned unparseable JSON. Aborting.\nRaw: ${testContent.slice(0, 300)}`);
        process.exit(1);
      }
      const testValidation = validateSemanticPrediction(testParsed);
      if (!testValidation.valid) {
        console.error(`PREFLIGHT FAIL: First job has invalid tokens: ${testValidation.errors.join(", ")}. Aborting.\nRaw: ${testContent.slice(0, 300)}`);
        process.exit(1);
      }
      console.log(`Preflight OK: model=${modelId}, first job parsed and validated.`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`PREFLIGHT FAIL: ${msg}`);
      console.error("Check --model name and OPENAI_API_KEY. Aborting.");
      process.exit(1);
    }
  }

  await processWithConcurrency(jobs, concurrency, async (job, index) => {
    // Check abort flag before processing
    if (aborted) return;

    const promptText = promptTemplate
      .replace(/\{\{job_title\}\}/g, job.title)
      .replace(/\{\{job_location\}\}/g, job.job_location ?? "")
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
            max_completion_tokens: maxCompletionTokens,
            temperature: temperature,
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
          // Non-retryable errors: abort immediately
          if (msg.includes("401") || msg.includes("403") || msg.includes("404")) {
            throw retryErr;
          }
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

      // Validate V7 semantic tokens (with fuzzy matching)
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

      // Compute scores from tokens (backward-compatible with V6)
      const computed = computeFromTokens(pred);

      // Build labeled output
      const labeledJob: LabeledJob = {
        job_id: job.job_id,
        title: job.title,
        company: job.company,
        job_location: job.job_location, // raw location string stored as job_location
        jd_text: job.jd_text,
        // Per-field raw text + tokens (V7 interleaved)
        loc_raw: pred.loc_raw,
        loc: pred.loc,
        arr_raw: pred.arr_raw,
        arr: pred.arr,
        sen_raw: pred.sen_raw,
        sen: pred.sen,
        tech_raw: pred.tech_raw,
        tech: pred.tech,
        comp_raw: pred.comp_raw,
        comp: pred.comp,
        // Computed scores (V6-compatible names)
        loc_score: computed.loc_score,
        role_score: computed.role_score,
        tech_score: computed.tech_score,
        comp_score: computed.comp_score,
        score: computed.score,
        label: computed.label,
        // Optional pass-through fields
        ...(job.source_url && { source_url: job.source_url }),
        ...(job.augmentation_type && { augmentation_type: job.augmentation_type }),
        ...(job.source_job_id && { source_job_id: job.source_job_id }),
        ...(job.source_file && { source_file: job.source_file }),
      };

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
        loc_raw: pred.loc_raw,
        loc: pred.loc,
        arr_raw: pred.arr_raw,
        arr: pred.arr,
        sen_raw: pred.sen_raw,
        sen: pred.sen,
        tech_raw: pred.tech_raw,
        tech: pred.tech,
        comp_raw: pred.comp_raw,
        comp: pred.comp,
        label: computed.label,
        score: computed.score,
        fuzzy_corrections: validation.fuzzyCorrections.length > 0 ? validation.fuzzyCorrections : undefined,
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

    // Abort early if too many failures
    if (maxFailures > 0 && failures.length >= maxFailures && !aborted) {
      aborted = true;
      console.error(`\n\u26A0 ABORTING: ${failures.length} failures reached --max-failures ${maxFailures}`);
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
  console.log("\n" + "\u2550".repeat(60));
  console.log("LABELING COMPLETE (V7)");
  console.log("\u2550".repeat(60));
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

  // Semantic token distributions (all 5 V7 fields)
  console.log("\nSemantic token distributions:");
  for (const field of V7_TOKEN_FIELDS) {
    const dist: Record<string, number> = {};
    for (const j of labeled) {
      const value = (j as unknown as Record<string, unknown>)[field];
      if (Array.isArray(value)) {
        // Tech array: count each individual token
        for (const t of value) {
          dist[String(t)] = (dist[String(t)] ?? 0) + 1;
        }
      } else if (value) {
        dist[String(value)] = (dist[String(value)] ?? 0) + 1;
      }
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

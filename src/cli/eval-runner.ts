import { spawnSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getLlama, LlamaChatSession, resolveModelFile } from "node-llama-cpp";
import YAML from "yaml";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { ensureDir } from "../lib/paths.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ThresholdConfig = {
  job_count: number;
  max_parse_fail_pct: number;
  min_label_accuracy_pct: number;
  max_avg_seconds: number;
  max_score_mae?: number;
};

type ProviderConfig = {
  id: string;
  config?: Record<string, unknown>;
};

type PromptfooConfig = {
  description?: string;
  prompts: unknown[];
  providers: (string | ProviderConfig)[];
  tests: unknown[];
  [key: string]: unknown;
};

type TestJob = {
  description: string;
  vars: {
    jd_text: string;
    job_title?: string;
    job_location?: string;
    expected_label: string;
    expected_score: number;
  };
};

type JobResult = {
  job_description: string;
  expected_label: string;
  expected_score: number;
  predicted_label: string | null;
  predicted_score: number | null;
  predicted_reasoning: string | null;
  label_correct: boolean;
  score_error: number | null;
  latency_ms: number;
  parse_failed: boolean;
  raw_output: string;
};

type ConfusionMatrix = Record<string, Record<string, number>>;

type ModelResult = {
  model_id: string;
  label_accuracy_pct: number;
  parse_fail_pct: number;
  score_mae: number;
  score_bias: number;
  avg_seconds: number;
  total_tests: number;
  passed: boolean;
  fail_reasons: string[];
  early_exit: boolean;
  confusion_matrix: ConfusionMatrix;
  job_results: JobResult[];
};

type AccumulatedMetrics = {
  labelCorrect: number;
  parseFails: number;
  scoreAbsErrors: number[];
  scoreSignedErrors: number[];
  totalLatencyMs: number;
  counted: number;
};

type EvalResults = {
  tag: string;
  timestamp: string;
  thresholds: ThresholdConfig;
  models: ModelResult[];
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getProviderId(provider: string | ProviderConfig): string {
  return typeof provider === "string" ? provider : provider.id;
}

function getDisplayName(provider: string | ProviderConfig): string {
  return getProviderId(provider).replace(/^node-llama-cpp:/, "");
}

function getModelConfig(provider: string | ProviderConfig): {
  hfModel: string;
  temperature: number;
  noThink: boolean;
  contextSize: number;
} {
  if (typeof provider === "string") {
    throw new Error(
      "Provider must be a ProviderConfig object with config.hf_model",
    );
  }
  const cfg = provider.config ?? {};
  const hfModel = typeof cfg.hf_model === "string" ? cfg.hf_model : "";
  if (!hfModel)
    throw new Error(`Provider ${provider.id} is missing config.hf_model`);
  return {
    hfModel,
    temperature: typeof cfg.temperature === "number" ? cfg.temperature : 0,
    noThink: cfg.no_think === true,
    contextSize: typeof cfg.context_size === "number" ? cfg.context_size : 4096,
  };
}

function extractPromptTemplate(prompts: unknown[]): string {
  const first = prompts[0];
  if (typeof first === "string") return first;
  if (typeof first === "object" && first !== null && "raw" in first) {
    return (first as { raw: string }).raw;
  }
  throw new Error(
    "Cannot extract prompt template from config — expected string or {raw: string}",
  );
}

function buildPrompt(
  template: string,
  vars: Record<string, string | number | boolean | null | undefined>,
): string {
  return template.replace(/{{\s*([a-zA-Z0-9_]+)\s*}}/g, (_match, key) => {
    const value = vars[key];
    if (value === undefined || value === null) {
      return "";
    }
    return String(value);
  });
}

// ---------------------------------------------------------------------------
// node-llama-cpp inference
// ---------------------------------------------------------------------------

async function createResponseGrammar(
  llama: Awaited<ReturnType<typeof getLlama>>,
) {
  return await llama.createGrammarForJsonSchema({
    type: "object",
    properties: {
      loc: { type: "number" },
      role: { type: "number" },
      tech: { type: "number" },
      comp: { type: "number" },
      score: { type: "number" },
      label: { type: "string", enum: ["good_fit", "maybe", "bad_fit"] },
      reasoning: { type: "string" },
    },
  } as const);
}

async function callModel(
  model: Awaited<ReturnType<Awaited<ReturnType<typeof getLlama>>["loadModel"]>>,
  grammar: Awaited<ReturnType<typeof createResponseGrammar>>,
  prompt: string,
  temperature: number,
  noThink: boolean,
  contextSize: number,
  timeoutMs: number,
): Promise<{ content: string; durationMs: number; timedOut: boolean }> {
  const start = Date.now();

  // Try progressively smaller context sizes if VRAM is insufficient
  const sizesToTry = [contextSize, 2048, 1024, 512].filter((s) => s <= contextSize);
  let context: Awaited<ReturnType<typeof model.createContext>> | undefined;
  let lastErr: unknown;

  for (const size of sizesToTry) {
    try {
      context = await model.createContext({ contextSize: size });
      break;
    } catch (err) {
      lastErr = err;
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("too large for the available VRAM") || msg.includes("VRAM")) {
        console.log(`    Context size ${size} too large for VRAM, trying smaller...`);
        continue;
      }
      throw err; // non-VRAM error, don't retry
    }
  }
  if (!context) {
    throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
  }

  try {
    const session = new LlamaChatSession({
      contextSequence: context.getSequence(),
      systemPrompt: noThink ? "/no_think" : "",
    });

    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("MODEL_TIMEOUT")), timeoutMs);
    });

    try {
      const response = await Promise.race([
        session.prompt(prompt, { grammar, temperature }),
        timeoutPromise,
      ]);
      clearTimeout(timer);
      return {
        content: response,
        durationMs: Date.now() - start,
        timedOut: false,
      };
    } catch (err) {
      clearTimeout(timer);
      if (err instanceof Error && err.message === "MODEL_TIMEOUT") {
        return { content: "", durationMs: Date.now() - start, timedOut: true };
      }
      throw err;
    }
  } finally {
    await context.dispose();
  }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function computeMetrics(
  acc: AccumulatedMetrics,
  testsCompleted: number,
): {
  labelAccuracy: number;
  parseFailPct: number;
  scoreMae: number;
  scoreBias: number;
  avgSeconds: number;
} {
  const n = Math.max(acc.counted, 1);
  return {
    labelAccuracy: (acc.labelCorrect / n) * 100,
    parseFailPct: (acc.parseFails / Math.max(testsCompleted, 1)) * 100,
    scoreMae:
      acc.scoreAbsErrors.length > 0
        ? acc.scoreAbsErrors.reduce((a, b) => a + b, 0) /
          acc.scoreAbsErrors.length
        : 0,
    scoreBias:
      acc.scoreSignedErrors.length > 0
        ? acc.scoreSignedErrors.reduce((a, b) => a + b, 0) /
          acc.scoreSignedErrors.length
        : 0,
    avgSeconds: acc.totalLatencyMs / 1000 / n,
  };
}

function buildConfusionMatrix(jobResults: JobResult[]): ConfusionMatrix {
  const labels = ["good_fit", "maybe", "bad_fit"];
  const matrix: ConfusionMatrix = {};
  for (const actual of labels) {
    matrix[actual] = {};
    for (const predicted of labels) {
      matrix[actual][predicted] = 0;
    }
    matrix[actual]["parse_fail"] = 0;
  }
  for (const jr of jobResults) {
    const actual = jr.expected_label;
    if (jr.parse_failed || jr.predicted_label === null) {
      if (matrix[actual])
        matrix[actual]["parse_fail"] = (matrix[actual]["parse_fail"] ?? 0) + 1;
    } else {
      const predicted = jr.predicted_label;
      if (matrix[actual])
        matrix[actual][predicted] = (matrix[actual][predicted] ?? 0) + 1;
    }
  }
  return matrix;
}

function evaluateModel(
  thresholds: ThresholdConfig,
  metrics: {
    labelAccuracy: number;
    parseFailPct: number;
    scoreMae: number;
    avgSeconds: number;
  },
): { passed: boolean; reasons: string[] } {
  const reasons: string[] = [];

  if (metrics.parseFailPct > thresholds.max_parse_fail_pct) {
    reasons.push(
      `parse_fail=${metrics.parseFailPct.toFixed(0)}% > ${thresholds.max_parse_fail_pct}%`,
    );
  }
  if (metrics.labelAccuracy < thresholds.min_label_accuracy_pct) {
    reasons.push(
      `label_acc=${metrics.labelAccuracy.toFixed(0)}% < ${thresholds.min_label_accuracy_pct}%`,
    );
  }
  if (metrics.avgSeconds > thresholds.max_avg_seconds) {
    reasons.push(
      `avg_time=${metrics.avgSeconds.toFixed(1)}s > ${thresholds.max_avg_seconds}s`,
    );
  }
  if (
    thresholds.max_score_mae !== undefined &&
    metrics.scoreMae > thresholds.max_score_mae
  ) {
    reasons.push(
      `score_mae=${metrics.scoreMae.toFixed(1)} > ${thresholds.max_score_mae}`,
    );
  }

  return { passed: reasons.length === 0, reasons };
}

// ---------------------------------------------------------------------------
// Fail-fast
// ---------------------------------------------------------------------------

type EarlyExitCtx = {
  accumulated: AccumulatedMetrics;
  testsCompleted: number;
  consecutiveWrong: number;
  seenLabels: Set<string>;
  validPredictionCount: number;
  avgSeconds: number;
  maxAvgSeconds: number;
  jobIdx: number;
};

function checkEarlyExit(
  ctx: EarlyExitCtx,
): { exit: true; reason: string } | null {
  const {
    accumulated,
    testsCompleted,
    consecutiveWrong,
    seenLabels,
    validPredictionCount,
    avgSeconds,
    maxAvgSeconds,
    jobIdx,
  } = ctx;

  if (accumulated.parseFails === testsCompleted && testsCompleted >= 2) {
    return {
      exit: true,
      reason: `100% parse failures after ${testsCompleted} jobs — model cannot produce JSON`,
    };
  }
  if (jobIdx >= 1 && avgSeconds >= maxAvgSeconds * 2.0) {
    return {
      exit: true,
      reason: `avg_time=${avgSeconds.toFixed(1)}s (${(avgSeconds / maxAvgSeconds).toFixed(1)}x limit) after ${testsCompleted} jobs`,
    };
  }
  if (consecutiveWrong >= 25) {
    return {
      exit: true,
      reason: `${consecutiveWrong} consecutive wrong predictions after ${testsCompleted} jobs`,
    };
  }
  if (validPredictionCount >= 5 && seenLabels.size === 1) {
    return {
      exit: true,
      reason: `yes-man: all ${validPredictionCount} predictions are "${[...seenLabels][0]!}"`,
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

function printConfusionMatrix(matrix: ConfusionMatrix): void {
  const labels = ["good_fit", "maybe", "bad_fit"];
  const rows: string[] = [];
  for (const actual of labels) {
    const row = matrix[actual];
    if (!row) continue;
    const cells = labels.map((pred) => String(row[pred] ?? 0).padStart(3));
    const pf = row["parse_fail"] ?? 0;
    rows.push(
      `    ${actual.padEnd(10)} ${cells.join(" ")}${pf > 0 ? `  (${pf} parse fail)` : ""}`,
    );
  }
  if (rows.length > 0) {
    console.log(`  Confusion (actual \\ predicted):`);
    console.log(
      `    ${"".padEnd(10)} ${"g_f".padStart(3)} ${"may".padStart(3)} ${"b_f".padStart(3)}`,
    );
    rows.forEach((r) => console.log(r));
  }
}

function printSummary(
  modelResults: ModelResult[],
  thresholds: ThresholdConfig,
): void {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`EVAL RESULTS`);
  console.log(`${"=".repeat(70)}`);
  console.log(
    `${"Model".padEnd(25)} ${"Acc%".padStart(5)} ${"Parse%".padStart(7)} ${"MAE".padStart(5)} ${"Bias".padStart(6)} ${"Avg(s)".padStart(7)} ${"Tests".padStart(7)} ${"Status".padStart(7)}`,
  );
  console.log("-".repeat(75));
  for (const m of [...modelResults].sort(
    (a, b) => b.label_accuracy_pct - a.label_accuracy_pct,
  )) {
    const name = getDisplayName(m.model_id).padEnd(25);
    const status = m.passed ? "PASS" : m.early_exit ? "FAIL*" : "FAIL";
    const tests = `${m.total_tests}/${thresholds.job_count}`;
    const bias = `${m.score_bias >= 0 ? "+" : ""}${m.score_bias}`;
    console.log(
      `${name} ${String(m.label_accuracy_pct).padStart(5)} ${String(m.parse_fail_pct).padStart(7)} ${String(m.score_mae).padStart(5)} ${bias.padStart(6)} ${String(m.avg_seconds).padStart(7)} ${tests.padStart(7)} ${status.padStart(7)}`,
    );
  }
  if (modelResults.some((m) => m.early_exit)) {
    console.log("  * = early exit (fail-fast triggered)");
  }
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

function commandName(base: "npm" | "npx"): string {
  return process.platform === "win32" ? `${base}.cmd` : base;
}

async function loadBaseConfig(configPath: string): Promise<PromptfooConfig> {
  const raw = await readFile(configPath, "utf8");
  return YAML.parse(raw) as PromptfooConfig;
}

async function loadTestJobs(testFilePath: string): Promise<TestJob[]> {
  const raw = await readFile(testFilePath, "utf8");
  return YAML.parse(raw) as TestJob[];
}

async function generateTestSubset(
  count: number,
  seed: number,
  outputPath: string,
  balanced: boolean = false,
): Promise<void> {
  const args = [
    "tsx",
    "src/cli/sample-test-subset.ts",
    "--count",
    String(count),
    "--seed",
    String(seed),
    "--output",
    outputPath,
  ];
  if (balanced) args.push("--balanced");

  const result = spawnSync(commandName("npx"), args, {
    stdio: "inherit",
    env: process.env,
  });
  if (result.status !== 0) {
    throw new Error(`Failed to generate test subset (exit ${result.status})`);
  }
}

// ---------------------------------------------------------------------------
// Core eval
// ---------------------------------------------------------------------------

const DEFAULT_THRESHOLDS: ThresholdConfig = {
  job_count: 103,
  max_parse_fail_pct: 10,
  min_label_accuracy_pct: 40,
  max_avg_seconds: 120,
  max_score_mae: 30,
};

export type EvalOptions = {
  configPath: string;
  modelFilter?: string | undefined;
  jobCount: number;
  seed: number;
  tag: string;
  outputDir?: string | undefined;
};

export async function runEval(options: EvalOptions): Promise<EvalResults> {
  const { configPath, modelFilter, jobCount, seed, tag, outputDir } = options;

  const baseConfig = await loadBaseConfig(configPath);

  // Filter providers if --model specified
  let providers = baseConfig.providers;
  if (modelFilter) {
    const needle = modelFilter.toLowerCase();
    providers = providers.filter((p) => {
      const id = getProviderId(p).toLowerCase();
      const display = getDisplayName(p).toLowerCase();
      return (
        id === needle ||
        display === needle ||
        id.includes(needle) ||
        display.includes(needle)
      );
    });
    if (providers.length === 0) {
      console.error(
        `No providers matched --model "${modelFilter}". Available:\n` +
          baseConfig.providers
            .map((p) => `  ${getDisplayName(p)}  (${getProviderId(p)})`)
            .join("\n"),
      );
      process.exit(1);
    }
  }

  const thresholds: ThresholdConfig = {
    ...DEFAULT_THRESHOLDS,
    job_count: jobCount,
  };
  const evalDir = outputDir ?? path.join("results", "eval", tag || "latest");
  await ensureDir(evalDir);

  // Generate test subset
  const testFile = path.join("data", `promptfoo_tests_eval.yaml`);
  await generateTestSubset(jobCount, seed, testFile, jobCount <= 30);

  const jobs = await loadTestJobs(testFile);
  const promptTemplate = extractPromptTemplate(baseConfig.prompts);
  const perJobTimeoutMs = 420_000; // 7 min per job

  // Initialize node-llama-cpp once
  console.log("Initializing llama.cpp runtime...");
  const llama = await getLlama();
  const grammar = await createResponseGrammar(llama);
  const modelsDir = path.join(process.cwd(), "models");
  await ensureDir(modelsDir);

  console.log(`\n${"=".repeat(60)}`);
  console.log(
    `EVAL — ${jobCount} jobs, ${providers.length} model(s)${tag ? ` [${tag}]` : ""}`,
  );
  console.log(`${"=".repeat(60)}\n`);

  const allModelResults: ModelResult[] = [];

  for (let i = 0; i < providers.length; i++) {
    const provider = providers[i]!;
    const modelId = getProviderId(provider);
    const displayName = getDisplayName(provider);
    const { hfModel, temperature, noThink, contextSize } = getModelConfig(provider);

    console.log(`\n[${i + 1}/${providers.length}] Evaluating: ${displayName}`);
    console.log("-".repeat(40));

    // Resolve model (downloads from HF on first run)
    let model: Awaited<ReturnType<typeof llama.loadModel>>;
    try {
      console.log(`  Resolving model: ${hfModel}...`);
      const modelPath = await resolveModelFile(hfModel, modelsDir);
      console.log(`  Loading model into memory...`);
      model = await llama.loadModel({ modelPath });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  SKIP — failed to load model: ${msg}`);
      allModelResults.push({
        model_id: modelId,
        label_accuracy_pct: 0,
        parse_fail_pct: 100,
        score_mae: 0,
        score_bias: 0,
        avg_seconds: 0,
        total_tests: 0,
        passed: false,
        fail_reasons: [`Model load failed: ${msg}`],
        early_exit: true,
        confusion_matrix: buildConfusionMatrix([]),
        job_results: [],
      });
      continue;
    }

    const accumulated: AccumulatedMetrics = {
      labelCorrect: 0,
      parseFails: 0,
      scoreAbsErrors: [],
      scoreSignedErrors: [],
      totalLatencyMs: 0,
      counted: 0,
    };
    const jobResults: JobResult[] = [];
    let testsCompleted = 0;
    let earlyExit = false;
    let earlyExitReason = "";
    let consecutiveWrong = 0;
    const seenLabels = new Set<string>();
    let validPredictionCount = 0;

    for (let jobIdx = 0; jobIdx < jobs.length; jobIdx++) {
      const job = jobs[jobIdx]!;
      const prompt = buildPrompt(promptTemplate, job.vars);
      const expectedLabel = job.vars.expected_label;
      const expectedScore = job.vars.expected_score;

      let content: string;
      let durationMs: number;
      let timedOut: boolean;

      try {
        ({ content, durationMs, timedOut } = await callModel(
          model,
          grammar,
          prompt,
          temperature,
          noThink,
          contextSize,
          perJobTimeoutMs,
        ));
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.log(`  Job ${jobIdx + 1}: ERROR — ${errMsg}`);
        accumulated.parseFails++;
        accumulated.counted++;
        testsCompleted++;
        jobResults.push({
          job_description: job.description,
          expected_label: expectedLabel,
          expected_score: expectedScore,
          predicted_label: null,
          predicted_score: null,
          predicted_reasoning: null,
          label_correct: false,
          score_error: null,
          latency_ms: 0,
          parse_failed: true,
          raw_output: `ERROR: ${errMsg}`,
        });
        if (accumulated.parseFails === testsCompleted && testsCompleted >= 2) {
          earlyExit = true;
          earlyExitReason = `Consecutive errors after ${testsCompleted} jobs`;
          break;
        }
        continue;
      }

      testsCompleted++;
      accumulated.counted++;
      accumulated.totalLatencyMs += durationMs;

      if (timedOut) {
        earlyExit = true;
        earlyExitReason = `TIMEOUT on job ${jobIdx + 1} (>${perJobTimeoutMs / 1000}s)`;
        accumulated.parseFails++;
        jobResults.push({
          job_description: job.description,
          expected_label: expectedLabel,
          expected_score: expectedScore,
          predicted_label: null,
          predicted_score: null,
          predicted_reasoning: null,
          label_correct: false,
          score_error: null,
          latency_ms: durationMs,
          parse_failed: true,
          raw_output: "TIMEOUT",
        });
        break;
      }

      // With grammar-constrained decoding, parse failures should be rare
      let parsed: {
        label?: string;
        score?: number;
        reasoning?: string;
      } | null = null;
      let parseFailed = false;
      try {
        const raw: unknown = JSON.parse(content);
        if (typeof raw === "object" && raw !== null && !Array.isArray(raw)) {
          parsed = raw as {
            label?: string;
            score?: number;
            reasoning?: string;
          };
        } else {
          accumulated.parseFails++;
          parseFailed = true;
        }
      } catch {
        accumulated.parseFails++;
        parseFailed = true;
      }

      const predictedLabel =
        typeof parsed?.label === "string" ? parsed.label : null;
      const predictedScore =
        typeof parsed?.score === "number" ? parsed.score : null;
      const predictedReasoning =
        typeof parsed?.reasoning === "string" ? parsed.reasoning : null;
      const labelCorrect = predictedLabel === expectedLabel;
      const scoreError =
        predictedScore !== null ? predictedScore - expectedScore : null;

      if (!parseFailed && parsed !== null) {
        if (labelCorrect) {
          accumulated.labelCorrect++;
          consecutiveWrong = 0;
        } else {
          consecutiveWrong++;
        }
        if (predictedLabel !== null) {
          seenLabels.add(predictedLabel);
          validPredictionCount++;
        }
        if (predictedScore !== null) {
          accumulated.scoreAbsErrors.push(Math.abs(scoreError!));
          accumulated.scoreSignedErrors.push(scoreError!);
        }
      }

      jobResults.push({
        job_description: job.description,
        expected_label: expectedLabel,
        expected_score: expectedScore,
        predicted_label: predictedLabel,
        predicted_score: predictedScore,
        predicted_reasoning: predictedReasoning,
        label_correct: labelCorrect,
        score_error: scoreError,
        latency_ms: durationMs,
        parse_failed: parseFailed,
        raw_output: content,
      });

      const m = computeMetrics(accumulated, testsCompleted);
      console.log(
        `  Job ${testsCompleted}/${thresholds.job_count} | acc=${m.labelAccuracy.toFixed(0)}% parse_fail=${m.parseFailPct.toFixed(0)}% mae=${m.scoreMae.toFixed(1)} bias=${m.scoreBias >= 0 ? "+" : ""}${m.scoreBias.toFixed(1)} avg=${m.avgSeconds.toFixed(1)}s`,
      );

      const earlyExitResult = checkEarlyExit({
        accumulated,
        testsCompleted,
        consecutiveWrong,
        seenLabels,
        validPredictionCount,
        avgSeconds: m.avgSeconds,
        maxAvgSeconds: thresholds.max_avg_seconds,
        jobIdx,
      });
      if (earlyExitResult) {
        earlyExit = true;
        earlyExitReason = earlyExitResult.reason;
        break;
      }

      // Release memory between jobs to reduce RAM pressure
      global.gc?.();
      await new Promise((r) => setTimeout(r, 100));
    }

    // Dispose model from GPU memory before loading the next one
    await model.dispose();

    const finalMetrics = computeMetrics(accumulated, testsCompleted);
    const { passed, reasons } = earlyExit
      ? { passed: false, reasons: [earlyExitReason] }
      : evaluateModel(thresholds, finalMetrics);

    const confusionMatrix = buildConfusionMatrix(jobResults);

    const result: ModelResult = {
      model_id: modelId,
      label_accuracy_pct: Math.round(finalMetrics.labelAccuracy * 10) / 10,
      parse_fail_pct: Math.round(finalMetrics.parseFailPct * 10) / 10,
      score_mae: Math.round(finalMetrics.scoreMae * 10) / 10,
      score_bias: Math.round(finalMetrics.scoreBias * 10) / 10,
      avg_seconds: Math.round(finalMetrics.avgSeconds * 10) / 10,
      total_tests: testsCompleted,
      passed,
      fail_reasons: reasons,
      early_exit: earlyExit,
      confusion_matrix: confusionMatrix,
      job_results: jobResults,
    };
    allModelResults.push(result);

    const status = passed ? "PASS" : `FAIL${earlyExit ? "*" : ""}`;
    const biasStr = `${result.score_bias >= 0 ? "+" : ""}${result.score_bias}`;
    console.log(
      `  ${status} | acc=${result.label_accuracy_pct}% | parse_fail=${result.parse_fail_pct}% | mae=${result.score_mae} | bias=${biasStr} | avg=${result.avg_seconds}s | tests=${testsCompleted}/${thresholds.job_count}`,
    );
    if (reasons.length > 0) {
      console.log(`  Reasons: ${reasons.join(", ")}`);
    }
    printConfusionMatrix(confusionMatrix);
  }

  printSummary(allModelResults, thresholds);

  // Save results
  const evalResults: EvalResults = {
    tag: tag || "eval",
    timestamp: new Date().toISOString(),
    thresholds,
    models: allModelResults.map(({ job_results: _jr, ...rest }) => ({
      ...rest,
      job_results: [],
    })),
  };

  const resultsPath = path.join(evalDir, "eval_results.json");
  await writeFile(resultsPath, JSON.stringify(evalResults, null, 2), "utf8");
  console.log(`\nResults saved to ${resultsPath}`);

  // Save per-model detail files
  const detailsDir = path.join(evalDir, "details");
  await ensureDir(detailsDir);
  for (const m of allModelResults) {
    const safeName = getDisplayName(m.model_id).replace(/[/:]/g, "_");
    const detailPath = path.join(detailsDir, `${safeName}.json`);
    await writeFile(detailPath, JSON.stringify(m, null, 2), "utf8");
  }
  console.log(`Per-model diagnostics saved to ${detailsDir}/`);

  return evalResults;
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs();
  const configPath = getStringArg(args, "config") ?? "promptfooconfig.yaml";
  const modelFilter = getStringArg(args, "model");
  const jobCount = getNumberArg(args, "jobs") ?? 103;
  const rawSeed = getNumberArg(args, "seed") ?? 42;
  const seed = rawSeed % 2 === 0 ? rawSeed : rawSeed + 1;
  const tag = getStringArg(args, "tag") ?? "";

  await runEval({
    configPath,
    modelFilter: modelFilter ?? undefined,
    jobCount,
    seed,
    tag,
  });
}

// Only run when this file is the direct entry point (not when imported by prompt-lab)
const __filename = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === __filename) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}

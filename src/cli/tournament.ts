import { spawnSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { Ollama } from "ollama";
import YAML from "yaml";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { ensureDir } from "../lib/paths.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type RoundName = "smoke" | "qualifying" | "full" | "auto";

type ThresholdConfig = {
  job_count: number;
  max_parse_fail_pct: number;
  min_label_accuracy_pct: number;
  max_avg_seconds: number;
  max_score_mae?: number;
};

type Thresholds = {
  smoke: ThresholdConfig;
  qualifying: ThresholdConfig;
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

type RoundResults = {
  round: string;
  timestamp: string;
  thresholds: ThresholdConfig;
  models: ModelResult[];
  survivors: string[];
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function commandName(base: "npm" | "npx"): string {
  return process.platform === "win32" ? `${base}.cmd` : base;
}

function getProviderId(provider: string | ProviderConfig): string {
  return typeof provider === "string" ? provider : provider.id;
}

function getOllamaModelName(provider: string | ProviderConfig): string {
  return getProviderId(provider).replace("ollama:chat:", "");
}

function getProviderOllamaOptions(provider: string | ProviderConfig): {
  format?: "json";
  temperature: number;
} {
  if (typeof provider === "string") return { temperature: 0 };
  const cfg = provider.config ?? {};
  const opts: { format?: "json"; temperature: number } = {
    temperature: typeof cfg.temperature === "number" ? cfg.temperature : 0,
  };
  if (cfg.format === "json") opts.format = "json";
  return opts;
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

function buildPrompt(template: string, jdText: string): string {
  // Use a function replacer to prevent $ special patterns in jdText
  // (e.g. $&, $', $1) from being interpreted by String.replace.
  return template.replace("{{jd_text}}", () => jdText);
}

// ---------------------------------------------------------------------------
// Ollama helpers
// ---------------------------------------------------------------------------

/** Explicitly evict a model from Ollama's memory before loading the next one. */
async function unloadModel(client: Ollama, model: string): Promise<void> {
  try {
    await client.chat({
      model,
      messages: [],
      keep_alive: 0, // 0 = unload immediately
      stream: false,
    });
  } catch {
    // Ignore errors — the model may already be unloaded or may have crashed
  }
}

async function callOllama(
  client: Ollama,
  model: string,
  prompt: string,
  format: "json" | undefined,
  temperature: number,
  timeoutMs: number,
): Promise<{ content: string; durationMs: number; timedOut: boolean }> {
  const start = Date.now();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeoutSignal = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error("OLLAMA_TIMEOUT")), timeoutMs);
  });

  try {
    const response = await Promise.race([
      client.chat({
        model,
        messages: [{ role: "user", content: prompt }],
        ...(format !== undefined ? { format } : {}),
        // Cap context to 4096 tokens. Each job is scored independently (no
        // conversation history), so prompts are ~1500-2000 tokens max.
        // Default Ollama context is 32k-40k, which pre-allocates a huge KV
        // cache — qwen3:8b uses 11 GB at 40k ctx vs ~3-4 GB at 4k ctx.
        options: { temperature, num_ctx: 4096 },
        stream: false,
      }),
      timeoutSignal,
    ]);
    clearTimeout(timer);
    return {
      content: response.message.content,
      durationMs: Date.now() - start,
      timedOut: false,
    };
  } catch (err) {
    clearTimeout(timer);
    const durationMs = Date.now() - start;
    if (err instanceof Error && err.message === "OLLAMA_TIMEOUT") {
      return { content: "", durationMs, timedOut: true };
    }
    throw err;
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
    // Positive = model overscores, negative = model underscores
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
// Loaders
// ---------------------------------------------------------------------------

async function loadThresholds(thresholdsPath: string): Promise<Thresholds> {
  const raw = await readFile(thresholdsPath, "utf8");
  return JSON.parse(raw) as Thresholds;
}

async function loadBaseConfig(configPath: string): Promise<PromptfooConfig> {
  const raw = await readFile(configPath, "utf8");
  return YAML.parse(raw) as PromptfooConfig;
}

async function loadSurvivors(survivorsPath: string): Promise<string[]> {
  const raw = await readFile(survivorsPath, "utf8");
  const data = JSON.parse(raw) as RoundResults;
  return data.survivors;
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
// Tournament rounds
// ---------------------------------------------------------------------------

async function runRound(
  roundName: "smoke" | "qualifying",
  baseConfig: PromptfooConfig,
  thresholds: ThresholdConfig,
  providers: (string | ProviderConfig)[],
  tournamentDir: string,
  seed: number,
): Promise<RoundResults> {
  const roundDir = path.join(tournamentDir, roundName);
  await ensureDir(roundDir);

  // Smoke uses balanced sampling — golden set is skewed (52 bad_fit, 40 maybe, 11 good_fit)
  // so balanced ensures every label is fairly represented in 10 jobs.
  // Qualifying uses proportional (stratified) — 30 jobs is enough volume.
  const useBalanced = roundName === "smoke";
  const testFile = path.join("data", `promptfoo_tests_${roundName}.yaml`);

  console.log(`\n${"=".repeat(60)}`);
  console.log(
    `ROUND: ${roundName.toUpperCase()} — ${thresholds.job_count} jobs, ${providers.length} models (${useBalanced ? "balanced" : "stratified"} sampling)`,
  );
  console.log(`${"=".repeat(60)}\n`);

  await generateTestSubset(thresholds.job_count, seed, testFile, useBalanced);

  const jobs = await loadTestJobs(testFile);
  const promptTemplate = extractPromptTemplate(baseConfig.prompts);
  const perJobTimeoutMs = 420_000; // 7 min per job
  const ollamaClient = new Ollama({ host: "http://127.0.0.1:11434" });

  const modelResults: ModelResult[] = [];

  for (let i = 0; i < providers.length; i++) {
    const provider = providers[i]!;
    const modelId = getProviderId(provider);
    const modelName = getOllamaModelName(provider);
    const { format, temperature } = getProviderOllamaOptions(provider);

    console.log(`\n[${i + 1}/${providers.length}] Testing: ${modelName}`);
    console.log("-".repeat(40));

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
    let consecutiveErrors = 0;

    for (let jobIdx = 0; jobIdx < jobs.length; jobIdx++) {
      const job = jobs[jobIdx]!;
      const prompt = buildPrompt(promptTemplate, job.vars.jd_text);
      const expectedLabel = job.vars.expected_label;
      const expectedScore = job.vars.expected_score;

      let content: string;
      let durationMs: number;
      let timedOut: boolean;

      try {
        ({ content, durationMs, timedOut } = await callOllama(
          ollamaClient,
          modelName,
          prompt,
          format,
          temperature,
          perJobTimeoutMs,
        ));
      } catch (err) {
        // Connection refused, model not found, Ollama crash, etc.
        const errMsg = err instanceof Error ? err.message : String(err);
        console.log(`  Job ${jobIdx + 1}: ERROR — ${errMsg}`);
        accumulated.parseFails++;
        accumulated.counted++;
        testsCompleted++;
        consecutiveErrors++;
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
        // Fail-fast: all errors so far (model never loaded / not found)
        if (accumulated.parseFails === testsCompleted && testsCompleted >= 2) {
          earlyExit = true;
          earlyExitReason = `Connection/load error after ${testsCompleted} jobs`;
          break;
        }
        // Fail-fast: 3+ consecutive errors = Ollama likely crashed mid-run
        if (consecutiveErrors >= 3) {
          earlyExit = true;
          earlyExitReason = `${consecutiveErrors} consecutive errors — Ollama may have crashed (fetch failed)`;
          break;
        }
        continue;
      }

      testsCompleted++;
      accumulated.counted++;
      accumulated.totalLatencyMs += durationMs;
      consecutiveErrors = 0; // reset on any successful HTTP response (even timeouts)

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

      let parsed: {
        label?: string;
        score?: number;
        reasoning?: string;
      } | null = null;
      let parseFailed = false;
      try {
        const raw: unknown = JSON.parse(content);
        if (
          typeof raw === "object" &&
          raw !== null &&
          !Array.isArray(raw)
        ) {
          parsed = raw as { label?: string; score?: number; reasoning?: string };
        } else {
          // Valid JSON but not an object (string, number, array, etc.)
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
        if (labelCorrect) accumulated.labelCorrect++;
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

      // Fail-fast: 100% parse failures after ≥2 jobs (≥2 avoids dropping on a single fluke)
      if (accumulated.parseFails === testsCompleted && testsCompleted >= 2) {
        earlyExit = true;
        earlyExitReason = `100% parse failures after ${testsCompleted} jobs — model cannot produce JSON`;
        break;
      }
      // Fail-fast: way too slow — skip first job (Ollama cold-loads the model on job 1)
      if (jobIdx >= 1 && m.avgSeconds >= thresholds.max_avg_seconds * 2.0) {
        earlyExit = true;
        earlyExitReason = `avg_time=${m.avgSeconds.toFixed(1)}s (${(m.avgSeconds / thresholds.max_avg_seconds).toFixed(1)}x limit) after ${testsCompleted} jobs`;
        break;
      }
    }

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
    modelResults.push(result);

    // Unload the model from Ollama memory before moving to the next one.
    // This prevents the previous model from eating RAM while the next one loads.
    await unloadModel(ollamaClient, modelName);

    const status = passed ? "PASS" : `FAIL${earlyExit ? "*" : ""}`;
    const biasStr = `${result.score_bias >= 0 ? "+" : ""}${result.score_bias}`;
    console.log(
      `  ${status} | acc=${result.label_accuracy_pct}% | parse_fail=${result.parse_fail_pct}% | mae=${result.score_mae} | bias=${biasStr} | avg=${result.avg_seconds}s | tests=${testsCompleted}/${thresholds.job_count}`,
    );
    if (reasons.length > 0) {
      console.log(`  Reasons: ${reasons.join(", ")}`);
    }

    // Print confusion matrix inline
    const labels = ["good_fit", "maybe", "bad_fit"];
    const cmRows: string[] = [];
    for (const actual of labels) {
      const row = confusionMatrix[actual];
      if (!row) continue;
      const cells = labels.map((pred) => String(row[pred] ?? 0).padStart(3));
      const pf = row["parse_fail"] ?? 0;
      cmRows.push(
        `    ${actual.padEnd(10)} ${cells.join(" ")}${pf > 0 ? `  (${pf} parse fail)` : ""}`,
      );
    }
    if (cmRows.length > 0) {
      console.log(`  Confusion (actual \\ predicted):`);
      console.log(
        `    ${"".padEnd(10)} ${"g_f".padStart(3)} ${"may".padStart(3)} ${"b_f".padStart(3)}`,
      );
      cmRows.forEach((r) => console.log(r));
    }
  }

  const survivors = modelResults.filter((m) => m.passed).map((m) => m.model_id);

  // Summary table
  console.log(`\n${"=".repeat(70)}`);
  console.log(`${roundName.toUpperCase()} RESULTS`);
  console.log(`${"=".repeat(70)}`);
  console.log(
    `${"Model".padEnd(30)} ${"Acc%".padStart(5)} ${"Parse%".padStart(7)} ${"MAE".padStart(5)} ${"Bias".padStart(6)} ${"Avg(s)".padStart(7)} ${"Tests".padStart(7)} ${"Status".padStart(7)}`,
  );
  console.log("-".repeat(85));
  for (const m of modelResults.sort(
    (a, b) => b.label_accuracy_pct - a.label_accuracy_pct,
  )) {
    const name = m.model_id.replace("ollama:chat:", "").padEnd(30);
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
  console.log(
    `\nSurvivors: ${survivors.length}/${modelResults.length} — ${survivors.map((s) => s.replace("ollama:chat:", "")).join(", ") || "(none)"}`,
  );

  // Save results — main summary (without bulky job_results for the overview)
  const roundResults: RoundResults = {
    round: roundName,
    timestamp: new Date().toISOString(),
    thresholds,
    models: modelResults.map(({ job_results: _jr, ...rest }) => ({
      ...rest,
      job_results: [],
    })),
    survivors,
  };

  const resultsPath = path.join(roundDir, `${roundName}_results.json`);
  await writeFile(resultsPath, JSON.stringify(roundResults, null, 2), "utf8");
  console.log(`\nResults saved to ${resultsPath}`);

  // Save per-model diagnostic files with full job-level detail
  const detailsDir = path.join(roundDir, "details");
  await ensureDir(detailsDir);
  for (const m of modelResults) {
    const safeName = m.model_id
      .replace("ollama:chat:", "")
      .replace(/[/:]/g, "_");
    const detailPath = path.join(detailsDir, `${safeName}.json`);
    await writeFile(detailPath, JSON.stringify(m, null, 2), "utf8");
  }
  console.log(`Per-model diagnostics saved to ${detailsDir}/`);

  return roundResults;
}

async function runFullEval(
  baseConfig: PromptfooConfig,
  survivorIds: string[],
  tournamentDir: string,
): Promise<void> {
  const fullDir = path.join(tournamentDir, "full");
  await ensureDir(fullDir);

  console.log(`\n${"=".repeat(60)}`);
  console.log(
    `ROUND: FULL EVAL — all 103 jobs, ${survivorIds.length} finalist models`,
  );
  console.log(`${"=".repeat(60)}\n`);

  const survivorProviders = baseConfig.providers.filter((p) =>
    survivorIds.includes(getProviderId(p)),
  );

  if (survivorProviders.length === 0) {
    console.error("No survivors to run full eval on!");
    return;
  }

  // Build full test set
  const buildResult = spawnSync(
    commandName("npm"),
    ["run", "promptfoo:tests"],
    {
      stdio: "inherit",
      env: process.env,
    },
  );
  if (buildResult.status !== 0) {
    throw new Error("Failed to build full test set");
  }

  // Write a config restricted to finalists
  const fullConfig: PromptfooConfig = {
    ...baseConfig,
    description: "Tournament Finals — full eval",
    providers: survivorProviders,
  };
  const configPath = path.join(fullDir, "promptfooconfig.yaml");
  await writeFile(
    configPath,
    YAML.stringify(fullConfig, { lineWidth: 0 }),
    "utf8",
  );

  const result = spawnSync(
    commandName("npx"),
    [
      "tsx",
      "src/cli/eval-tagged.ts",
      "--tag",
      "tournament_finals",
      "--config",
      configPath,
      "--skip-build-tests",
    ],
    { stdio: "inherit", env: process.env },
  );

  if (result.status !== 0) {
    console.error(`Full eval exited with code ${result.status}`);
  } else {
    console.log(
      "\nFull eval complete! Check results/runs/ for detailed output.",
    );
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs();
  const round = (getStringArg(args, "round") ?? "smoke") as RoundName;
  const configPath = getStringArg(args, "config") ?? "promptfooconfig.yaml";
  const thresholdsPath =
    getStringArg(args, "thresholds") ?? "configs/tournament_thresholds.json";
  const seed = getNumberArg(args, "seed") ?? 42;
  const survivorsInput = getStringArg(args, "survivors");

  const tournamentDir = "results/tournament";
  await ensureDir(tournamentDir);

  const thresholds = await loadThresholds(thresholdsPath);
  const baseConfig = await loadBaseConfig(configPath);

  switch (round) {
    case "smoke": {
      await runRound(
        "smoke",
        baseConfig,
        thresholds.smoke,
        baseConfig.providers,
        tournamentDir,
        seed,
      );
      break;
    }

    case "qualifying": {
      const survivorsPath =
        survivorsInput ??
        path.join(tournamentDir, "smoke", "smoke_results.json");
      let providerIds: string[];
      try {
        providerIds = await loadSurvivors(survivorsPath);
      } catch {
        console.error(
          `Cannot load survivors from ${survivorsPath}. Run smoke round first.`,
        );
        process.exit(1);
      }
      const providers = baseConfig.providers.filter((p) =>
        providerIds.includes(getProviderId(p)),
      );
      if (providers.length === 0) {
        console.error(
          "No matching providers found for survivors. Check model IDs.",
        );
        process.exit(1);
      }
      console.log(`Loaded ${providers.length} survivors from ${survivorsPath}`);
      await runRound(
        "qualifying",
        baseConfig,
        thresholds.qualifying,
        providers,
        tournamentDir,
        seed,
      );
      break;
    }

    case "full": {
      const survivorsPath =
        survivorsInput ??
        path.join(tournamentDir, "qualifying", "qualifying_results.json");
      let survivorIds: string[];
      try {
        survivorIds = await loadSurvivors(survivorsPath);
      } catch {
        console.error(
          `Cannot load survivors from ${survivorsPath}. Run qualifying round first.`,
        );
        process.exit(1);
      }
      console.log(
        `Loaded ${survivorIds.length} finalists from ${survivorsPath}`,
      );
      await runFullEval(baseConfig, survivorIds, tournamentDir);
      break;
    }

    case "auto": {
      console.log("TOURNAMENT AUTO MODE — running all 3 rounds back-to-back\n");

      const smokeResults = await runRound(
        "smoke",
        baseConfig,
        thresholds.smoke,
        baseConfig.providers,
        tournamentDir,
        seed,
      );
      if (smokeResults.survivors.length === 0) {
        console.error(
          "\nNo models survived smoke test. Adjust thresholds or fix models.",
        );
        process.exit(1);
      }

      const qualifyingProviders = baseConfig.providers.filter((p) =>
        smokeResults.survivors.includes(getProviderId(p)),
      );
      const qualifyingResults = await runRound(
        "qualifying",
        baseConfig,
        thresholds.qualifying,
        qualifyingProviders,
        tournamentDir,
        seed,
      );
      if (qualifyingResults.survivors.length === 0) {
        console.error(
          "\nNo models survived qualifying. Adjust thresholds or check results.",
        );
        process.exit(1);
      }

      await runFullEval(baseConfig, qualifyingResults.survivors, tournamentDir);
      console.log("\nTOURNAMENT COMPLETE!");
      break;
    }

    default: {
      const _exhaustive: never = round;
      console.error(
        `Unknown round: ${String(_exhaustive)}. Use: smoke, qualifying, full, or auto`,
      );
      process.exit(1);
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import YAML from "yaml";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { ensureDir, sanitizeId, timestampForId } from "../lib/paths.js";
import { runEval } from "./eval-runner.js";

// ---------------------------------------------------------------------------
// Types (mirrors eval-runner for reading detail files)
// ---------------------------------------------------------------------------

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

type ModelDetail = {
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
  confusion_matrix: Record<string, Record<string, number>>;
  job_results: JobResult[];
};

// ---------------------------------------------------------------------------
// Markdown report generation
// ---------------------------------------------------------------------------

function labelIcon(correct: boolean): string {
  return correct ? "[OK]" : "[MISS]";
}

function formatConfusionMatrix(
  matrix: Record<string, Record<string, number>>,
): string {
  const labels = ["good_fit", "maybe", "bad_fit"];
  const lines: string[] = [];

  lines.push(
    `| Actual \\ Predicted | good_fit | maybe | bad_fit | parse_fail |`,
  );
  lines.push(`|---|---|---|---|---|`);

  for (const actual of labels) {
    const row = matrix[actual];
    if (!row) continue;
    const cells = labels.map((pred) => String(row[pred] ?? 0));
    const pf = String(row["parse_fail"] ?? 0);
    lines.push(`| **${actual}** | ${cells.join(" | ")} | ${pf} |`);
  }

  return lines.join("\n");
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + "...";
}

function generateReport(
  detail: ModelDetail,
  tag: string,
  timestamp: string,
  jobCount: number,
  prompt: string,
  configPath: string,
): string {
  const displayName = detail.model_id.replace(/^node-llama-cpp:/, "");
  const biasStr = `${detail.score_bias >= 0 ? "+" : ""}${detail.score_bias}`;
  const status = detail.passed ? "PASS" : detail.early_exit ? "FAIL*" : "FAIL";

  const lines: string[] = [];

  // Header
  lines.push(`# Prompt Lab Report`);
  lines.push(``);
  lines.push(`| | |`);
  lines.push(`|---|---|`);
  lines.push(`| **Model** | ${displayName} |`);
  lines.push(`| **Tag** | ${tag} |`);
  lines.push(`| **Config** | ${configPath} |`);
  lines.push(`| **Timestamp** | ${timestamp} |`);
  lines.push(`| **Jobs** | ${jobCount} |`);
  lines.push(`| **Status** | ${status} |`);
  lines.push(``);

  // Summary metrics
  lines.push(`## Summary`);
  lines.push(``);
  lines.push(`| Metric | Value |`);
  lines.push(`|---|---|`);
  lines.push(`| Label Accuracy | ${detail.label_accuracy_pct}% |`);
  lines.push(`| Parse Fail | ${detail.parse_fail_pct}% |`);
  lines.push(`| Score MAE | ${detail.score_mae} |`);
  lines.push(`| Score Bias | ${biasStr} |`);
  lines.push(`| Avg Latency | ${detail.avg_seconds}s |`);
  lines.push(`| Tests Run | ${detail.total_tests} |`);
  lines.push(``);

  if (detail.fail_reasons.length > 0) {
    lines.push(`**Fail reasons:** ${detail.fail_reasons.join(", ")}`);
    lines.push(``);
  }

  // Confusion matrix
  lines.push(`## Confusion Matrix`);
  lines.push(``);
  lines.push(formatConfusionMatrix(detail.confusion_matrix));
  lines.push(``);

  // Per-job results
  lines.push(`## All Jobs`);
  lines.push(``);
  lines.push(
    `| # | Result | Expected | Predicted | Score (exp/pred/err) | Latency | Job |`,
  );
  lines.push(`|---|---|---|---|---|---|---|`);

  for (let i = 0; i < detail.job_results.length; i++) {
    const jr = detail.job_results[i]!;
    const icon = jr.parse_failed ? "[PARSE]" : labelIcon(jr.label_correct);
    const predicted = jr.predicted_label ?? "—";
    const predScore = jr.predicted_score !== null ? String(jr.predicted_score) : "—";
    const scoreErr =
      jr.score_error !== null
        ? `${jr.score_error >= 0 ? "+" : ""}${jr.score_error}`
        : "—";
    const latency = `${(jr.latency_ms / 1000).toFixed(1)}s`;
    const desc = truncate(jr.job_description, 50);

    lines.push(
      `| ${i + 1} | ${icon} | ${jr.expected_label} (${jr.expected_score}) | ${predicted} (${predScore}) | ${jr.expected_score}/${predScore}/${scoreErr} | ${latency} | ${desc} |`,
    );
  }
  lines.push(``);

  // Misses detail
  const misses = detail.job_results.filter(
    (jr) => !jr.label_correct || jr.parse_failed,
  );
  if (misses.length > 0) {
    lines.push(`## Misses (${misses.length}/${detail.job_results.length})`);
    lines.push(``);

    for (let i = 0; i < misses.length; i++) {
      const jr = misses[i]!;
      lines.push(`### Miss ${i + 1}: ${jr.job_description}`);
      lines.push(``);
      lines.push(`- **Expected:** ${jr.expected_label} (score ${jr.expected_score})`);
      lines.push(
        `- **Predicted:** ${jr.predicted_label ?? "PARSE FAIL"} (score ${jr.predicted_score ?? "—"})`,
      );
      if (jr.score_error !== null) {
        lines.push(`- **Score error:** ${jr.score_error >= 0 ? "+" : ""}${jr.score_error}`);
      }
      lines.push(`- **Latency:** ${(jr.latency_ms / 1000).toFixed(1)}s`);
      lines.push(``);
      if (jr.predicted_reasoning) {
        lines.push(`> **Model reasoning:** ${jr.predicted_reasoning}`);
        lines.push(``);
      }
      if (jr.parse_failed) {
        lines.push(`> **Raw output:** \`${truncate(jr.raw_output, 200)}\``);
        lines.push(``);
      }
    }
  } else {
    lines.push(`## Misses`);
    lines.push(``);
    lines.push(`None — all ${detail.job_results.length} jobs correct.`);
    lines.push(``);
  }

  // Prompt used
  lines.push(`## Prompt Used`);
  lines.push(``);
  lines.push("```");
  lines.push(prompt);
  lines.push("```");
  lines.push(``);

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs();
  const modelFilter = getStringArg(args, "model");
  const tag = getStringArg(args, "tag") ?? "iteration";
  const jobCount = getNumberArg(args, "jobs") ?? 10;
  const rawSeed = getNumberArg(args, "seed") ?? 42;
  const seed = rawSeed % 2 === 0 ? rawSeed : rawSeed + 1;
  const configPath = getStringArg(args, "config") ?? "promptfooconfig.yaml";

  if (!modelFilter) {
    console.error(
      "Usage: npm run prompt-lab -- --model <name> [--tag <tag>] [--jobs <n>] [--seed <n>] [--config <path>]",
    );
    console.error("\n--model is required (substring match, e.g. 'qwen3-8b', 'gemma')");
    console.error("\nA/B testing example:");
    console.error("  npm run prompt-lab -- --model gemma-3-4b-it --tag v1-baseline");
    console.error("  npm run prompt-lab -- --model gemma-3-4b-it --tag v2-strict --config promptfooconfig_v2.yaml");
    console.error("  npm run compare");
    process.exit(1);
  }

  // Include config name in the run folder for easy identification
  const configName = path.basename(configPath, path.extname(configPath));
  const configSuffix = configName !== "promptfooconfig" ? `_${sanitizeId(configName)}` : "";
  const timestamp = timestampForId();
  const runId = `${timestamp}_${sanitizeId(tag) || "iteration"}${configSuffix}`;
  const runDir = path.join("results", "prompt-lab", runId);
  await ensureDir(runDir);

  // Run eval
  const results = await runEval({
    configPath,
    modelFilter,
    jobCount,
    seed,
    tag,
    outputDir: runDir,
  });

  if (results.models.length === 0) {
    console.error("No models were evaluated. Check --model filter.");
    process.exit(1);
  }

  // Read the per-model detail file (has full job_results)
  const detailsDir = path.join(runDir, "details");
  const model = results.models[0]!;
  const safeName = model.model_id
    .replace(/^node-llama-cpp:/, "")
    .replace(/[/:]/g, "_");
  const detailPath = path.join(detailsDir, `${safeName}.json`);
  const detail: ModelDetail = JSON.parse(
    await readFile(detailPath, "utf8"),
  ) as ModelDetail;

  // Extract prompt template from config
  const rawConfig = await readFile(configPath, "utf8");
  const config = YAML.parse(rawConfig) as { prompts: unknown[] };
  let promptText = "";
  const first = config.prompts[0];
  if (typeof first === "string") {
    promptText = first;
  } else if (typeof first === "object" && first !== null && "raw" in first) {
    promptText = (first as { raw: string }).raw;
  }

  // Save prompt snapshot
  await writeFile(path.join(runDir, "prompt_snapshot.txt"), promptText, "utf8");

  // Generate and save markdown report
  const report = generateReport(
    detail,
    tag,
    new Date().toISOString(),
    jobCount,
    promptText,
    configPath,
  );
  const reportPath = path.join(runDir, "report.md");
  await writeFile(reportPath, report, "utf8");

  // Print summary to stdout
  const displayName = model.model_id.replace(/^node-llama-cpp:/, "");
  const misses = detail.job_results.filter(
    (jr) => !jr.label_correct || jr.parse_failed,
  ).length;
  console.log(`\n${"=".repeat(60)}`);
  console.log(`PROMPT LAB — ${displayName} [${tag}]`);
  console.log(`${"=".repeat(60)}`);
  console.log(`  Config:     ${configPath}`);
  console.log(`  Accuracy:   ${detail.label_accuracy_pct}%`);
  console.log(`  Misses:     ${misses}/${detail.total_tests}`);
  console.log(`  MAE:        ${detail.score_mae}`);
  console.log(`  Bias:       ${detail.score_bias >= 0 ? "+" : ""}${detail.score_bias}`);
  console.log(`  Avg speed:  ${detail.avg_seconds}s/job`);
  console.log(`\n  Report:     ${reportPath}`);
  console.log(`  Run dir:    ${runDir}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

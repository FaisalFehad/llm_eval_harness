import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";

// ---------------------------------------------------------------------------
// Types (mirror eval-runner output)
// ---------------------------------------------------------------------------

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
};

type EvalResults = {
  tag: string;
  timestamp: string;
  thresholds: { job_count: number };
  models: ModelResult[];
};

type Run = {
  label: string;
  timestamp: Date;
  jobCount: number;
  models: Map<string, ModelResult>;
};

// ---------------------------------------------------------------------------
// ANSI colors
// ---------------------------------------------------------------------------

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  green: "\x1b[32m",
  red: "\x1b[31m",
  yellow: "\x1b[33m",
  cyan: "\x1b[36m",
  white: "\x1b[37m",
  bgGreen: "\x1b[42m",
  bgRed: "\x1b[41m",
  bgYellow: "\x1b[43m",
};

// ---------------------------------------------------------------------------
// Discovery — find all eval_results.json files
// ---------------------------------------------------------------------------

async function findResultFiles(baseDir: string): Promise<string[]> {
  const results: string[] = [];
  const searchDirs = [
    path.join(baseDir, "results", "eval"),
    path.join(baseDir, "results", "runs"),
  ];

  for (const dir of searchDirs) {
    let entries: string[];
    try {
      entries = await readdir(dir);
    } catch {
      continue;
    }

    for (const entry of entries) {
      const candidate = path.join(dir, entry, "eval_results.json");
      try {
        await readFile(candidate, "utf8");
        results.push(candidate);
      } catch {
        // not a result dir
      }
    }
  }

  return results;
}

async function loadRuns(baseDir: string): Promise<Run[]> {
  const files = await findResultFiles(baseDir);
  const runs: Run[] = [];

  for (const filePath of files) {
    try {
      const raw = await readFile(filePath, "utf8");
      const data = JSON.parse(raw) as EvalResults;
      const models = new Map<string, ModelResult>();
      for (const m of data.models) {
        models.set(m.model_id, m);
      }

      const dirName = path.basename(path.dirname(filePath));
      const label = data.tag || dirName;

      runs.push({
        label,
        timestamp: new Date(data.timestamp),
        jobCount: data.thresholds.job_count,
        models,
      });
    } catch {
      // skip malformed files
    }
  }

  runs.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  return runs;
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

function shortDate(d: Date): string {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[d.getMonth()]} ${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function displayName(modelId: string): string {
  return modelId.replace(/^node-llama-cpp:/, "");
}

function pad(s: string, len: number): string {
  return s.length >= len ? s.slice(0, len) : s + " ".repeat(len - s.length);
}

function rpad(s: string, len: number): string {
  return s.length >= len ? s.slice(0, len) : " ".repeat(len - s.length) + s;
}

/** Bar chart using Unicode blocks. Width = maxWidth chars. */
function bar(value: number, maxValue: number, maxWidth: number): string {
  const blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];
  const ratio = Math.max(0, Math.min(1, value / maxValue));
  const fullBlocks = Math.floor(ratio * maxWidth);
  const remainder = (ratio * maxWidth) - fullBlocks;
  const partialIdx = Math.round(remainder * 8);

  let result = "█".repeat(fullBlocks);
  if (partialIdx > 0 && fullBlocks < maxWidth) {
    result += blocks[partialIdx];
  }
  const usedLen = fullBlocks + (partialIdx > 0 ? 1 : 0);
  result += "░".repeat(Math.max(0, maxWidth - usedLen));
  return result;
}

/**
 * Format a delta with color.
 * higherIsBetter: true for accuracy, false for MAE/parse_fail/speed
 */
function delta(current: number, previous: number, higherIsBetter: boolean, unit = ""): string {
  const diff = current - previous;
  if (Math.abs(diff) < 0.05) return `${c.dim} ─${c.reset}`;

  const sign = diff > 0 ? "+" : "";
  const improved = higherIsBetter ? diff > 0 : diff < 0;
  const color = improved ? c.green : c.red;
  const arrow = improved ? "▲" : "▼";

  return `${color}${arrow}${sign}${diff.toFixed(1)}${unit}${c.reset}`;
}

/** Color accuracy value based on thresholds */
function colorAcc(v: number): string {
  if (v >= 70) return `${c.green}${v.toFixed(1)}%${c.reset}`;
  if (v >= 40) return `${c.yellow}${v.toFixed(1)}%${c.reset}`;
  return `${c.red}${v.toFixed(1)}%${c.reset}`;
}

/** Color for passed/failed */
function passLabel(passed: boolean): string {
  return passed ? `${c.green}PASS${c.reset}` : `${c.red}FAIL${c.reset}`;
}

// ---------------------------------------------------------------------------
// Render comparison
// ---------------------------------------------------------------------------

function renderComparison(runs: Run[], modelFilter: string | undefined): void {
  if (runs.length === 0) {
    console.log(`${c.yellow}No eval results found.${c.reset}`);
    console.log("Run an eval first: npm run eval -- --jobs 10");
    return;
  }

  // Collect all model IDs across runs
  const allModelIds = new Set<string>();
  for (const run of runs) {
    for (const id of run.models.keys()) {
      allModelIds.add(id);
    }
  }

  // Filter models if requested
  const needle = modelFilter?.toLowerCase();
  const modelIds = [...allModelIds].filter((id) => {
    if (!needle) return true;
    return id.toLowerCase().includes(needle) || displayName(id).toLowerCase().includes(needle);
  });

  if (modelIds.length === 0) {
    console.log(`${c.yellow}No models matched filter "${modelFilter}".${c.reset}`);
    return;
  }

  console.log();
  console.log(`${c.bold}${c.cyan}  Eval Run Comparison${c.reset}`);
  console.log(`${c.dim}  ${runs.length} run(s) found, ${modelIds.length} model(s)${c.reset}`);
  console.log();

  for (const modelId of modelIds) {
    const name = displayName(modelId);
    console.log(`${c.bold}  ${name}${c.reset}`);
    console.log(`  ${"─".repeat(78)}`);

    // Header
    console.log(
      `  ${pad("Run", 22)} ${rpad("Acc", 7)} ${rpad("MAE", 7)} ${rpad("Parse", 7)} ${rpad("Speed", 7)} ${rpad("Bias", 7)}  ${"Accuracy"}`,
    );
    console.log(`  ${"─".repeat(78)}`);

    let prevResult: ModelResult | undefined;

    for (const run of runs) {
      const result = run.models.get(modelId);
      if (!result) continue;

      const runLabel = `${shortDate(run.timestamp)} ${run.label}`;
      const truncLabel = pad(runLabel.slice(0, 20), 20);
      const jobs = `${c.dim}(${result.total_tests})${c.reset}`;
      const accBar = bar(result.label_accuracy_pct, 100, 15);
      const accColor = colorAcc(result.label_accuracy_pct);

      console.log(
        `  ${truncLabel} ${jobs} ${rpad(accColor, 18)} ${rpad(result.score_mae.toFixed(1), 7)} ${rpad(result.parse_fail_pct.toFixed(0) + "%", 7)} ${rpad(result.avg_seconds.toFixed(1) + "s", 7)} ${rpad((result.score_bias >= 0 ? "+" : "") + result.score_bias.toFixed(1), 7)}  ${accBar}`,
      );

      // Delta row
      if (prevResult) {
        const dAcc = delta(result.label_accuracy_pct, prevResult.label_accuracy_pct, true);
        const dMae = delta(result.score_mae, prevResult.score_mae, false);
        const dParse = delta(result.parse_fail_pct, prevResult.parse_fail_pct, false);
        const dSpeed = delta(result.avg_seconds, prevResult.avg_seconds, false, "s");
        const dBias = delta(Math.abs(result.score_bias), Math.abs(prevResult.score_bias), false);

        console.log(
          `  ${pad("", 22)}       ${rpad(dAcc, 18)} ${rpad(dMae, 18)} ${rpad(dParse, 18)} ${rpad(dSpeed, 18)} ${rpad(dBias, 18)}`,
        );
      }

      prevResult = result;
    }

    console.log();
  }

  // Summary: best model per latest run
  const latestRun = runs.length > 0 ? runs[runs.length - 1] : undefined;
  if (latestRun && latestRun.models.size > 1) {
    console.log(`${c.bold}  Latest run summary (${shortDate(latestRun.timestamp)} "${latestRun.label}")${c.reset}`);
    console.log(`  ${"─".repeat(78)}`);

    const sorted = [...latestRun.models.values()].sort(
      (a, b) => b.label_accuracy_pct - a.label_accuracy_pct,
    );

    for (const m of sorted) {
      const name = pad(displayName(m.model_id), 35);
      const pass = passLabel(m.passed);
      const accBar = bar(m.label_accuracy_pct, 100, 20);
      console.log(
        `  ${name} ${pass}  ${colorAcc(m.label_accuracy_pct)} ${accBar}  mae=${m.score_mae.toFixed(1)} ${m.avg_seconds.toFixed(1)}s`,
      );
    }
    console.log();
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const args = parseArgs();
  const last = getNumberArg(args, "last") ?? 20;
  const modelFilter = getStringArg(args, "model");

  const runs = await loadRuns(process.cwd());
  const recentRuns = runs.slice(-last);

  renderComparison(recentRuns, modelFilter);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

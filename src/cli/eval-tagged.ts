import { spawnSync } from "node:child_process";
import { access, appendFile, copyFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { getBooleanArg, getStringArg, parseArgs } from "../lib/args.js";
import { ensureDir, ensureParentDir, sanitizeId, timestampForId } from "../lib/paths.js";

function commandName(base: "npm" | "npx"): string {
  return process.platform === "win32" ? `${base}.cmd` : base;
}

function runCommand(command: string, args: string[]): void {
  const run = spawnSync(command, args, {
    stdio: "inherit",
    env: process.env,
  });

  if (typeof run.status === "number" && run.status === 0) {
    return;
  }

  const failureCode = run.status ?? 1;
  process.exit(failureCode);
}

function toCsvRow(columns: string[]): string {
  return columns
    .map((value) => {
      const escaped = value.replaceAll('"', '""');
      return `"${escaped}"`;
    })
    .join(",");
}

async function ensureIterationLog(logPath: string): Promise<void> {
  try {
    await access(logPath);
  } catch {
    await ensureParentDir(logPath);
    await writeFile(logPath, "timestamp,tag,run_dir\n", "utf8");
  }
}

async function main(): Promise<void> {
  const args = parseArgs();
  const tag = getStringArg(args, "tag") ?? "iteration";
  const configPath = getStringArg(args, "config") ?? "promptfooconfig.yaml";
  const skipBuildTests = getBooleanArg(args, "skip-build-tests");
  const timestamp = timestampForId();
  const runId = `${timestamp}_${sanitizeId(tag) || "iteration"}`;
  const runDir = path.join("results", "runs", runId);
  const jsonOut = path.join(runDir, "eval.json");
  const htmlOut = path.join(runDir, "eval.html");
  const metadataOut = path.join(runDir, "run_metadata.json");
  const logPath = path.join("results", "iteration_log.csv");

  if (!skipBuildTests) {
    runCommand(commandName("npm"), ["run", "promptfoo:tests"]);
  }

  await ensureDir(runDir);

  runCommand(commandName("npx"), [
    "promptfoo",
    "eval",
    "-c",
    configPath,
    "-o",
    jsonOut,
    "-o",
    htmlOut,
    "--description",
    tag,
  ]);

  // Snapshot config and prompts so every run is reproducible.
  await copyFile(configPath, path.join(runDir, "promptfooconfig.yaml"));
  await copyFile("prompts/scorer_v1.txt", path.join(runDir, "scorer_v1.txt"));
  await copyFile("prompts/scorer_v2.txt", path.join(runDir, "scorer_v2.txt"));
  await copyFile("prompts/scorer_v3_cot.txt", path.join(runDir, "scorer_v3_cot.txt"));

  const metadata = {
    tag,
    run_id: runId,
    created_at: new Date().toISOString(),
    config: configPath,
    outputs: {
      json: jsonOut,
      html: htmlOut,
    },
  };

  await writeFile(metadataOut, JSON.stringify(metadata, null, 2), "utf8");

  await ensureIterationLog(logPath);
  await appendFile(logPath, `${toCsvRow([timestamp, tag, runDir])}\n`, "utf8");

  console.log(`Saved tagged run to ${runDir}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

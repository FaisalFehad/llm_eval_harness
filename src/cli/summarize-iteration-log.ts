import { readFile } from "node:fs/promises";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";

type IterationRow = {
  timestamp: string;
  tag: string;
  run_dir: string;
};

function parseCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      i += 1;
      continue;
    }

    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }

    if (char === "," && !inQuotes) {
      result.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  result.push(current);
  return result;
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "results/iteration_log.csv";
  const limit = getNumberArg(args, "limit") ?? 20;

  const raw = await readFile(inputPath, "utf8");
  const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);

  if (lines.length <= 1) {
    console.log("No iteration runs logged yet.");
    return;
  }

  const rows: IterationRow[] = lines
    .slice(1)
    .map((line) => parseCsvLine(line))
    .filter((parts) => parts.length >= 3)
    .map((parts) => ({
      timestamp: parts[0] ?? "",
      tag: parts[1] ?? "",
      run_dir: parts[2] ?? "",
    }));

  const subset = rows.slice(Math.max(0, rows.length - limit));
  console.table(subset);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

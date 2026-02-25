import { getBooleanArg, getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";
import { GoldenJobSchema, type GoldenJob } from "../schema.js";

function printSizeWarning(count: number, minSize: number, maxSize: number): void {
  console.warn(
    `[warn] Dataset size is ${count}. Recommended range is ${minSize}-${maxSize} labeled rows.`,
  );
}

async function main(): Promise<void> {
  // --- 1. Argument Parsing ---
  const args = parseArgs();
  const inputPath = getStringArg(args, "input") ?? "data/golden_jobs.jsonl";
  const minSize = getNumberArg(args, "min-size") ?? 80;
  const maxSize = getNumberArg(args, "max-size") ?? 100;
  const strictSize = getBooleanArg(args, "strict-size");

  // --- 2. File Reading ---
  const rawRows = await readJsonlFile<unknown>(inputPath);
  if (rawRows.length === 0) {
    console.error(`No rows found in ${inputPath}.`);
    process.exit(1);
  }

  // --- 3. Schema Validation ---
  const validRows: GoldenJob[] = [];
  const parseErrors: string[] = [];
  rawRows.forEach((row, index) => {
    const parsed = GoldenJobSchema.safeParse(row);
    if (!parsed.success) {
      const issues = parsed.error.issues
        .map((issue) => `${issue.path.join(".") || "root"}: ${issue.message}`)
        .join("; ");
      parseErrors.push(`line ${index + 1}: ${issues}`);
      return;
    }
    validRows.push(parsed.data);
  });

  if (parseErrors.length > 0) {
    console.error(`Validation failed for ${parseErrors.length} row(s) in ${inputPath}:`);
    parseErrors.forEach((line) => console.error(`  - ${line}`));
    process.exit(1);
  }

  // --- 4. Duplicate ID Check ---
  const seen = new Set<string>();
  const duplicateIds = new Set<string>();
  validRows.forEach((row) => {
    if (seen.has(row.job_id)) {
      duplicateIds.add(row.job_id);
    }
    seen.add(row.job_id);
  });

  if (duplicateIds.size > 0) {
    console.error(
      `Duplicate job_id values found: ${Array.from(duplicateIds).sort().join(", ")}`,
    );
    process.exit(1);
  }

  // --- 5. Dataset Size Check ---
  const count = validRows.length;
  if (count < minSize || count > maxSize) {
    if (strictSize) {
      console.error(
        `Dataset has ${count} rows, but strict range is ${minSize}-${maxSize}.`,
      );
      process.exit(1);
    }
    printSizeWarning(count, minSize, maxSize);
  }

  // --- 6. Statistics Calculation ---
  const distribution = validRows.reduce<Record<string, number>>((acc, row) => {
    acc[row.label] = (acc[row.label] ?? 0) + 1;
    return acc;
  }, {});

  const averageScore =
    validRows.reduce((sum, row) => sum + row.score, 0) / Math.max(validRows.length, 1);

  console.log(`[ok] ${inputPath} passed validation`);
  console.log(`rows=${validRows.length} avg_score=${averageScore.toFixed(1)}`);
  console.log(
    `labels good_fit=${distribution.good_fit ?? 0} maybe=${distribution.maybe ?? 0} bad_fit=${distribution.bad_fit ?? 0}`,
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

import { readFile } from "node:fs/promises";

import { getNumberArg, getStringArg, parseArgs } from "../lib/args.js";
import { writeJsonlFile } from "../lib/jsonl.js";
import { scoreJob } from "../lib/deterministic-scorer.js";
import type { GoldenJob } from "../schema.js";

type SalaryRow = {
  Company: string;
  "Company Score": string;
  "Job Title": string;
  Location: string;
  Date: string;
  Salary: string;
};

function normalizeText(value: string | undefined): string {
  if (!value) return "";
  return value.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

function parseCsv(content: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  for (let i = 0; i < normalized.length; i += 1) {
    const ch = normalized[i]!;

    if (inQuotes) {
      if (ch === '"') {
        const next = normalized[i + 1];
        if (next === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      continue;
    }

    if (ch === ",") {
      row.push(field);
      field = "";
      continue;
    }

    if (ch === "\n") {
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") {
        rows.push(row);
      }
      row = [];
      continue;
    }

    field += ch;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function rowToRecord(headers: string[], values: string[]): SalaryRow {
  const record: Record<string, string> = {};
  headers.forEach((header, index) => {
    record[header] = values[index] ?? "";
  });

  return record as SalaryRow;
}

function buildJdText(record: SalaryRow): string {
  const company = normalizeText(record.Company);
  const location = normalizeText(record.Location) || "Not specified";
  const date = normalizeText(record.Date);
  const salary = normalizeText(record.Salary) || "Not disclosed";
  const companyScore = normalizeText(record["Company Score"]);

  const parts = [
    `Company: ${company}.`,
    `Location: ${location}.`,
  ];

  if (date) parts.push(`Posted: ${date}.`);
  parts.push(`Salary: ${salary}.`);
  if (companyScore) parts.push(`Company score: ${companyScore}.`);

  return parts.join(" ");
}

function buildReasoning(
  title: string,
  location: string | undefined,
  salaryText: string,
  breakdown: ReturnType<typeof scoreJob>,
): string {
  const parts: string[] = [];

  if (breakdown.role === 25) {
    parts.push(`${title}: strong seniority match (+25)`);
  } else if (breakdown.role === 15) {
    parts.push(`${title}: acceptable mid-level role (+15)`);
  } else {
    parts.push(`${title}: no seniority keyword (0 pts)`);
  }

  parts.push(`Stack signals score ${breakdown.tech}/25`);

  if (breakdown.loc === 25) {
    parts.push("Remote/London location (+25)");
  } else if (breakdown.loc === 10) {
    parts.push("UK outside London (+10)");
  } else if (breakdown.loc === -50) {
    parts.push("Outside UK (-50)");
  } else if (!location || location.trim().length === 0) {
    parts.push("Location unknown (0 pts)");
  } else {
    parts.push("Location unclear (0 pts)");
  }

  const hasSalary = salaryText.trim().length > 0 && salaryText !== "Not disclosed";
  const hasGBP = /£/.test(salaryText);

  if (!hasSalary) {
    parts.push("Salary unknown (0 pts)");
  } else if (!hasGBP) {
    parts.push("Salary not in GBP (0 pts)");
  } else if (breakdown.comp === 25) {
    parts.push("Salary ≥£100k (+25)");
  } else if (breakdown.comp === 15) {
    parts.push("Salary £75–99k (+15)");
  } else if (breakdown.comp === 5) {
    parts.push("Salary £55–74k (+5)");
  } else if (breakdown.comp === -30) {
    parts.push("Salary <£45k (-30)");
  } else {
    parts.push("Salary in GBP but outside scoring bands (0 pts)");
  }

  parts.push(`Total: ${breakdown.score}/100`);
  return parts.join(". ").slice(0, 600);
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ?? "data/Software Engineer Salaries.csv";
  const outputPath =
    getStringArg(args, "output") ?? "data/software_engineer_salaries_golden.jsonl";
  const limit = getNumberArg(args, "limit");

  const raw = await readFile(inputPath, "utf8");
  const rows = parseCsv(raw);

  if (rows.length === 0) {
    console.error(`No rows found in ${inputPath}`);
    process.exit(1);
  }

  const headers = rows[0] ?? [];
  if (headers.length === 0) {
    console.error(`Missing header row in ${inputPath}`);
    process.exit(1);
  }

  if (headers[0] && headers[0].charCodeAt(0) === 0xfeff) {
    headers[0] = headers[0].slice(1);
  }

  const dataRows = rows.slice(1);
  const items: GoldenJob[] = [];
  const max = limit ? Math.min(limit, dataRows.length) : dataRows.length;

  for (let i = 0; i < max; i += 1) {
    const values = dataRows[i] ?? [];
    const record = rowToRecord(headers, values);

    const title = normalizeText(record["Job Title"]);
    const company = normalizeText(record.Company);
    const location = normalizeText(record.Location);
    const salary = normalizeText(record.Salary);

    if (!title || !company) {
      continue;
    }

    const jdText = buildJdText(record);
    const breakdown = scoreJob(title, location || undefined, jdText);
    const reasoning = buildReasoning(title, location || undefined, salary, breakdown);

    const job: GoldenJob = {
      job_id: `ses-${String(i + 1).padStart(6, "0")}`,
      title,
      company,
      jd_text: jdText,
      label: breakdown.label,
      score: breakdown.score,
      reasoning,
    };

    if (location) {
      job.location = location;
    }

    items.push(job);
  }

  if (items.length === 0) {
    console.error("No usable rows found after normalization.");
    process.exit(1);
  }

  await writeJsonlFile(outputPath, items);

  console.log(`Wrote ${items.length} rows to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

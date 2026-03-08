import { getStringArg, parseArgs } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

type RawJob = {
  title?: string;
  company?: string;
  location?: string;
  description?: string;
  jd_text?: string;
};

type SlimJob = {
  title: string;
  company: string;
  location: string;
  jd_text: string;
};

function normalizeText(value: string | undefined): string {
  if (!value) return "";
  return value.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ??
    "data/custom_linkedin_data_to_train_teacher_v2.jsonl";
  const outputPath =
    getStringArg(args, "output") ??
    "data/custom_linkedin_data_to_train_teacher_v2_minimal.jsonl";

  const rows = await readJsonlFile<RawJob>(inputPath);
  const slim: SlimJob[] = [];

  let missingTitle = 0;
  let missingCompany = 0;
  let missingLocation = 0;
  let missingJdText = 0;

  for (const row of rows) {
    const title = normalizeText(row.title);
    const company = normalizeText(row.company);
    const location = normalizeText(row.location);
    const jdText = normalizeText(row.jd_text ?? row.description);

    if (!title) missingTitle++;
    if (!company) missingCompany++;
    if (!location) missingLocation++;
    if (!jdText) missingJdText++;

    slim.push({ title, company, location, jd_text: jdText });
  }

  await writeJsonlFile(outputPath, slim);

  console.log(`Loaded ${rows.length} rows from ${inputPath}`);
  console.log(
    `Missing fields: title=${missingTitle}, company=${missingCompany}, location=${missingLocation}, jd_text=${missingJdText}`,
  );
  console.log(`Wrote ${slim.length} rows to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});

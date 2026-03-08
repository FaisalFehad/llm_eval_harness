/**
 * Preprocess raw LinkedIn scraper output into canonical format for teacher labeling.
 *
 * Renames fields (description→jd_text, url→source_url), normalizes text,
 * generates job IDs, and filters out empty/short JDs.
 *
 * Usage:
 *   npx tsx src/cli/preprocess-raw-jobs.ts \
 *     --input "data/Student Training Data/real_linkedin_500.jsonl" \
 *     --output "data/Student Training Data/preprocessed_500.jsonl" \
 *     --min-length 50
 */

import { createHash } from "node:crypto";
import { parseArgs, getStringArg, getNumberArg, getBooleanArg } from "../lib/args.js";
import { readJsonlFile, writeJsonlFile } from "../lib/jsonl.js";

// ── Types ────────────────────────────────────────────────────────────────────

type RawLinkedInJob = {
  url?: string;
  id?: string | number;
  job_id?: string | number;
  title?: string;
  company?: string;
  location?: string;
  description?: string;
  jd_text?: string;
  jd_test?: string;
  salary?: string;
  source_url?: string;
};

type PreprocessedJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  source_url?: string;
  salary_field?: string;
};

// ── Helpers (duplicated from score-raw-jobs-to-golden-format.ts to avoid
//    touching working code) ───────────────────────────────────────────────────

function normalizeText(value: string | undefined): string {
  if (!value) return "";
  return value.replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
}

function extractJobId(
  rawId: string | number | undefined,
  sourceUrl: string,
  title: string,
  company: string,
  location: string,
): string {
  if (rawId !== undefined && String(rawId).trim().length > 0) {
    return String(rawId).trim();
  }

  if (sourceUrl) {
    const matches = sourceUrl.match(/\b\d{6,}\b/g);
    if (matches && matches.length > 0) {
      return matches[matches.length - 1]!;
    }
  }

  const basis = `${title}|${company}|${location}`;
  const hash = createHash("sha1").update(basis).digest("hex").slice(0, 12);
  return `ukjob_${hash}`;
}

// ── Location extraction ──────────────────────────────────────────────────────

const NON_UK_SUBDOMAINS: Record<string, string> = {
  au: "Australia", ca: "Canada", in: "India", nl: "Netherlands",
  de: "Germany", ie: "Ireland", sg: "Singapore", es: "Spain",
  fr: "France", nz: "New Zealand", za: "South Africa", br: "Brazil",
  se: "Sweden", ch: "Switzerland", at: "Austria", be: "Belgium",
  dk: "Denmark", fi: "Finland", no: "Norway", pt: "Portugal",
  it: "Italy", jp: "Japan", mx: "Mexico", hk: "Hong Kong",
  ph: "Philippines", pk: "Pakistan", kr: "South Korea",
};

const UK_CITIES: Record<string, string> = {
  london: "London, England, United Kingdom",
  manchester: "Manchester, England, United Kingdom",
  bristol: "Bristol, England, United Kingdom",
  edinburgh: "Edinburgh, Scotland, United Kingdom",
  glasgow: "Glasgow, Scotland, United Kingdom",
  birmingham: "Birmingham, England, United Kingdom",
  leeds: "Leeds, England, United Kingdom",
  cambridge: "Cambridge, England, United Kingdom",
  oxford: "Oxford, England, United Kingdom",
  belfast: "Belfast, Northern Ireland, United Kingdom",
  cardiff: "Cardiff, Wales, United Kingdom",
  newcastle: "Newcastle, England, United Kingdom",
  brighton: "Brighton, England, United Kingdom",
  sheffield: "Sheffield, England, United Kingdom",
  nottingham: "Nottingham, England, United Kingdom",
  reading: "Reading, England, United Kingdom",
  bath: "Bath, England, United Kingdom",
  liverpool: "Liverpool, England, United Kingdom",
  coventry: "Coventry, England, United Kingdom",
  southampton: "Southampton, England, United Kingdom",
};

function extractLocationFromUrl(url: string, description: string): string {
  const subdomainMatch = url.match(/^https?:\/\/(\w+)\.linkedin/);
  const subdomain = subdomainMatch?.[1] ?? "www";

  // Non-UK subdomains → country name (will score loc=-50)
  if (subdomain in NON_UK_SUBDOMAINS) {
    return NON_UK_SUBDOMAINS[subdomain]!;
  }

  // www subdomain — check for US patterns in description
  if (subdomain === "www") {
    const descHead = description.slice(0, 600).toLowerCase();
    const usPatterns = [
      /(?:united states|usa|u\.s\.)/i,
      /(?:san francisco|new york|seattle|austin|boston|chicago|denver|los angeles|portland|atlanta)/i,
      /(?:remote|based in|location)[:\s]*[\w\s]*,\s*(?:ca|ny|tx|wa|fl|il|ma|co|ga|or|va)\b/i,
    ];
    for (const pat of usPatterns) {
      if (pat.test(descHead)) return "United States";
    }
  }

  // UK subdomain or www with potential UK content
  const descLower = description.slice(0, 800).toLowerCase();

  // Check for Remote UK patterns
  if (/(?:fully\s+)?remote.*(?:uk|united kingdom)/i.test(descLower) ||
      /(?:uk|united kingdom).*(?:fully\s+)?remote/i.test(descLower)) {
    return "United Kingdom (Remote)";
  }

  // Check for specific UK cities
  for (const [city, fullLocation] of Object.entries(UK_CITIES)) {
    // Use word boundary to avoid false positives (e.g., "bath" in "bathroom")
    const cityRegex = new RegExp(`\\b${city}\\b`, "i");
    if (cityRegex.test(descLower.slice(0, 500))) {
      // Check for hybrid/on-site modifiers
      const nearCity = descLower.slice(
        Math.max(0, descLower.indexOf(city) - 30),
        descLower.indexOf(city) + city.length + 40,
      );
      if (/hybrid/i.test(nearCity)) return fullLocation + " (Hybrid)";
      if (/on-?site/i.test(nearCity)) return fullLocation + " (On-site)";
      return fullLocation;
    }
  }

  // UK subdomain but no city found
  if (subdomain === "uk") return "United Kingdom";

  return "";
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ??
    "data/Student Training Data/real_linkedin_500.jsonl";
  const outputPath =
    getStringArg(args, "output") ??
    "data/Student Training Data/preprocessed_500.jsonl";
  const minLength = getNumberArg(args, "min-length") ?? 50;
  const extractLocation = getBooleanArg(args, "extract-location");

  const rawJobs = await readJsonlFile<RawLinkedInJob>(inputPath);
  console.log(`Read ${rawJobs.length} raw jobs from ${inputPath}`);

  const results: PreprocessedJob[] = [];
  let noTitle = 0;
  let shortJd = 0;

  for (const raw of rawJobs) {
    const title = normalizeText(raw.title);
    if (!title) {
      noTitle++;
      continue;
    }

    const jdText = normalizeText(raw.description ?? raw.jd_text ?? raw.jd_test);
    if (jdText.length < minLength) {
      shortJd++;
      continue;
    }

    const company = normalizeText(raw.company);
    const sourceUrl = normalizeText(raw.url ?? raw.source_url);
    const location = normalizeText(raw.location) ||
      (extractLocation && sourceUrl ? extractLocationFromUrl(sourceUrl, jdText) : "");
    const jobId = extractJobId(raw.job_id ?? raw.id, sourceUrl, title, company, location);
    const salary = normalizeText(raw.salary);

    const job: PreprocessedJob = {
      job_id: jobId,
      title,
      company,
      location,
      jd_text: jdText,
    };

    if (sourceUrl) job.source_url = sourceUrl;
    if (salary) job.salary_field = salary;

    results.push(job);
  }

  await writeJsonlFile(outputPath, results);

  console.log(`\nPreprocessing complete:`);
  console.log(`  Input:          ${rawJobs.length}`);
  console.log(`  No title:       ${noTitle}`);
  console.log(`  Short JD (<${minLength}): ${shortJd}`);
  console.log(`  Output:         ${results.length}`);
  console.log(`  Written to:     ${outputPath}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

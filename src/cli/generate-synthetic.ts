/**
 * Generate synthetic job postings for V5 distribution gaps.
 *
 * Reads the distribution report, identifies gaps, and uses OpenAI
 * to generate realistic job postings targeting specific gap categories.
 * Post-processes JDs with noise (random capitalization, formatting variation).
 *
 * After generation, labels them with the teacher prompt and validates.
 *
 * Usage:
 *   npx tsx src/cli/generate-synthetic.ts \
 *     --distribution data/v5/all_labeled_pool.distribution.json \
 *     --output data/v5/synthetic_jobs.jsonl \
 *     --model gpt-4o-mini \
 *     --max-per-category 60 \
 *     --max-total-pct 25
 *
 * Requires OPENAI_API_KEY environment variable.
 */

import * as fs from "node:fs";
import OpenAI from "openai";
import { parseArgs, getStringArg, getNumberArg } from "../lib/args.js";

// ── Target minimums (same as in check-distribution.ts) ────────────────

const LOC_MINIMUMS: Record<string, number> = {
  OUTSIDE_UK: 100, UK_OTHER: 80, LONDON_OR_REMOTE: 250, MISSING: 15,
};
const TECH_MINIMUMS: Record<string, number> = {
  NONE: 150, JS_TS: 100, NODE: 80, NODE_JS_TS: 120,
  AI_ML: 20, JS_TS_AI_ML: 20, NODE_AI_ML: 20, NODE_JS_TS_AI_ML: 30,
};
const COMP_MINIMUMS: Record<string, number> = {
  NO_GBP: 205, UP_TO_ONLY: 40, BELOW_45K: 20,
  RANGE_55_74K: 50, RANGE_75_99K: 80, ABOVE_100K: 100,
};
const ROLE_MINIMUMS: Record<string, number> = {
  SENIOR_PLUS: 290, MID_LEVEL: 120, NO_SENIORITY: 120,
};
const LABEL_MINIMUMS: Record<string, number> = {
  good_fit: 150, maybe: 200, bad_fit: 300,
};

// ── Gap spec → generation prompt ─────────────────────────────────────────

type GapSpec = {
  field: string;
  token: string;
  needed: number;
  generationHint: string;
};

function buildGapSpecs(dist: Record<string, Record<string, number>>): GapSpec[] {
  const gaps: GapSpec[] = [];

  // Location gaps
  for (const [token, min] of Object.entries(LOC_MINIMUMS)) {
    const actual = dist.loc?.[token] ?? 0;
    const needed = Math.max(0, min - actual);
    if (needed === 0) continue;
    let hint = "";
    switch (token) {
      case "OUTSIDE_UK":
        hint = "Location MUST be outside the UK (e.g., San Francisco, Berlin, Dublin Ireland, Singapore, Sydney, Tokyo, Toronto). Vary the countries. Include realistic JDs for software engineering roles.";
        break;
      case "UK_OTHER":
        hint = "Location MUST be in the UK but NOT London and NOT Remote (e.g., Manchester, Bristol, Edinburgh, Glasgow, Cardiff, Birmingham, Leeds, Belfast, Nottingham, Cambridge, Oxford, Liverpool, Brighton, Reading). Vary the UK cities.";
        break;
      case "LONDON_OR_REMOTE":
        hint = "Location MUST contain 'London' or 'Remote'. Vary formats: 'London, England', 'London, England, United Kingdom', 'Remote', 'UK Remote', 'United Kingdom (Remote)', 'London (Remote)', 'Fully Remote', 'Remote - UK'.";
        break;
      case "MISSING":
        hint = "Location MUST be empty or unclear (e.g., empty string, 'TBD', 'Various', 'Multiple Locations').";
        break;
    }
    gaps.push({ field: "loc", token, needed, generationHint: hint });
  }

  // Tech gaps
  for (const [token, min] of Object.entries(TECH_MINIMUMS)) {
    const actual = dist.tech?.[token] ?? 0;
    const needed = Math.max(0, min - actual);
    if (needed === 0) continue;
    let hint = "";
    switch (token) {
      case "NONE":
        hint = "JD must NOT mention Node.js, JavaScript, TypeScript, or required AI/ML. Use other tech: React (alone), Python, Go, Java, Ruby, .NET, PHP, Rust, Kotlin, C#, Scala. Include prominent non-qualifying tech in the JD.";
        break;
      case "JS_TS":
        hint = "JD must mention JavaScript or TypeScript (or both) but NOT Node.js/NodeJS. Include other tech as 'nice to have'. Vary capitalization: 'JavaScript', 'javascript', 'JS', 'TypeScript', 'typescript', 'TS'.";
        break;
      case "NODE":
        hint = "JD must mention Node.js/NodeJS/node but NOT JavaScript/TypeScript separately. Vary: 'Node.js', 'node.js', 'NodeJS', 'nodejs', 'Nodejs', 'Node'.";
        break;
      case "NODE_JS_TS":
        hint = "JD must mention BOTH Node.js AND JavaScript/TypeScript. This is a full-stack Node role.";
        break;
      case "AI_ML":
        hint = "JD must REQUIRE AI/ML/LLM experience (not 'nice to have', not just in company description). Must NOT mention Node.js or JavaScript/TypeScript.";
        break;
      case "JS_TS_AI_ML":
        hint = "JD must mention JavaScript/TypeScript AND explicitly require AI/ML experience. No Node.js.";
        break;
      case "NODE_AI_ML":
        hint = "JD must mention Node.js AND explicitly require AI/ML experience. No separate JavaScript/TypeScript mention.";
        break;
      case "NODE_JS_TS_AI_ML":
        hint = "JD must mention Node.js, JavaScript/TypeScript, AND explicitly require AI/ML/LLM experience. Full stack AI role.";
        break;
    }
    gaps.push({ field: "tech", token, needed, generationHint: hint });
  }

  // Comp gaps
  for (const [token, min] of Object.entries(COMP_MINIMUMS)) {
    const actual = dist.comp?.[token] ?? 0;
    const needed = Math.max(0, min - actual);
    if (needed === 0) continue;
    let hint = "";
    switch (token) {
      case "NO_GBP":
        hint = "JD must have NO GBP annual salary. Mix these variants: (a) USD salary like '$120,000-$180,000', (b) EUR salary like '€80,000-€100,000', (c) daily rate like '£500/day' or '£600 per day', (d) no salary at all, (e) midpoint £45k-£54,999 like '£40,000-£55,000'.";
        break;
      case "UP_TO_ONLY":
        hint = "JD must have 'Up to £X' or 'to £X' salary with NO lower bound. Vary: 'Up to £90k', 'up to £120,000', 'to £80k', 'To £95,000 per annum'. Must not have a range like '£X-£Y'.";
        break;
      case "BELOW_45K":
        hint = "JD must have a GBP salary with midpoint below £45,000. Examples: '£25,000-£35,000', '£30k-£40k', '£35,000 per annum'.";
        break;
      case "RANGE_55_74K":
        hint = "JD must have a GBP salary with midpoint £55,000-£74,999. Examples: '£55k-£70k', '£60,000-£75,000', '£65k p.a.', '£70,000 per annum'.";
        break;
      case "RANGE_75_99K":
        hint = "JD must have a GBP salary with midpoint £75,000-£99,999. Examples: '£70k-£90k', '£75,000-£95,000', '£80k-£100k'.";
        break;
      case "ABOVE_100K":
        hint = "JD must have a GBP salary with midpoint ≥ £100,000. Examples: '£100k-£130k', '£110,000-£140,000', '£120k+', '£105,000 per annum'.";
        break;
    }
    gaps.push({ field: "comp", token, needed, generationHint: hint });
  }

  // Role gaps
  for (const [token, min] of Object.entries(ROLE_MINIMUMS)) {
    const actual = dist.role?.[token] ?? 0;
    const needed = Math.max(0, min - actual);
    if (needed === 0) continue;
    let hint = "";
    switch (token) {
      case "SENIOR_PLUS":
        hint = "Title MUST contain one of: Senior, Staff, Principal, Lead, Tech Lead, Head, Distinguished, VP, Snr, Founding, Engineer III, SWE III. Mix standard and edge variants.";
        break;
      case "MID_LEVEL":
        hint = "Title MUST contain one of: Full Stack, Full-Stack, Fullstack, Mid-Level, Midlevel, Software Engineer II, Engineer II, SWE II. Vary the format.";
        break;
      case "NO_SENIORITY":
        hint = "Title must NOT contain any seniority keyword. Examples: 'Software Engineer', 'Developer', 'Backend Engineer', 'Platform Engineer'. AND the first hiring sentence must not mention senior/staff/etc.";
        break;
    }
    gaps.push({ field: "role", token, needed, generationHint: hint });
  }

  // Label gaps (good_fit needs score ≥70, maybe needs 50-69)
  for (const [token, min] of Object.entries(LABEL_MINIMUMS)) {
    const actual = dist.label?.[token] ?? 0;
    const needed = Math.max(0, min - actual);
    if (needed === 0) continue;
    let hint = "";
    switch (token) {
      case "good_fit":
        hint = `These MUST be high-scoring jobs (score ≥70). ALL of these must be true:
- Location: London or Remote (vary: "London, UK", "Remote", "London (Remote)", "UK Remote")
- Title: MUST include seniority: Senior, Staff, Lead, Principal, Head, Founding
- Tech: MUST require Node.js AND (JavaScript or TypeScript). Some should also require AI/ML.
- Salary: MUST include GBP annual salary with midpoint ≥ £75,000 (e.g. "£80k-£110k", "£90,000-£120,000", "£100k+")`;
        break;
      case "maybe":
        hint = `These MUST be medium-scoring jobs (score 50-69). Mix of moderate signals:
- Location: London or Remote (25 pts) but weaker tech/comp, OR UK_OTHER (10 pts) with strong tech+comp+role
- Salary: mid-range GBP (£55k-£74k) or no salary, combined with good location + seniority
- Example combos: London + Senior + JS_TS + NO_GBP = 55, London + Senior + NODE + BELOW_45K = 30 (too low), Remote + Senior + NODE_JS_TS + RANGE_55_74K = 70 (too high, that's good_fit)
- Aim for borderline combinations that total 50-69`;
        break;
      case "bad_fit":
        hint = `These MUST be low-scoring jobs (score <50). Examples:
- Outside UK location (-50 pts makes almost anything bad_fit)
- UK_OTHER with no seniority, no relevant tech, no GBP salary
- Below £45k salary (-30 pts) with weak other signals`;
        break;
    }
    gaps.push({ field: "label", token, needed, generationHint: hint });
  }

  return gaps;
}

// ── Generation prompt ────────────────────────────────────────────────────

function buildGenerationPrompt(spec: GapSpec, batchSize: number): string {
  return `Generate ${batchSize} realistic job postings as a JSON array. Each element: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

Requirements for ALL jobs:
- ${spec.generationHint}
- jd_text: 200-400 words
- Include: equal opportunity statement, 2-3 sentence company description, benefits section
- Use inconsistent formatting: mix bullet points with prose, vary capitalisation of tech names
- Include at least one misleading detail: e.g. mention a different city in company description, include a non-GBP salary alongside GBP, list a scored tech as "nice to have" while listing unscored tech as "required"
- Make each job distinct — different companies, different tech stacks, different seniority levels
- Use realistic company names (not obviously fake)

Return ONLY the JSON array, no markdown fences.`;
}

// ── Post-processing noise ────────────────────────────────────────────────

function addNoise(jd: string): string {
  // Randomly vary tech capitalization
  const techVariants: Record<string, string[]> = {
    "node.js": ["Node.js", "NodeJS", "nodejs", "Nodejs", "node.js"],
    "javascript": ["JavaScript", "javascript", "Javascript", "JS"],
    "typescript": ["TypeScript", "typescript", "Typescript", "TS"],
    "react": ["React", "react", "ReactJS", "React.js"],
    "python": ["Python", "python", "PYTHON"],
  };

  let result = jd;
  for (const [, variants] of Object.entries(techVariants)) {
    // Randomly pick a variant for each occurrence
    for (const variant of variants) {
      if (result.includes(variant)) {
        const replacement = variants[Math.floor(Math.random() * variants.length)]!;
        result = result.replace(variant, replacement);
        break; // Only replace first occurrence
      }
    }
  }

  return result;
}

// ── Concurrency helper ───────────────────────────────────────────────────

async function processWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      results[index] = await fn(items[index]!, index);
    }
  }

  const workers = Array.from(
    { length: Math.min(concurrency, items.length) },
    () => worker(),
  );
  await Promise.all(workers);
  return results;
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const distPath = getStringArg(args, "distribution") ?? "data/v5/all_labeled_pool.distribution.json";
  const outputPath = getStringArg(args, "output") ?? "data/v5/synthetic_jobs.jsonl";
  const modelId = getStringArg(args, "model") ?? "gpt-4o-mini";
  const maxPerCategory = getNumberArg(args, "max-per-category") ?? 60;
  const poolSize = getNumberArg(args, "pool-size") ?? 800; // Expected final training set size

  if (!process.env.OPENAI_API_KEY) {
    console.error("ERROR: OPENAI_API_KEY environment variable is required");
    process.exit(1);
  }

  if (!fs.existsSync(distPath)) {
    console.error(`Distribution report not found: ${distPath}`);
    console.error("Run check-distribution.ts first.");
    process.exit(1);
  }

  const dist = JSON.parse(fs.readFileSync(distPath, "utf-8"));
  const totalExisting = dist.total_jobs as number;
  const maxSyntheticTotal = Math.floor(poolSize * 0.25); // 25% cap

  console.log(`Existing pool: ${totalExisting} jobs`);
  console.log(`Max synthetic (25% of ~${poolSize}): ${maxSyntheticTotal}`);
  console.log(`Max per category: ${maxPerCategory}`);
  console.log(`Model: ${modelId}`);
  console.log("─".repeat(60));

  // Build gap specs
  const gaps = buildGapSpecs(dist);

  if (gaps.length === 0) {
    console.log("\nNo gaps found! All minimums met. ✅");
    return;
  }

  // Cap each gap at maxPerCategory and total at maxSyntheticTotal
  let totalNeeded = 0;
  for (const gap of gaps) {
    gap.needed = Math.min(gap.needed, maxPerCategory);
    totalNeeded += gap.needed;
  }

  // If total exceeds cap, proportionally reduce
  if (totalNeeded > maxSyntheticTotal) {
    const ratio = maxSyntheticTotal / totalNeeded;
    for (const gap of gaps) {
      gap.needed = Math.max(1, Math.floor(gap.needed * ratio));
    }
    totalNeeded = gaps.reduce((sum, g) => sum + g.needed, 0);
  }

  console.log(`\nGaps to fill (${gaps.length} categories, ${totalNeeded} total jobs):`);
  for (const gap of gaps) {
    console.log(`  ${gap.field}.${gap.token}: need ${gap.needed}`);
  }

  const client = new OpenAI();
  const allSynthetic: Array<Record<string, unknown>> = [];
  let syntheticId = 1;

  for (const gap of gaps) {
    if (gap.needed <= 0) continue;

    // Generate in batches of 5-10
    const batchSize = Math.min(10, gap.needed);
    const batches = Math.ceil(gap.needed / batchSize);

    console.log(`\nGenerating ${gap.needed} jobs for ${gap.field}.${gap.token} (${batches} batches of ~${batchSize})...`);

    for (let batch = 0; batch < batches; batch++) {
      const thisSize = Math.min(batchSize, gap.needed - batch * batchSize);
      const prompt = buildGenerationPrompt(gap, thisSize);

      try {
        let content = "";
        for (let attempt = 0; attempt <= 6; attempt++) {
          try {
            const response = await client.chat.completions.create({
              model: modelId,
              messages: [
                { role: "system", content: "You generate realistic job postings. Output ONLY valid JSON arrays." },
                { role: "user", content: prompt },
              ],
              max_tokens: 4000,
              temperature: 0.8, // Higher temp for diversity
            });
            content = response.choices[0]?.message?.content ?? "";
            break;
          } catch (retryErr) {
            const msg = retryErr instanceof Error ? retryErr.message : String(retryErr);
            if (msg.includes("429") && attempt < 6) {
              const waitMs = 2000 * Math.pow(2, attempt);
              console.warn(`  Rate limited, waiting ${(waitMs / 1000).toFixed(0)}s...`);
              await new Promise((r) => setTimeout(r, waitMs));
              continue;
            }
            throw retryErr;
          }
        }

        // Parse JSON array
        let jobs: Array<{ title: string; company: string; location: string; jd_text: string }>;
        try {
          const cleaned = content.replace(/```json?\n?/g, "").replace(/```/g, "").trim();
          jobs = JSON.parse(cleaned);
          if (!Array.isArray(jobs)) jobs = [jobs];
        } catch {
          console.warn(`  Batch ${batch + 1}: parse error, skipping`);
          continue;
        }

        for (const job of jobs) {
          if (!job.title || !job.jd_text) continue;

          // Apply noise
          const noisyJd = addNoise(job.jd_text);

          allSynthetic.push({
            job_id: `synth_v5_${String(syntheticId++).padStart(4, "0")}`,
            title: job.title,
            company: job.company ?? "Synthetic Co",
            location: job.location ?? "",
            jd_text: noisyJd,
            source_file: "synthetic_v5",
            synthetic_target: `${gap.field}.${gap.token}`,
          });
        }

        console.log(`  Batch ${batch + 1}/${batches}: generated ${jobs.length} jobs`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`  Batch ${batch + 1}: API error: ${msg}`);
      }
    }
  }

  // Write output
  const outStream = fs.createWriteStream(outputPath);
  for (const job of allSynthetic) {
    outStream.write(JSON.stringify(job) + "\n");
  }
  outStream.end();

  console.log("\n" + "═".repeat(60));
  console.log("SYNTHETIC GENERATION COMPLETE");
  console.log("═".repeat(60));
  console.log(`Generated: ${allSynthetic.length} synthetic jobs`);
  console.log(`Output: ${outputPath}`);
  console.log();
  console.log("Next steps:");
  console.log(`  1. Label with teacher prompt: npx tsx src/cli/label-jobs.ts --input ${outputPath} --output data/v5/synthetic_labeled.jsonl`);
  console.log("  2. Combine: cat data/v5/all_labeled_pool.jsonl data/v5/synthetic_labeled.jsonl > data/v5/full_pool.jsonl");
  console.log("  3. Re-check distributions: npx tsx src/cli/check-distribution.ts --input data/v5/full_pool.jsonl");
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

/**
 * Generate V7 synthetic jobs from multi-field recipes.
 *
 * Each recipe specifies all 6 V7 fields. Uses OpenAI to generate realistic
 * job descriptions matching the specified combination of tokens.
 *
 * Usage:
 *   npx tsx src/cli/generate-from-recipes-v7.ts \
 *     --recipes data/v7/generation_recipes.jsonl \
 *     --output data/v7/generated_jobs.jsonl \
 *     --model gpt-4o-mini \
 *     --batch-size 5 \
 *     --concurrency 3
 *
 * Requires OPENAI_API_KEY in .env or environment.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import OpenAI from "openai";

// ── Load .env ────────────────────────────────────────────────────────────

function loadEnv(): void {
  const envPath = path.resolve(process.cwd(), ".env");
  if (fs.existsSync(envPath)) {
    const envContent = fs.readFileSync(envPath, "utf-8");
    for (const line of envContent.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIdx = trimmed.indexOf("=");
      if (eqIdx === -1) continue;
      const key = trimmed.slice(0, eqIdx).trim();
      const value = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
      if (!process.env[key]) {
        process.env[key] = value;
      }
    }
  }
}

// ── Types ────────────────────────────────────────────────────────────────

interface Recipe {
  recipe_id: string;
  location: string;
  work_arrangement: string;
  scope: string;
  seniority: string;
  tech: string;
  comp: string;
  location_text: string;
  suggested_title: string;
  tech_description: string;
  comp_description: string;
  scope_description: string;
}

interface GeneratedJob {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  source_file: string;
  recipe_id: string;
  recipe_targets: Record<string, string>;
}

// ── Prompt Builder ───────────────────────────────────────────────────────

function buildPrompt(recipes: Recipe[]): string {
  const jobSpecs = recipes.map((r, i) => {
    const lines = [
      `Job ${i + 1}:`,
      `  Title: ${r.suggested_title} (or similar)`,
      `  Location field: "${r.location_text}"`,
      `  Work arrangement: ${r.work_arrangement}`,
    ];

    // Tech requirements
    if (r.tech === "NONE") {
      lines.push(
        `  Tech stack: Use Python, Go, Java, C#, Ruby, or other non-JS/TS/Node tech. Do NOT mention Node.js, JavaScript, TypeScript, React, Angular, Vue, or AI/ML as requirements.`,
      );
    } else if (r.tech === "JS_TS") {
      lines.push(
        `  Tech stack: MUST mention JavaScript and/or TypeScript (React, Angular, Vue, Next.js are OK). Do NOT mention Node.js. Do NOT require AI/ML.`,
      );
    } else if (r.tech === "NODE") {
      lines.push(
        `  Tech stack: MUST mention Node.js (vary: "Node.js", "NodeJS", "node"). Do NOT separately mention JavaScript/TypeScript/React/Angular/Vue. Do NOT require AI/ML.`,
      );
    } else if (r.tech === "NODE_JS_TS") {
      lines.push(
        `  Tech stack: MUST mention BOTH Node.js AND JavaScript/TypeScript. Full-stack Node role. Do NOT require AI/ML.`,
      );
    } else if (r.tech === "AI_ML") {
      lines.push(
        `  Tech stack: AI/ML MUST be a CORE REQUIREMENT in the Requirements section (e.g., "3+ years of hands-on experience with machine learning" or "Must have production experience deploying ML models"). List AI/ML tools as required: TensorFlow, PyTorch, LLMs, NLP, etc. Do NOT mention Node.js or JavaScript/TypeScript. Do NOT list AI/ML as "nice to have" — it must be MANDATORY.`,
      );
    } else if (r.tech === "JS_TS_AI_ML") {
      lines.push(
        `  Tech stack: MUST mention JavaScript/TypeScript AND AI/ML MUST be a CORE REQUIREMENT (not nice-to-have). List AI/ML in the required skills section, e.g., "Required: TypeScript, React, machine learning, TensorFlow". Do NOT mention Node.js.`,
      );
    } else if (r.tech === "NODE_AI_ML") {
      lines.push(
        `  Tech stack: MUST mention Node.js AND AI/ML MUST be a CORE REQUIREMENT (not nice-to-have). List both in required skills, e.g., "Required: Node.js, machine learning, PyTorch". Do NOT separately mention JavaScript/TypeScript front-end.`,
      );
    } else if (r.tech === "NODE_JS_TS_AI_ML") {
      lines.push(
        `  Tech stack: MUST mention Node.js, JavaScript/TypeScript, AND AI/ML MUST be a CORE REQUIREMENT (not nice-to-have). Full-stack AI role. List all in required skills, e.g., "Required: Node.js, TypeScript, React, TensorFlow, machine learning".`,
      );
    }

    // Comp requirements
    if (r.comp === "NO_GBP") {
      lines.push(
        `  Salary: Either no salary mentioned, OR use USD/EUR (e.g. "$120,000-$160,000"), OR daily rate (e.g. "£500/day"). Do NOT include annual GBP salary.`,
      );
    } else if (r.comp === "UP_TO_ONLY") {
      lines.push(
        `  Salary: Use "Up to £X" format with NO lower bound (e.g. "Up to £85,000", "to £95k"). Do NOT give a range like "£X-£Y".`,
      );
    } else if (r.comp === "BELOW_45K") {
      lines.push(
        `  Salary: GBP annual salary with midpoint BELOW £45,000 (e.g. "£25,000-£35,000", "£30k-£42k").`,
      );
    } else if (r.comp === "RANGE_55_74K") {
      lines.push(
        `  Salary: GBP annual salary with midpoint £55,000-£74,999 (e.g. "£55k-£70k", "£60,000-£80,000", "£65k pa").`,
      );
    } else if (r.comp === "RANGE_75_99K") {
      lines.push(
        `  Salary: GBP annual salary with midpoint £75,000-£99,999 (e.g. "£70k-£90k", "£75,000-£95,000", "£80k-£100k").`,
      );
    } else if (r.comp === "ABOVE_100K") {
      lines.push(
        `  Salary: GBP annual salary with midpoint >= £100,000 (e.g. "£100k-£130k", "£110,000-£140,000", "£120k+").`,
      );
    }

    // Work arrangement details
    if (r.work_arrangement === "HYBRID") {
      lines.push(
        `  Work: Mention hybrid arrangement (e.g. "2-3 days in office", "hybrid working").`,
      );
    } else if (r.work_arrangement === "REMOTE") {
      lines.push(
        `  Work: Mention fully remote working (e.g. "100% remote", "work from anywhere").`,
      );
    } else if (r.work_arrangement === "IN_OFFICE") {
      lines.push(
        `  Work: Mention office-based or on-site working (e.g. "office-based", "on-site 5 days").`,
      );
    }

    return lines.join("\n");
  });

  return `Generate ${recipes.length} realistic, distinct job postings as a JSON array.
Each element: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}

CRITICAL RULES:
- jd_text: 200-500 words, realistic formatting
- Include company description (2-3 sentences), responsibilities, requirements, benefits
- Use inconsistent formatting: mix bullet points with prose
- Vary tech name capitalization (Node.js vs NodeJS vs node.js)
- Include at least one misleading detail per JD (e.g., mention a tech as "nice to have")
- Make each job distinct with different companies and contexts
- Use realistic but fictional company names
- The location field MUST be EXACTLY as specified

Job specifications:
${jobSpecs.join("\n\n")}

Return ONLY the JSON array, no markdown fences.`;
}

// ── Noise ────────────────────────────────────────────────────────────────

function addNoise(jd: string): string {
  const techVariants: Record<string, string[]> = {
    "node.js": ["Node.js", "NodeJS", "nodejs", "Nodejs", "node.js"],
    javascript: ["JavaScript", "javascript", "Javascript", "JS"],
    typescript: ["TypeScript", "typescript", "Typescript", "TS"],
    react: ["React", "react", "ReactJS", "React.js"],
    python: ["Python", "python", "PYTHON"],
  };

  let result = jd;
  for (const [, variants] of Object.entries(techVariants)) {
    for (const variant of variants) {
      if (result.includes(variant)) {
        const replacement =
          variants[Math.floor(Math.random() * variants.length)]!;
        result = result.replace(variant, replacement);
        break;
      }
    }
  }
  return result;
}

// ── Concurrency ──────────────────────────────────────────────────────────

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

// ── Args ─────────────────────────────────────────────────────────────────

function parseCliArgs(): Record<string, string> {
  const args: Record<string, string> = {};
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i]!;
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const value = process.argv[i + 1] ?? "";
      args[key] = value;
      i++;
    }
  }
  return args;
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  loadEnv();

  const args = parseCliArgs();
  const recipesPath =
    args["recipes"] ?? "data/v7/generation_recipes.jsonl";
  const outputPath =
    args["output"] ?? "data/v7/generated_jobs.jsonl";
  const modelId = args["model"] ?? "gpt-4o-mini";
  const batchSize = parseInt(args["batch-size"] ?? "5", 10);
  const concurrency = parseInt(args["concurrency"] ?? "3", 10);

  if (!process.env.OPENAI_API_KEY) {
    console.error(
      "ERROR: OPENAI_API_KEY not found. Set it in .env or environment.",
    );
    process.exit(1);
  }

  // Load recipes
  const recipes: Recipe[] = fs
    .readFileSync(recipesPath, "utf-8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l) as Recipe);

  console.log(`Loaded ${recipes.length} recipes from ${recipesPath}`);
  console.log(`Model: ${modelId}, batch size: ${batchSize}, concurrency: ${concurrency}`);
  console.log("─".repeat(60));

  // Split into batches
  const batches: Recipe[][] = [];
  for (let i = 0; i < recipes.length; i += batchSize) {
    batches.push(recipes.slice(i, i + batchSize));
  }
  console.log(`${batches.length} batches to process\n`);

  const client = new OpenAI();
  const allGenerated: GeneratedJob[] = [];
  let jobCounter = 1;

  // Process batches with concurrency
  await processWithConcurrency(
    batches,
    concurrency,
    async (batch: Recipe[], batchIdx: number) => {
      const prompt = buildPrompt(batch);

      for (let attempt = 0; attempt <= 6; attempt++) {
        try {
          const response = await client.chat.completions.create({
            model: modelId,
            messages: [
              {
                role: "system",
                content:
                  "You generate realistic job postings. Output ONLY valid JSON arrays. Each job must have title, company, location, jd_text fields.",
              },
              { role: "user", content: prompt },
            ],
            max_tokens: 8000,
            temperature: 0.7,
          });

          const content = response.choices[0]?.message?.content ?? "";

          // Parse JSON array
          let jobs: Array<{
            title: string;
            company: string;
            location: string;
            jd_text: string;
          }>;
          try {
            const cleaned = content
              .replace(/```json?\n?/g, "")
              .replace(/```/g, "")
              .trim();
            jobs = JSON.parse(cleaned);
            if (!Array.isArray(jobs)) jobs = [jobs];
          } catch {
            console.warn(
              `  Batch ${batchIdx + 1}: JSON parse error, retrying...`,
            );
            if (attempt < 6) continue;
            console.error(`  Batch ${batchIdx + 1}: FAILED after 7 attempts`);
            return;
          }

          // Map generated jobs to recipes
          for (let j = 0; j < Math.min(jobs.length, batch.length); j++) {
            const job = jobs[j]!;
            const recipe = batch[j]!;
            if (!job.title || !job.jd_text) continue;

            const noisyJd = addNoise(job.jd_text);
            const genJob: GeneratedJob = {
              job_id: `gen_v7_${String(jobCounter++).padStart(4, "0")}`,
              title: job.title,
              company: job.company ?? "Generated Co",
              location: recipe.location_text, // Use exact recipe location
              jd_text: noisyJd,
              source_file: "generated_v7",
              recipe_id: recipe.recipe_id,
              recipe_targets: {
                location: recipe.location,
                work_arrangement: recipe.work_arrangement,
                scope: recipe.scope,
                seniority: recipe.seniority,
                tech: recipe.tech,
                comp: recipe.comp,
              },
            };
            allGenerated.push(genJob);
          }

          console.log(
            `  Batch ${batchIdx + 1}/${batches.length}: generated ${Math.min(jobs.length, batch.length)} jobs`,
          );
          return;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg.includes("429") && attempt < 6) {
            const waitMs = 2000 * Math.pow(2, attempt);
            console.warn(
              `  Rate limited, waiting ${(waitMs / 1000).toFixed(0)}s...`,
            );
            await new Promise((r) => setTimeout(r, waitMs));
            continue;
          }
          console.error(
            `  Batch ${batchIdx + 1}: ERROR: ${msg}`,
          );
          return;
        }
      }
    },
  );

  // Output
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  fs.writeFileSync(
    outputPath,
    allGenerated.map((j) => JSON.stringify(j)).join("\n") + "\n",
  );

  console.log(`\n${"─".repeat(60)}`);
  console.log(`Generated ${allGenerated.length} jobs`);
  console.log(`Output: ${outputPath}`);

  // Distribution summary
  const counts: Record<string, Record<string, number>> = {};
  for (const job of allGenerated) {
    for (const [field, token] of Object.entries(job.recipe_targets)) {
      if (!counts[field]) counts[field] = {};
      counts[field][token] = (counts[field][token] ?? 0) + 1;
    }
  }
  console.log(`\nGenerated distribution:`);
  for (const [field, tokens] of Object.entries(counts)) {
    const sorted = Object.entries(tokens).sort((a, b) => b[1] - a[1]);
    console.log(`  ${field}: ${sorted.map(([t, c]) => `${t}=${c}`).join(", ")}`);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});

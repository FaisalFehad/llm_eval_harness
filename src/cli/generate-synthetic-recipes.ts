/**
 * Generate synthetic jobs using specific "recipes" — field combinations
 * designed to fill multiple gaps at once.
 *
 * Each recipe specifies exact loc/role/tech/comp targets so the generated
 * job hits a known score and fills specific distribution gaps.
 *
 * Usage:
 *   npx tsx src/cli/generate-synthetic-recipes.ts \
 *     --output data/v5/synthetic_recipes.jsonl \
 *     --model gpt-4o-mini
 */

import * as fs from "node:fs";
import OpenAI from "openai";
import { parseArgs, getStringArg } from "../lib/args.js";

// ── Recipes ─────────────────────────────────────────────────────────────

type Recipe = {
  id: string;
  count: number;
  score: number;
  label: string;
  gapsFilled: string[];
  prompt: string;
};

const RECIPES: Recipe[] = [
  // ── good_fit recipes (score ≥70) ──────────────────────────────────
  {
    id: "A",
    count: 8,
    score: 80,
    label: "good_fit",
    gapsFilled: ["good_fit", "RANGE_75_99K"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK (e.g. "London, UK", "Remote - UK", "London (Remote)")
- Title: MUST contain "Senior" or "Lead" (seniority keyword required)
- Tech in description: MUST mention Node.js AND TypeScript as required skills. Also mention React and PostgreSQL but these don't affect scoring.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£70,000-£90,000", "£80k-£95k")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement. Use inconsistent capitalisation for tech names. Include at least one misleading detail (e.g. mention a US office in company description).`,
  },
  {
    id: "B",
    count: 5,
    score: 80,
    label: "good_fit",
    gapsFilled: ["good_fit", "MID_LEVEL"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK
- Title: MUST contain "Full Stack" or "Mid-Level" (mid-level keyword required, NOT Senior)
- Tech in description: MUST mention Node.js AND TypeScript/JavaScript as required skills
- Salary: MUST include a GBP salary with midpoint ≥ £100k (e.g. "£95,000-£120,000", "£100k-£130k")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement. Vary formatting.`,
  },
  {
    id: "C",
    count: 10,
    score: 75,
    label: "good_fit",
    gapsFilled: ["good_fit", "NODE", "RANGE_75_99K"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK
- Title: MUST contain "Senior", "Lead", or "Staff" (seniority keyword required)
- Tech in description: MUST mention Node.js (or NodeJS/node.js) as a required skill. Do NOT separately mention JavaScript or TypeScript — only Node.js. Can mention Python, Go, Docker as additional tech.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£75,000-£95,000", "£80k-£100k")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement. Vary capitalisation of Node.js.`,
  },
  {
    id: "D",
    count: 5,
    score: 75,
    label: "good_fit",
    gapsFilled: ["good_fit", "NODE_JS_TS_AI_ML"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK
- Title: MUST contain "Senior", "Lead", or "Staff"
- Tech in description: MUST mention Node.js AND TypeScript AND explicitly REQUIRE AI/ML/LLM experience (not just "nice to have" — must be in required skills section). This is an AI-focused full-stack role.
- Salary: No GBP annual salary visible (can have USD or no salary at all)

jd_text: 200-400 words. Include company description, benefits. The AI/ML requirement must be clear and prominent.`,
  },
  {
    id: "E",
    count: 5,
    score: 70,
    label: "good_fit",
    gapsFilled: ["good_fit", "JS_TS_AI_ML"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK
- Title: MUST contain "Senior", "Lead", or "Staff"
- Tech in description: MUST mention TypeScript or JavaScript AND explicitly REQUIRE AI/ML/LLM experience. Do NOT mention Node.js. This is a frontend/AI hybrid role.
- Salary: MUST include a GBP range with midpoint £55k-£74k (e.g. "£55,000-£70,000", "£60k-£75k")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement.`,
  },

  // ── maybe recipes (score 50-69) ───────────────────────────────────
  {
    id: "F",
    count: 10,
    score: 50,
    label: "maybe",
    gapsFilled: ["maybe", "NODE"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: UK but NOT London and NOT Remote (e.g. "Manchester, UK", "Bristol, England", "Edinburgh, Scotland", "Leeds, UK", "Birmingham, UK"). Vary the city.
- Title: MUST contain "Senior" or "Lead" (seniority keyword required)
- Tech in description: MUST mention Node.js (or NodeJS/node.js) as required. Do NOT separately mention JavaScript or TypeScript. Can mention other tech like Python, Redis, Docker.
- Salary: MUST include a GBP range with midpoint £55k-£74k (e.g. "£55,000-£70,000", "£60k-£75k")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement. Vary formatting.`,
  },
  {
    id: "G",
    count: 5,
    score: 55,
    label: "maybe",
    gapsFilled: ["maybe", "NODE"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK
- Title: Do NOT include any seniority keyword (no Senior, Lead, Staff, etc.). Just "Software Engineer", "Backend Developer", "Platform Engineer", etc.
- Tech in description: MUST mention Node.js (or NodeJS) as required. Do NOT separately mention JavaScript or TypeScript.
- Salary: MUST include a GBP range with midpoint £75k-£99k (e.g. "£75,000-£90,000")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement.`,
  },
  {
    id: "H",
    count: 3,
    score: 50,
    label: "maybe",
    gapsFilled: ["maybe", "NODE_JS_TS_AI_ML"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: UK but NOT London and NOT Remote (e.g. "Cambridge, UK", "Oxford, England", "Reading, UK")
- Title: MUST contain "Senior" or "Lead"
- Tech in description: MUST mention Node.js AND TypeScript AND explicitly REQUIRE AI/ML experience
- Salary: No GBP annual salary (no salary info, or USD salary)

jd_text: 200-400 words. Include company description, benefits.`,
  },
  {
    id: "I",
    count: 3,
    score: 55,
    label: "maybe",
    gapsFilled: ["maybe", "JS_TS_AI_ML"],
    prompt: `Generate a realistic job posting as JSON: {"title": "...", "company": "...", "location": "...", "jd_text": "..."}.

This job must have these properties when scored:
- Location: London or Remote UK
- Title: Do NOT include any seniority keyword
- Tech in description: MUST mention TypeScript or JavaScript AND explicitly REQUIRE AI/ML experience. Do NOT mention Node.js.
- Salary: MUST include a GBP range with midpoint £55k-£74k (e.g. "£60,000-£75,000")

jd_text: 200-400 words. Include company description, benefits, equal opportunity statement.`,
  },
];

// ── Main ─────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const outputPath = getStringArg(args, "output") ?? "data/v5/synthetic_recipes.jsonl";
  const modelId = getStringArg(args, "model") ?? "gpt-4o-mini";

  if (!process.env.OPENAI_API_KEY) {
    console.error("ERROR: OPENAI_API_KEY environment variable is required");
    process.exit(1);
  }

  const client = new OpenAI();
  const totalJobs = RECIPES.reduce((sum, r) => sum + r.count, 0);

  console.log(`Generating ${totalJobs} synthetic jobs across ${RECIPES.length} recipes`);
  console.log(`Model: ${modelId}`);
  console.log("─".repeat(60));

  const allJobs: Array<Record<string, unknown>> = [];
  let globalId = 1;

  for (const recipe of RECIPES) {
    console.log(`\nRecipe ${recipe.id}: ${recipe.count} jobs → ${recipe.label} (score ~${recipe.score})`);
    console.log(`  Gaps filled: ${recipe.gapsFilled.join(", ")}`);

    // Generate one at a time for quality control
    for (let i = 0; i < recipe.count; i++) {
      let content = "";
      for (let attempt = 0; attempt <= 6; attempt++) {
        try {
          const response = await client.chat.completions.create({
            model: modelId,
            messages: [
              {
                role: "system",
                content: "You generate realistic job postings. Output ONLY valid JSON, no markdown fences.",
              },
              { role: "user", content: recipe.prompt },
            ],
            max_tokens: 2000,
            temperature: 0.9,
          });
          content = response.choices[0]?.message?.content ?? "";
          break;
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          if (msg.includes("429") && attempt < 6) {
            const waitMs = 2000 * Math.pow(2, attempt);
            console.warn(`  Rate limited, waiting ${(waitMs / 1000).toFixed(0)}s...`);
            await new Promise((r) => setTimeout(r, waitMs));
            continue;
          }
          throw err;
        }
      }

      // Parse
      try {
        const cleaned = content.replace(/```json?\n?/g, "").replace(/```/g, "").trim();
        let parsed = JSON.parse(cleaned);
        // Handle array or single object
        if (Array.isArray(parsed)) parsed = parsed[0];
        if (!parsed?.title || !parsed?.jd_text) {
          console.warn(`  Job ${i + 1}: missing fields, skipping`);
          continue;
        }

        allJobs.push({
          job_id: `synth_recipe_${String(globalId++).padStart(4, "0")}`,
          title: parsed.title,
          company: parsed.company ?? "Synthetic Co",
          location: parsed.location ?? "",
          jd_text: parsed.jd_text,
          source_file: "synthetic_v5",
          synthetic_target: `recipe_${recipe.id}`,
          recipe_label: recipe.label,
          recipe_score: recipe.score,
          recipe_gaps: recipe.gapsFilled,
        });

        process.stdout.write(".");
      } catch {
        console.warn(`  Job ${i + 1}: parse error, skipping`);
      }
    }
    console.log(` (${allJobs.length} total)`);
  }

  // Write output
  const outStream = fs.createWriteStream(outputPath);
  for (const job of allJobs) {
    outStream.write(JSON.stringify(job) + "\n");
  }
  outStream.end();

  console.log("\n" + "═".repeat(60));
  console.log("RECIPE GENERATION COMPLETE");
  console.log("═".repeat(60));
  console.log(`Generated: ${allJobs.length} synthetic jobs`);
  console.log(`Output: ${outputPath}`);
  console.log(`\nNext: label with teacher prompt, then verify tokens match intended recipes.`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

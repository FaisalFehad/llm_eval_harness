import { scoreJob, scoreComp, scoreTech } from "../lib/deterministic-scorer-human-corrected.js";
import { readFileSync } from "fs";

// Test comp parsing for specific salary strings
console.log("=== COMP PARSING TESTS ===");
console.log("£50-100k:", scoreComp("Benefits £50-100k salary, depending on experience"));
console.log("£40,000-£60,000:", scoreComp("Salary: £40,000-£60,000"));
console.log("£93,000—£100,000:", scoreComp("Salary Range£93,000—£100,000 GBP"));
console.log("Up to £180k:", scoreComp("Salary: Up to £180k Base + Exceptional Bonus"));
console.log("Salary to £100,000:", scoreComp("Salary to £100,000"));
console.log("up to £120,000:", scoreComp("opportunity paying up to £120,000 base salary"));
console.log();

// Test tech scoring for ML-heavy JDs
console.log("=== TECH SCORING TESTS ===");
console.log("ML core duty:", scoreTech("design, train, and evaluate the large-scale ML models that are used throughout"));
console.log("ML nice-to-have:", scoreTech("Exposure to machine learning or a strong interest in practical AI"));
console.log("AI SaaS:", scoreTech("Experience with REST APIs, GitHub, AI SaaS workflows"));
console.log();

// Check all error jobs
console.log("=== ERROR JOB ANALYSIS ===");
const lines = readFileSync("data/linkedin_teacher_v2_eval_human_corrected.jsonl", "utf-8").trim().split("\n");
const checkJobs = [2, 37, 40, 58, 61, 63, 90, 100, 101];

for (const idx of checkJobs) {
  const j = JSON.parse(lines[idx - 1]);
  const result = scoreJob(j.title, j.location, j.jd_text);
  const stored = { loc: j.loc, role: j.role, tech: j.tech, comp: j.comp, score: j.score, label: j.label };

  const diffs: string[] = [];
  for (const k of ["loc", "role", "tech", "comp"] as const) {
    if (result[k] !== stored[k]) {
      diffs.push(`${k}: stored=${stored[k]} computed=${result[k]}`);
    }
  }
  if (result.score !== stored.score) diffs.push(`score: ${stored.score}->${result.score}`);
  if (result.label !== stored.label) diffs.push(`label: ${stored.label}->${result.label}`);

  console.log(`[${idx}] ${j.title.substring(0, 55)}`);
  console.log(`  Stored:   loc=${stored.loc} role=${stored.role} tech=${stored.tech} comp=${stored.comp} => ${stored.label} (${stored.score})`);
  console.log(`  Computed: loc=${result.loc} role=${result.role} tech=${result.tech} comp=${result.comp} => ${result.label} (${result.score})`);
  if (diffs.length > 0) {
    console.log(`  DIFFS:    ${diffs.join(", ")}`);
  } else {
    console.log(`  MATCH`);
  }
  console.log();
}

/**
 * Validate API-generated labels against the deterministic scorer.
 *
 * Compares label, loc, role, tech, comp for each job and prints
 * agreement stats + disagreement details.
 *
 * Usage:
 *   npx tsx src/cli/validate-labels.ts \
 *     --input "data/Student Training Data/teacher_labeled_500.jsonl"
 */

import { parseArgs, getStringArg } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";
import { scoreJob } from "../lib/deterministic-scorer-human-corrected.js";

type LabeledJob = {
  job_id: string;
  title: string;
  company: string;
  location: string;
  jd_text: string;
  label: string;
  score: number;
  loc: number;
  role: number;
  tech: number;
  comp: number;
  reasoning?: string;
};

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath =
    getStringArg(args, "input") ??
    "data/Student Training Data/teacher_labeled_500.jsonl";

  const jobs = await readJsonlFile<LabeledJob>(inputPath);
  console.log(`Loaded ${jobs.length} labeled jobs from ${inputPath}\n`);

  let labelMatch = 0;
  let locMatch = 0;
  let roleMatch = 0;
  let techMatch = 0;
  let compMatch = 0;

  const disagreements: Array<{
    job_id: string;
    title: string;
    location: string;
    api: { label: string; loc: number; role: number; tech: number; comp: number };
    det: { label: string; loc: number; role: number; tech: number; comp: number };
  }> = [];

  for (const job of jobs) {
    const det = scoreJob(job.title, job.location, job.jd_text);

    if (job.label === det.label) labelMatch++;
    if (job.loc === det.loc) locMatch++;
    if (job.role === det.role) roleMatch++;
    if (job.tech === det.tech) techMatch++;
    if (job.comp === det.comp) compMatch++;

    if (job.label !== det.label || job.loc !== det.loc || job.comp !== det.comp) {
      disagreements.push({
        job_id: job.job_id,
        title: job.title,
        location: job.location,
        api: { label: job.label, loc: job.loc, role: job.role, tech: job.tech, comp: job.comp },
        det: { label: det.label, loc: det.loc, role: det.role, tech: det.tech, comp: det.comp },
      });
    }
  }

  const n = jobs.length;
  const pct = (v: number) => `${((v / n) * 100).toFixed(1)}%`;

  console.log("═══════════════════════════════════════════════════");
  console.log("  AGREEMENT: API labels vs Deterministic Scorer");
  console.log("═══════════════════════════════════════════════════");
  console.log(`  Label:  ${labelMatch}/${n} (${pct(labelMatch)})`);
  console.log(`  Loc:    ${locMatch}/${n} (${pct(locMatch)})`);
  console.log(`  Role:   ${roleMatch}/${n} (${pct(roleMatch)})`);
  console.log(`  Tech:   ${techMatch}/${n} (${pct(techMatch)})`);
  console.log(`  Comp:   ${compMatch}/${n} (${pct(compMatch)})`);
  console.log("═══════════════════════════════════════════════════\n");

  // Label confusion matrix
  const labels = ["good_fit", "maybe", "bad_fit"];
  console.log("Label confusion matrix (rows=API, cols=Deterministic):");
  const header = "           " + labels.map((l) => l.padStart(10)).join("") + "   TOTAL";
  console.log(header);
  for (const apiLabel of labels) {
    const cells = labels.map((detLabel) => {
      const count = jobs.filter(
        (j) => j.label === apiLabel && scoreJob(j.title, j.location, j.jd_text).label === detLabel,
      ).length;
      return String(count).padStart(10);
    });
    const total = jobs.filter((j) => j.label === apiLabel).length;
    console.log(`${apiLabel.padEnd(10)} ${cells.join("")}   ${total}`);
  }

  // Show sample disagreements
  if (disagreements.length > 0) {
    console.log(`\n── ${disagreements.length} disagreements (label or loc or comp) ──\n`);
    const show = disagreements.slice(0, 20);
    for (const d of show) {
      console.log(`  ${d.title.slice(0, 50)}`);
      console.log(`    Location: ${d.location}`);
      console.log(
        `    API: label=${d.api.label}, loc=${d.api.loc}, role=${d.api.role}, ` +
        `tech=${d.api.tech}, comp=${d.api.comp}`,
      );
      console.log(
        `    DET: label=${d.det.label}, loc=${d.det.loc}, role=${d.det.role}, ` +
        `tech=${d.det.tech}, comp=${d.det.comp}`,
      );
      console.log();
    }
    if (disagreements.length > 20) {
      console.log(`  ... and ${disagreements.length - 20} more`);
    }
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

/**
 * Analyze student model predictions against teacher (OpenAI) labels.
 *
 * Reads a .predictions.jsonl file from eval_finetuned.py --save-predictions,
 * generates a detailed terminal report + markdown file.
 *
 * Usage:
 *   npx tsx src/cli/analyze-student-eval.ts \
 *     --input eval_results/2026-03-06_openai_eval_labels_scorer_v9.8_Qwen2.5-0.5B-Instruct-4bit.predictions.jsonl \
 *     --output eval_results/student_analysis.md
 */

import * as fs from "node:fs";
import { parseArgs, getStringArg } from "../lib/args.js";
import { readJsonlFile } from "../lib/jsonl.js";

// ── Types ────────────────────────────────────────────────────────────────────

type Prediction = {
  job_index: number;
  title: string;
  company: string;
  location: string;
  parse_fail: boolean;
  golden: {
    label: string;
    loc: number | null;
    role: number | null;
    tech: number | null;
    comp: number | null;
    score?: number;
  };
  pred: {
    label: string;
    loc: number;
    role: number;
    tech: number;
    comp: number;
    score: number;
    reasoning: string;
  } | null;
};

type TrainingSuggestion = {
  priority: "HIGH" | "MEDIUM" | "LOW";
  category: string;
  description: string;
  count: number;
  examples: string[];
};

const LABELS = ["good_fit", "maybe", "bad_fit"] as const;
const FIELDS = ["loc", "role", "tech", "comp"] as const;

// ── Analysis Functions ───────────────────────────────────────────────────────

function computeSummary(preds: Prediction[]) {
  const valid = preds.filter((p) => !p.parse_fail && p.pred);
  const parseFailures = preds.filter((p) => p.parse_fail);

  const labelCorrect = valid.filter((p) => p.pred!.label === p.golden.label).length;

  const fieldAccuracy: Record<string, { correct: number; total: number }> = {};
  for (const field of FIELDS) {
    const scored = valid.filter((p) => p.golden[field] != null);
    const correct = scored.filter((p) => p.pred![field] === p.golden[field]).length;
    fieldAccuracy[field] = { correct, total: scored.length };
  }

  return { total: preds.length, valid: valid.length, parseFailures: parseFailures.length,
    labelCorrect, labelPct: valid.length > 0 ? (labelCorrect / valid.length) * 100 : 0,
    fieldAccuracy };
}

function computePerLabelBreakdown(preds: Prediction[]) {
  const valid = preds.filter((p) => !p.parse_fail && p.pred);
  const breakdown: Record<string, { correct: number; total: number; fieldAcc: Record<string, { correct: number; total: number }> }> = {};

  for (const label of LABELS) {
    const group = valid.filter((p) => p.golden.label === label);
    const correct = group.filter((p) => p.pred!.label === label).length;
    const fieldAcc: Record<string, { correct: number; total: number }> = {};
    for (const field of FIELDS) {
      const scored = group.filter((p) => p.golden[field] != null);
      const fc = scored.filter((p) => p.pred![field] === p.golden[field]).length;
      fieldAcc[field] = { correct: fc, total: scored.length };
    }
    breakdown[label] = { correct, total: group.length, fieldAcc };
  }

  return breakdown;
}

function computeConfusionMatrix(preds: Prediction[]) {
  const valid = preds.filter((p) => !p.parse_fail && p.pred);
  // matrix[golden][predicted] = count
  const matrix: Record<string, Record<string, number>> = {};
  for (const g of LABELS) {
    matrix[g] = {};
    for (const p of LABELS) matrix[g][p] = 0;
  }
  for (const pred of valid) {
    const g = pred.golden.label;
    const p = pred.pred!.label;
    if (g in matrix && p in matrix[g]!) matrix[g]![p]!++;
  }
  return matrix;
}

function computeFieldTransitions(preds: Prediction[]) {
  const valid = preds.filter((p) => !p.parse_fail && p.pred);
  const transitions: Record<string, Record<string, number>> = {};

  for (const field of FIELDS) {
    transitions[field] = {};
    for (const pred of valid) {
      const golden = pred.golden[field];
      const predicted = pred.pred![field];
      if (golden == null || golden === predicted) continue;
      const key = `${golden}→${predicted}`;
      transitions[field]![key] = (transitions[field]![key] ?? 0) + 1;
    }
  }

  return transitions;
}

function findWorstErrors(preds: Prediction[], count: number) {
  const valid = preds.filter((p) => !p.parse_fail && p.pred);
  const errors = valid
    .filter((p) => p.pred!.label !== p.golden.label)
    .map((p) => {
      const fieldErrors = FIELDS.filter(
        (f) => p.golden[f] != null && p.pred![f] !== p.golden[f],
      ).length;
      const scoreDiff = Math.abs((p.pred!.score ?? 0) - (p.golden.score ?? 0));
      return { ...p, fieldErrors, scoreDiff };
    })
    .sort((a, b) => b.scoreDiff - a.scoreDiff || b.fieldErrors - a.fieldErrors);

  return errors.slice(0, count);
}

function generateTrainingSuggestions(
  preds: Prediction[],
  confusionMatrix: Record<string, Record<string, number>>,
  fieldTransitions: Record<string, Record<string, number>>,
): TrainingSuggestion[] {
  const valid = preds.filter((p) => !p.parse_fail && p.pred);
  const suggestions: TrainingSuggestion[] = [];
  const MIN_COUNT = 3; // ignore patterns with fewer than 3 occurrences

  // ── 1. Confusion matrix: label-level over/under-scoring ──
  for (const golden of LABELS) {
    for (const predicted of LABELS) {
      if (golden === predicted) continue;
      const count = confusionMatrix[golden]?.[predicted] ?? 0;
      if (count < MIN_COUNT) continue;

      const goldenTotal = LABELS.reduce((s, p) => s + (confusionMatrix[golden]?.[p] ?? 0), 0);
      const errorRate = goldenTotal > 0 ? count / goldenTotal : 0;
      const priority = errorRate >= 0.2 ? "HIGH" : errorRate >= 0.1 ? "MEDIUM" : "LOW";

      const direction = LABELS.indexOf(predicted) < LABELS.indexOf(golden) ? "over-scoring" : "under-scoring";
      const examples = valid
        .filter((p) => p.golden.label === golden && p.pred!.label === predicted)
        .map((p) => p.title)
        .slice(0, 5);

      suggestions.push({
        priority,
        category: `Label confusion: ${golden}→${predicted}`,
        description: `Student is ${direction} ${golden} jobs as ${predicted} (${(errorRate * 100).toFixed(0)}% error rate). Add more ${golden} training examples with clear scoring.`,
        count,
        examples,
      });
    }
  }

  // ── 2. Field-level error transitions ──
  for (const field of FIELDS) {
    const trans = fieldTransitions[field] ?? {};
    for (const [key, count] of Object.entries(trans)) {
      if (count < MIN_COUNT) continue;

      const [goldenStr, predStr] = key.split("→");
      const goldenVal = Number(goldenStr);
      const predVal = Number(predStr);

      const fieldTotal = valid.filter((p) => p.golden[field] != null).length;
      const errorRate = fieldTotal > 0 ? count / fieldTotal : 0;
      const priority = errorRate >= 0.1 ? "HIGH" : errorRate >= 0.05 ? "MEDIUM" : "LOW";

      // Build actionable description based on common patterns
      let description: string;
      if (goldenVal === 0 && predVal > 0) {
        description = `Student hallucinating ${field}=${predVal} when golden is 0. Add more ${field}=0 examples to teach the model to score conservatively.`;
      } else if (goldenVal > 0 && predVal === 0) {
        description = `Student missing ${field}=${goldenVal}, predicting 0. Add more examples where ${field}=${goldenVal} with explicit evidence in the JD.`;
      } else if (predVal > goldenVal) {
        description = `Student over-scoring ${field} (${goldenVal}→${predVal}). Add training examples where ${field}=${goldenVal} with similar JD patterns.`;
      } else {
        description = `Student under-scoring ${field} (${goldenVal}→${predVal}). Add examples where ${field}=${goldenVal} to teach correct scoring.`;
      }

      const examples = valid
        .filter((p) => p.golden[field] === goldenVal && p.pred![field] === predVal)
        .map((p) => p.title)
        .slice(0, 5);

      suggestions.push({ priority, category: `Field error: ${field} ${key}`, description, count, examples });
    }
  }

  // ── 3. Parse failures ──
  const parseFailCount = preds.filter((p) => p.parse_fail).length;
  if (parseFailCount >= MIN_COUNT) {
    const rate = parseFailCount / preds.length;
    suggestions.push({
      priority: rate >= 0.05 ? "HIGH" : "MEDIUM",
      category: "Parse failures",
      description: `${parseFailCount} jobs produced unparseable output. Consider adding more diverse training examples with strict JSON format, or lowering max_tokens.`,
      count: parseFailCount,
      examples: preds.filter((p) => p.parse_fail).map((p) => p.title).slice(0, 5),
    });
  }

  // ── 4. Systematic bias: net over vs under-scoring ──
  const overScored = valid.filter((p) => (p.pred!.score ?? 0) > (p.golden.score ?? 0)).length;
  const underScored = valid.filter((p) => (p.pred!.score ?? 0) < (p.golden.score ?? 0)).length;
  const biasRatio = valid.length > 0 ? Math.abs(overScored - underScored) / valid.length : 0;
  if (biasRatio >= 0.15 && Math.abs(overScored - underScored) >= MIN_COUNT) {
    const direction = overScored > underScored ? "over" : "under";
    const count = Math.abs(overScored - underScored);
    suggestions.push({
      priority: biasRatio >= 0.3 ? "HIGH" : "MEDIUM",
      category: `Systematic ${direction}-scoring bias`,
      description: `Student ${direction}-scores ${count} more jobs than it ${direction === "over" ? "under" : "over"}-scores (${overScored} over vs ${underScored} under). Rebalance training data toward ${direction === "over" ? "lower" : "higher"}-scoring examples.`,
      count,
      examples: [],
    });
  }

  // Sort: HIGH → MEDIUM → LOW, then by count desc within each priority
  const priorityOrder = { HIGH: 0, MEDIUM: 1, LOW: 2 };
  suggestions.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority] || b.count - a.count);

  return suggestions;
}

// ── Formatting ───────────────────────────────────────────────────────────────

function formatReport(preds: Prediction[]): string {
  const summary = computeSummary(preds);
  const perLabel = computePerLabelBreakdown(preds);
  const matrix = computeConfusionMatrix(preds);
  const transitions = computeFieldTransitions(preds);
  const worstErrors = findWorstErrors(preds, 5);
  const suggestions = generateTrainingSuggestions(preds, matrix, transitions);

  const lines: string[] = [];
  const hr = "═".repeat(60);
  const divider = "─".repeat(60);

  // ── 1. Summary ──
  lines.push(hr);
  lines.push("STUDENT MODEL EVALUATION REPORT");
  lines.push(hr);
  lines.push(`Total jobs:       ${summary.total}`);
  lines.push(`Valid outputs:    ${summary.valid}`);
  lines.push(`Parse failures:   ${summary.parseFailures}`);
  lines.push("");
  lines.push(`Label accuracy:   ${summary.labelCorrect}/${summary.valid} = ${summary.labelPct.toFixed(1)}%`);
  lines.push("");
  lines.push("Field accuracy:");
  for (const field of FIELDS) {
    const fa = summary.fieldAccuracy[field]!;
    const pct = fa.total > 0 ? ((fa.correct / fa.total) * 100).toFixed(1) : "N/A";
    lines.push(`  ${field.padEnd(6)}: ${fa.correct}/${fa.total} = ${pct}%`);
  }

  // ── 2. Per-label breakdown ──
  lines.push("");
  lines.push(divider);
  lines.push("PER-LABEL BREAKDOWN");
  lines.push(divider);
  for (const label of LABELS) {
    const b = perLabel[label]!;
    if (b.total === 0) continue;
    const pct = ((b.correct / b.total) * 100).toFixed(0);
    lines.push(`  ${label.padEnd(10)}: ${b.correct}/${b.total} = ${pct}%`);
    for (const field of FIELDS) {
      const fa = b.fieldAcc[field]!;
      if (fa.total === 0) continue;
      const fpct = ((fa.correct / fa.total) * 100).toFixed(0);
      lines.push(`    ${field.padEnd(6)}: ${fa.correct}/${fa.total} = ${fpct}%`);
    }
  }

  // ── 3. Confusion matrix ──
  lines.push("");
  lines.push(divider);
  lines.push("CONFUSION MATRIX (rows=golden, cols=predicted)");
  lines.push(divider);
  const colW = 10;
  lines.push(`  ${"".padEnd(colW)} │ ${LABELS.map((l) => l.padEnd(colW)).join("│ ")}`);
  lines.push(`  ${"─".repeat(colW)}─┼─${LABELS.map(() => "─".repeat(colW)).join("┼─")}`);
  for (const g of LABELS) {
    const cells = LABELS.map((p) => {
      const count = matrix[g]![p]!;
      const str = String(count);
      return (g === p ? `[${str}]` : str).padEnd(colW);
    });
    lines.push(`  ${g.padEnd(colW)} │ ${cells.join("│ ")}`);
  }

  // ── 4. Field error transitions ──
  lines.push("");
  lines.push(divider);
  lines.push("FIELD ERROR TRANSITIONS (golden→predicted: count)");
  lines.push(divider);
  for (const field of FIELDS) {
    const trans = transitions[field]!;
    const sorted = Object.entries(trans).sort((a, b) => b[1] - a[1]);
    if (sorted.length === 0) {
      lines.push(`  ${field}: (no errors)`);
      continue;
    }
    lines.push(`  ${field}:`);
    for (const [key, count] of sorted) {
      lines.push(`    ${key}: ${count}`);
    }
  }

  // ── 5. Top worst errors ──
  lines.push("");
  lines.push(divider);
  lines.push("TOP 5 WORST ERRORS (by score difference)");
  lines.push(divider);
  for (let i = 0; i < worstErrors.length; i++) {
    const e = worstErrors[i]!;
    lines.push(`  ${i + 1}. [#${e.job_index}] ${e.title}`);
    lines.push(`     ${e.company} | ${e.location}`);
    lines.push(`     Golden: ${e.golden.label} (score=${e.golden.score ?? "?"})`);
    lines.push(`     Pred:   ${e.pred!.label} (score=${e.pred!.score})`);
    for (const field of FIELDS) {
      const g = e.golden[field];
      const p = e.pred![field];
      if (g != null && g !== p) {
        lines.push(`     ${field}: ${g}→${p}`);
      }
    }
    if (e.pred!.reasoning) {
      lines.push(`     Reasoning: ${e.pred!.reasoning.slice(0, 120)}`);
    }
    lines.push("");
  }

  // ── 6. Training suggestions ──
  if (suggestions.length > 0) {
    lines.push(divider);
    lines.push("TRAINING SUGGESTIONS (priority-ranked)");
    lines.push(divider);
    for (let i = 0; i < suggestions.length; i++) {
      const s = suggestions[i]!;
      lines.push(`  ${i + 1}. [${s.priority}] ${s.category}`);
      lines.push(`     ${s.description} (${s.count} cases)`);
      if (s.examples.length > 0) {
        lines.push(`     Examples: ${s.examples.slice(0, 3).join(", ")}`);
      }
    }
  }

  lines.push("");
  return lines.join("\n");
}

function formatMarkdown(preds: Prediction[]): string {
  const summary = computeSummary(preds);
  const perLabel = computePerLabelBreakdown(preds);
  const matrix = computeConfusionMatrix(preds);
  const transitions = computeFieldTransitions(preds);
  const worstErrors = findWorstErrors(preds, 5);
  const suggestions = generateTrainingSuggestions(preds, matrix, transitions);

  const lines: string[] = [];

  lines.push("# Student Model Evaluation Report");
  lines.push(`> Generated: ${new Date().toISOString().split("T")[0]}`);
  lines.push("");

  // Summary
  lines.push("## Summary");
  lines.push(`| Metric | Value |`);
  lines.push(`|--------|-------|`);
  lines.push(`| Total jobs | ${summary.total} |`);
  lines.push(`| Valid outputs | ${summary.valid} |`);
  lines.push(`| Parse failures | ${summary.parseFailures} |`);
  lines.push(`| **Label accuracy** | **${summary.labelCorrect}/${summary.valid} = ${summary.labelPct.toFixed(1)}%** |`);
  lines.push("");

  // Field accuracy
  lines.push("## Field Accuracy");
  lines.push(`| Field | Correct | Total | Accuracy |`);
  lines.push(`|-------|---------|-------|----------|`);
  for (const field of FIELDS) {
    const fa = summary.fieldAccuracy[field]!;
    const pct = fa.total > 0 ? ((fa.correct / fa.total) * 100).toFixed(1) : "N/A";
    lines.push(`| ${field} | ${fa.correct} | ${fa.total} | ${pct}% |`);
  }
  lines.push("");

  // Per-label
  lines.push("## Per-Label Breakdown");
  lines.push(`| Label | Correct | Total | Accuracy |`);
  lines.push(`|-------|---------|-------|----------|`);
  for (const label of LABELS) {
    const b = perLabel[label]!;
    if (b.total === 0) continue;
    const pct = ((b.correct / b.total) * 100).toFixed(0);
    lines.push(`| ${label} | ${b.correct} | ${b.total} | ${pct}% |`);
  }
  lines.push("");

  // Confusion matrix
  lines.push("## Confusion Matrix");
  lines.push(`| Golden \\ Predicted | ${LABELS.join(" | ")} |`);
  lines.push(`|---|${LABELS.map(() => "---").join("|")}|`);
  for (const g of LABELS) {
    const cells = LABELS.map((p) => {
      const count = matrix[g]![p]!;
      return g === p ? `**${count}**` : String(count);
    });
    lines.push(`| ${g} | ${cells.join(" | ")} |`);
  }
  lines.push("");

  // Field transitions
  lines.push("## Field Error Transitions");
  for (const field of FIELDS) {
    const trans = transitions[field]!;
    const sorted = Object.entries(trans).sort((a, b) => b[1] - a[1]);
    if (sorted.length === 0) continue;
    lines.push(`### ${field}`);
    lines.push(`| Transition | Count |`);
    lines.push(`|------------|-------|`);
    for (const [key, count] of sorted) {
      lines.push(`| ${key} | ${count} |`);
    }
    lines.push("");
  }

  // Worst errors
  lines.push("## Top 5 Worst Errors");
  for (let i = 0; i < worstErrors.length; i++) {
    const e = worstErrors[i]!;
    lines.push(`### ${i + 1}. ${e.title} (#${e.job_index})`);
    lines.push(`- **Company:** ${e.company}`);
    lines.push(`- **Location:** ${e.location}`);
    lines.push(`- **Golden:** ${e.golden.label} (score=${e.golden.score ?? "?"})`);
    lines.push(`- **Predicted:** ${e.pred!.label} (score=${e.pred!.score})`);
    const fieldDiffs = FIELDS.filter((f) => e.golden[f] != null && e.pred![f] !== e.golden[f])
      .map((f) => `${f}: ${e.golden[f]}→${e.pred![f]}`);
    if (fieldDiffs.length > 0) lines.push(`- **Field diffs:** ${fieldDiffs.join(", ")}`);
    if (e.pred!.reasoning) lines.push(`- **Reasoning:** ${e.pred!.reasoning.slice(0, 200)}`);
    lines.push("");
  }

  // Training suggestions
  if (suggestions.length > 0) {
    lines.push("## Training Suggestions");
    for (let i = 0; i < suggestions.length; i++) {
      const s = suggestions[i]!;
      const icon = s.priority === "HIGH" ? "🔴" : s.priority === "MEDIUM" ? "🟡" : "🟢";
      lines.push(`${i + 1}. ${icon} **[${s.priority}] ${s.category}** — ${s.description} (${s.count} cases)`);
      if (s.examples.length > 0) {
        lines.push(`   - Examples: ${s.examples.slice(0, 3).join(", ")}`);
      }
    }
  }

  lines.push("");
  return lines.join("\n");
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const args = parseArgs();
  const inputPath = getStringArg(args, "input");
  const outputPath = getStringArg(args, "output") ?? inputPath?.replace(/\.predictions\.jsonl$/, ".analysis.md");

  if (!inputPath) {
    console.error("Usage: npx tsx src/cli/analyze-student-eval.ts --input <predictions.jsonl> [--output <report.md>]");
    process.exit(1);
  }

  if (!fs.existsSync(inputPath)) {
    console.error(`File not found: ${inputPath}`);
    process.exit(1);
  }

  const preds = await readJsonlFile<Prediction>(inputPath);
  console.log(`Loaded ${preds.length} predictions from ${inputPath}\n`);

  // Print terminal report
  const report = formatReport(preds);
  console.log(report);

  // Write markdown report
  if (outputPath) {
    const md = formatMarkdown(preds);
    fs.writeFileSync(outputPath, md);
    console.log(`\nMarkdown report: ${outputPath}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});

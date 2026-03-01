# Data & Script Reference

This file tracks every data file and script in the project — what each one is,
how it was generated, and critically, **which data the fine-tuned model has seen**
so you never accidentally eval a model on its own training data.

---

## ⚠️ Contamination Boundary

The fine-tuned model (`finetune/adapters/0000200_adapters.safetensors`) was trained on
`data/finetune/mlx/train.jsonl`, which contains **93 examples derived from 70 of the
103 golden jobs** (split seed=42).

```
CONTAMINATED (model trained on these)     CLEAN (model never seen these)
─────────────────────────────────────     ──────────────────────────────────────
data/finetune/train.jsonl (70 jobs)       data/finetune/test.jsonl (33 jobs)
data/finetune/mlx/train.jsonl (93 ex.)    data/new_uk_jobs_golden.jsonl (72 jobs)
                                          data/software_engineer_salaries_golden.jsonl (868 jobs)
```

The 33-job test set and both "golden" datasets were **never touched during training**.
Use these for honest eval of the fine-tuned model.

---

## Data Files

### Master Ground Truth

#### `data/golden_jobs.jsonl`
- **What**: 103 hand-labeled jobs — the master ground truth dataset for this project.
- **How created**: Manually scored by the project author against the v9 scoring rubric
  (loc/role/tech/comp). Corrected in Phase 5 after finding 20 wrong labels (13 location
  errors, 8 compensation errors). 57 are real LinkedIn jobs; 46 are synthetic jobs
  generated with Faker.js to cover edge cases.
- **Fields**: `job_id, title, company, location, jd_text, label, score, reasoning`
- **Label distribution**: 51 maybe / 39 bad_fit / 13 good_fit
- **Contamination**: ⚠️ 70 of these jobs were used for training. Do not use this file
  directly for eval of the fine-tuned model — use `finetune/test.jsonl` instead.
- **Script to regenerate**: Not regeneratable — hand-labeled ground truth.

#### `data/golden_jobs_original.jsonl`
- **What**: Backup of `golden_jobs.jsonl` before the Phase 5 label corrections.
- **How created**: Copied before corrections were applied.
- **Use**: Reference only. Do not use for training or eval — contains known wrong labels.

---

### Fine-tuning Pipeline Data

These files are derived from `golden_jobs.jsonl` and feed the LoRA training pipeline.
**They share the contamination boundary of the source data.**

#### `data/golden_jobs_scored.jsonl`
- **What**: All 103 golden jobs with computed `loc`, `role`, `tech`, `comp` scores
  added by the deterministic scorer.
- **How created**: `npm run golden:breakdowns` → runs `add-score-breakdowns-to-golden-jobs.ts`
  against `golden_jobs.jsonl`.
- **Fields**: All golden_jobs fields + `loc, role, tech, comp, computed_score,
  computed_label, score_match, label_match`
- **Contamination**: ⚠️ Input to the train/test split — do not use for final eval.
- **Script to regenerate**: `npm run golden:breakdowns`

#### `data/finetune/train.jsonl`
- **What**: 70 jobs selected for LoRA training (stratified split, seed=42).
- **How created**: `npm run finetune:split` → splits `golden_jobs_scored.jsonl` 70/33.
- **Label distribution**: 34 maybe / 27 bad_fit / 9 good_fit
- **Contamination**: 🔴 **TRAINING DATA** — the fine-tuned model was trained on these.
  Never eval the fine-tuned model against this file.
- **Script to regenerate**: `npm run finetune:split`

#### `data/finetune/mlx/train.jsonl`
- **What**: 93 MLX chat-format training examples (63 unique jobs from train.jsonl +
  30 oversampled duplicates of hard cases: non-UK locations, non-senior titles,
  "Up to £X" comp, non-qualifying tech).
- **How created**: `npm run finetune:format` → formats `finetune/train.jsonl` into
  MLX `{messages: [system, user, assistant]}` format, then hard cases were duplicated
  manually.
- **Contamination**: 🔴 **TRAINING DATA** — this is the exact file fed to `mlx_lm.lora`.
- **Script to regenerate**: `npm run finetune:format` (produces 70 examples; oversampling
  was done manually on top)

#### `data/finetune/test.jsonl`
- **What**: 33 held-out jobs — the honest test set for the fine-tuned model.
- **How created**: `npm run finetune:split` → the 33 jobs NOT selected for train.jsonl.
  These jobs were never seen during training, prompt engineering, or data preparation.
- **Label distribution**: 17 maybe / 12 bad_fit / 4 good_fit
- **Contamination**: ✅ **CLEAN** — safe to eval the fine-tuned model against this.
- **Eval result**: Fine-tuned model scored **90.9%** label accuracy on this set.
- **Script to regenerate**: `npm run finetune:split` (same seed=42 always produces the
  same split)

#### `data/finetune/split_manifest.json`
- **What**: Metadata about the train/test split: seed, counts, distributions, and the
  exact list of job_ids in each set.
- **Use**: Reference this to check whether a specific job_id is in the training set
  before using it in an eval.

---

### New Unseen Data (Safe for Eval)

These files were created **after** fine-tuning and have **zero overlap** with training data.

#### `data/new_uk_jobs.jsonl`
- **What**: 205 raw UK jobs scraped from LinkedIn after fine-tuning was complete.
- **How created**: Manual LinkedIn scrape. 75 records have empty `jd_text`, 58 have
  very short JD (<50 chars) — likely LinkedIn blocked full JD extraction.
- **Fields**: `job_id, title, company, location, jd_text, source_url`
- **Contamination**: ✅ **CLEAN** — no labels yet, not usable for eval directly.
- **Use**: Source data. Run through `score-raw-jobs-to-golden-format.ts` to produce
  eval-ready labeled data.

#### `data/new_uk_jobs_golden.jsonl`
- **What**: 72 UK jobs from `new_uk_jobs.jsonl` that had sufficient JD text, auto-labeled
  by the deterministic scorer.
- **How created**: `npx tsx src/cli/score-raw-jobs-to-golden-format.ts` — filters out
  empty/short JDs then applies the deterministic scorer to generate labels.
- **Fields**: `job_id, title, company, location, jd_text, label, score, reasoning,
  loc, role, tech, comp, source_url`
- **Label distribution**: 28 maybe / 44 bad_fit / 0 good_fit (no GBP salaries in JDs)
- **Contamination**: ✅ **CLEAN** — safe to eval fine-tuned model against this.
- **Eval result**: Fine-tuned model scored **81.9%** label accuracy on this set.
- **Limitation**: Labels are deterministic-scorer output, not hand-verified. Treat as
  approximate ground truth.

#### `data/software_engineer_salaries_golden.jsonl`
- **What**: 868 US-based software engineering jobs from a Glassdoor/Indeed salary CSV,
  auto-labeled by the deterministic scorer.
- **How created**: `npx tsx src/cli/convert-salary-csv-to-golden-jobs.ts` from
  `data/Software Engineer Salaries.csv`.
- **Fields**: `job_id, title, company, location, jd_text, label, score, reasoning`
- **Label distribution**: 868 bad_fit / 0 maybe / 0 good_fit
- **Contamination**: ✅ **CLEAN** — safe to eval against, but severely limited.
- **Limitation**: All 868 records are `bad_fit` (US locations → loc=-50 overwhelms
  everything else). A model that always outputs `bad_fit` would score 100%. Only useful
  as a sanity check that the model correctly rejects non-UK jobs — not a true accuracy
  test.
- **Eval result**: Fine-tuned model scored **100%** (expected and uninformative).

---

### Promptfoo Test Cases

#### `data/promptfoo_tests.yaml`
- **What**: Promptfoo-format test cases built from `golden_jobs.jsonl`. Used by
  `eval-runner.ts` as the source of test jobs.
- **How created**: `npm run promptfoo:tests`
- **Contamination**: ⚠️ Contains all 103 golden jobs including training jobs.
  `eval-runner.ts` samples from this file at runtime — when running on fine-tuned model,
  use only `finetune/test.jsonl` (via `finetune/eval_finetuned.py`) to stay clean.

#### `data/promptfoo_tests_original.yaml`
- **What**: Backup of `promptfoo_tests.yaml` before Phase 5 label corrections.
- **Use**: Reference only. Do not use for eval.

#### `data/promptfoo_tests_subset.yaml`
- **What**: Temporary subset file generated by eval runs at runtime.
- **Use**: Intermediate artefact. Ignored by git (`data/promptfoo_tests_eval.yaml`
  pattern).

---

### Source / Raw Data

#### `data/Software Engineer Salaries.csv`
- **What**: Raw CSV export of US software engineer salary listings from Glassdoor/Indeed.
- **Use**: Source data for `convert-salary-csv-to-golden-jobs.ts`. Keep for
  reproducibility — regenerating `software_engineer_salaries_golden.jsonl` requires it.

---

## Scripts

### Eval Scripts

#### `src/cli/eval-runner.ts`  →  `npm run eval`
Runs inference on all models in a Promptfoo config YAML using node-llama-cpp with
grammar-constrained JSON decoding. Samples a balanced/stratified subset of test jobs
at runtime. Core of the eval pipeline for prompt engineering and model comparison.
- `--model <name>` — run a single model
- `--jobs <n>` — number of jobs (default 103)
- `--tag <name>` — label this run
- `--config <path>` — use a different Promptfoo config

#### `src/cli/run-tagged-eval-with-snapshot.ts`  →  `npm run eval:tagged`
Wrapper around `eval-runner.ts` that also snapshots the config YAML and prompt files
into `results/runs/<timestamp>_<tag>/` and appends an entry to `results/iteration_log.csv`.
Use this when you want a reproducible record of an experiment (prompt change, model swap).

#### `src/cli/prompt-lab.ts`  →  `npm run prompt-lab`
Quick 10-job eval for a single model. Writes a markdown report for easy comparison.
Designed for fast A/B testing of prompt changes — you get a result in ~5 minutes.
Requires `--model <name>`.

#### `finetune/eval_finetuned.py`
Evaluates the MLX fine-tuned model against any JSONL test file. Unlike `eval-runner.ts`
(which uses node-llama-cpp + GGUF), this loads the MLX model with its LoRA adapter
directly. Required fields: `title`, `location`, `jd_text`, `label`. Component scores
(`loc`, `role`, `tech`, `comp`) are optional — if absent, only label accuracy is reported.
```bash
source .venv/bin/activate
python finetune/eval_finetuned.py \
  --adapter finetune/adapters/0000200_adapters.safetensors \
  --test-file data/finetune/test.jsonl   # or any other CLEAN file
```

---

### Data Preparation Scripts

#### `src/cli/add-score-breakdowns-to-golden-jobs.ts`  →  `npm run golden:breakdowns`
Reads `golden_jobs.jsonl`, runs the deterministic scorer on every job, and writes
`golden_jobs_scored.jsonl` with `loc`, `role`, `tech`, `comp`, `computed_score`, and
`computed_label` added. Also prints a discrepancy report showing where the deterministic
scorer disagrees with the hand-labeled scores. Run this first in the fine-tuning pipeline.

#### `src/cli/split-scored-jobs-into-train-and-test.ts`  →  `npm run finetune:split`
Stratified train/test split of `golden_jobs_scored.jsonl`. Defaults: 70 train / 33 test,
seed=42. Writes `data/finetune/train.jsonl`, `data/finetune/test.jsonl`, and
`data/finetune/split_manifest.json`. **Always use the same seed** to keep the split
stable — changing it would shuffle jobs between train and test, invalidating past results.

#### `src/cli/format-finetune-training-data-for-mlx.ts`  →  `npm run finetune:format`
Converts `data/finetune/train.jsonl` into the MLX LoRA chat format
(`{messages: [system, user, assistant]}`), filling the v9 prompt template with each
job's fields and building the target JSON assistant response from the golden scores.
Writes to `data/finetune/mlx/train.jsonl`.

#### `src/cli/score-raw-jobs-to-golden-format.ts`
Takes raw JSONL job data (flexible field names: `job_id/id`, `title`, `company/company_name`,
`location`, `jd_text/description`, `source_url/url`) and auto-labels each job using the
deterministic scorer. Filters out jobs with empty or very short JD text. Writes a
complete golden-format file ready for eval.
```bash
npx tsx src/cli/score-raw-jobs-to-golden-format.ts \
  --input data/new_uk_jobs.jsonl \
  --output data/new_uk_jobs_golden.jsonl \
  --min-length 50
```

#### `src/cli/convert-salary-csv-to-golden-jobs.ts`
Converts a salary CSV (columns: `Job Title`, `Company`, `Location`, `Date`, `Salary`,
`Company Score`) to golden-format JSONL. Builds `jd_text` from the CSV fields, runs the
deterministic scorer to generate labels, and assigns sequential `ses-XXXXXX` job IDs.
```bash
npx tsx src/cli/convert-salary-csv-to-golden-jobs.ts \
  --input "data/Software Engineer Salaries.csv" \
  --output data/software_engineer_salaries_golden.jsonl
```

#### `src/cli/sample-jobs-from-export-for-labeling.ts`  →  `npm run golden:sample`
Samples raw jobs from a LinkedIn/scraper export (JSONL or JSON array) into a golden jobs
stub file. Each record gets `label: "maybe"` and a `TODO` reasoning placeholder — you
must manually review and correct `label`, `score`, and `reasoning` before using for
training. Use this as the starting point when adding new hand-labeled jobs to
`golden_jobs.jsonl`.

#### `src/cli/build-promptfoo-test-cases.ts`  →  `npm run promptfoo:tests`
Builds `data/promptfoo_tests.yaml` from `golden_jobs.jsonl`. This is the test case file
read by `eval-runner.ts`. Run after any changes to `golden_jobs.jsonl`.

---

### Golden Data Management

#### `src/cli/validate-golden.ts`  →  `npm run golden:validate`
Validates `golden_jobs.jsonl` against the schema: required fields present, labels are
valid, scores are in range. Use `npm run golden:validate:strict` to also enforce JD text
length limits (80–100 chars min/max).

#### `src/cli/generate-synthetic-golden-jobs.ts`  →  `npm run golden:generate-fake`
Generates synthetic job records using Faker.js and appends them to `golden_jobs.jsonl`.
Used to fill gaps in the dataset (e.g., more bad_fit or good_fit examples). Accepts
`--count`, `--seed`, `--output`.

---

### Reporting Scripts

#### `src/cli/compare-eval-run-results.ts`  →  `npm run compare`
Reads eval result JSON files from `results/` and prints a comparison table across runs.
Useful for tracking accuracy changes across prompt iterations.
- `--model <name>` — filter to a specific model
- `--last <n>` — limit to the N most recent runs

#### `src/cli/summarize-iteration-log.ts`  →  `npm run iterations:summary`
Reads `results/iteration_log.csv` (written by `run-tagged-eval-with-snapshot.ts`) and
prints a table of the last N tagged runs with their timestamps and result directories.

---

## Pipeline Diagram

```
data/golden_jobs.jsonl  (103 hand-labeled — MASTER)
        │
        ├─ npm run golden:breakdowns
        │        └─▶ data/golden_jobs_scored.jsonl  (+loc/role/tech/comp)
        │                   │
        │                   └─ npm run finetune:split  (seed=42)
        │                            ├─▶ data/finetune/train.jsonl  (70)  ⚠️ CONTAMINATED
        │                            ├─▶ data/finetune/test.jsonl   (33)  ✅ CLEAN
        │                            └─▶ data/finetune/split_manifest.json
        │
        │   data/finetune/train.jsonl
        │        │
        │        └─ npm run finetune:format
        │                 └─▶ data/finetune/mlx/train.jsonl  (93, oversampled)  ⚠️ CONTAMINATED
        │                              │
        │                              └─ mlx_lm.lora  (LoRA fine-tuning)
        │                                       └─▶ finetune/adapters/  (checkpoints)
        │
        └─ npm run promptfoo:tests
                 └─▶ data/promptfoo_tests.yaml  (used by eval-runner.ts)


New unseen data (scraped AFTER fine-tuning):
        │
        ├─ data/new_uk_jobs.jsonl  (205 raw)
        │        └─ score-raw-jobs-to-golden-format.ts
        │                 └─▶ data/new_uk_jobs_golden.jsonl  (72)  ✅ CLEAN
        │
        └─ data/Software Engineer Salaries.csv
                 └─ convert-salary-csv-to-golden-jobs.ts
                          └─▶ data/software_engineer_salaries_golden.jsonl  (868)  ✅ CLEAN
                                  (all bad_fit — limited eval value, sanity check only)
```

---

## Quick Reference: Which File to Use for Eval?

| Scenario | File to use |
|----------|-------------|
| Eval fine-tuned model (honest) | `data/finetune/test.jsonl` |
| Eval fine-tuned model (real-world UK jobs) | `data/new_uk_jobs_golden.jsonl` |
| Sanity check non-UK rejection | `data/software_engineer_salaries_golden.jsonl` |
| Eval baseline model (no fine-tuning) | `data/promptfoo_tests.yaml` via `npm run eval` |
| Prompt engineering iteration | `npm run prompt-lab -- --model <name>` (10 jobs) |
| Adding new unseen UK jobs for future eval | Scrape → `score-raw-jobs-to-golden-format.ts` |
| Adding new hand-labeled jobs | Scrape → `sample-jobs-from-export-for-labeling.ts` → manual label → append to `golden_jobs.jsonl` |

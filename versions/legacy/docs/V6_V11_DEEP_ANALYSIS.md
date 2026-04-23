# Deep Analysis: Student Model Training V6 → V11

**Date**: 2026-03-13
**Scope**: Comprehensive analysis of all training attempts from V6 through V11, identifying what worked, what failed, root causes, theoretical limits, and a recommended path to 90%+ label accuracy.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Results Timeline](#2-results-timeline)
3. [What Helped](#3-what-helped)
4. [What Didn't Help (or Actively Hurt)](#4-what-didnt-help-or-actively-hurt)
5. [Version-by-Version Breakdown](#5-version-by-version-breakdown)
6. [Data Pipeline Evolution](#6-data-pipeline-evolution)
7. [Prompt Evolution](#7-prompt-evolution)
8. [The Label Quality Problem](#8-the-label-quality-problem)
9. [V7 Data Quality Audit](#9-v7-data-quality-audit)
10. [The Tech Field: Why It's Uniquely Hard](#10-the-tech-field-why-its-uniquely-hard)
11. [Training Data Distribution Analysis](#11-training-data-distribution-analysis)
12. [Label Accuracy vs Model Accuracy Relationship](#12-label-accuracy-vs-model-accuracy-relationship)
13. [Critique of This Analysis](#13-critique-of-this-analysis) **(16 items, 4 resolved)**
14. [Key Lessons](#14-key-lessons)
15. [Recommended Training Plan for V12](#15-recommended-training-plan-for-v12)
16. [Deterministic Baseline & Hybrid Approach](#16-deterministic-baseline--hybrid-approach) **(MEASURED: 87.0%)**
17. [Appendix: File Reference](#17-appendix-file-reference)

---

## 1. Executive Summary

### The Central Finding

**The model learned the easy parts of the data but failed to learn the hard parts — and for the hard parts, deterministic rules are better.**

V7 with 642 training examples achieved **75.3% on all 239 test jobs** (0.5B) or **80.8%** (1.5B) — the best model results across 8+ experiments. The headline "84.9%" number excludes 27 failed predictions (15 parse failures + 12 invalid tokens) from the denominator. A deterministic regex baseline achieves 73.6% without any training, beating the model on tech (81.6% vs 70.3%) and comp (90.4% vs 71.7%). The model's primary value-add is seniority detection (+60pp over regex). The measured hybrid approach (model for loc/arr/sen + regex for tech/comp) achieves **87.0%** (208/239) — a new high with no retraining.

The model faithfully reproduced its training data including errors (data accuracy ~90-93%, model accuracy ~75-81% on all jobs, gap ~10-15pp explained by learning imperfection, parse failures, and invalid tokens). But for inherently deterministic tasks like salary parsing and keyword matching, the model adds noise rather than insight. The V8→V11 journey was trying to solve a data quality problem with training strategy changes — and simultaneously trying to make the model learn tasks that regex does better.

### Key Numbers

- **MEASURED hybrid (model + regex)**: **87.0%** (208/239), 95% CI [82.2%, 90.7%] — best result, no retraining needed
- **Best model (all jobs)**: V7 1.5B = **80.8%** (193/239); V7 0.5B = **75.3%** (180/239)
- **Best model (valid only)**: V7 0.5B = 84.9% (180/212) — but excludes 27 jobs (15 parse failures + 12 invalid tokens)
- **Deterministic regex baseline**: 73.6% (176/239) — beats model on tech and comp; ties V8 on all-jobs
- **V7 vs V8 all-jobs**: 75.3% vs 73.6% = **1.7pp gap** (not 10pp — the "valid only" framing inflated the `_raw` advantage)
- **Estimated V7 data label accuracy**: ~90-93% (not "clean" — ~25-30% of examples have some bias or error)
- **Gap (model vs data)**: ~6pp on valid predictions, ~16pp on all jobs (the extra 10pp = parse failures + invalid tokens)
- **Worst regression**: V10 = 45.5% valid / 40.6% all-jobs (RESOLVED: undertrained at 600 iters vs V7's 2000)
- **V11c tokens on V7 test set**: 60.6% valid / 59.8% all-jobs (not 71% — different eval set inflated the earlier number)
- **Model size effect**: 0.5B vs 1.5B = +5.5pp on all-jobs (75.3% vs 80.8%) — mostly from fewer parse failures (15 vs 0)
- **Hybrid errors**: 31 total (239−208). 18 jobs wrong for both standalone approaches, but field mixing creates ~13 additional errors

### Root Causes of the Plateau

1. **V7 training data is ~90-93% correct, not ~99%** — ~50 wrong tech labels (AI_ML), ~35 wrong comp labels (fuzzy corrections), 212 biased generated jobs (99.5% NODE)
2. **Systematic label noise** — AI_ML has 52.5% teacher self-consistency on edge cases; this is the dangerous kind of noise (model can't generalize past it)
3. **Tech is a multi-label array** — exact-match evaluation compounds per-token errors exponentially
4. **Co-occurrence bias** — 83.3% of REACT co-occurs with JS_TS, creating false correlations
5. **Pipeline bugs masked as data problems** — V9's 50-char truncation bug, not data quality, caused most of its regression
6. **Output format is load-bearing** — `_raw` fields provide critical chain-of-thought for the 0.5B model

### Path to 90%+

**Path 1 — Hybrid (no retraining, MEASURED)**: Use V7 model for loc/arr/sen + regex for tech/comp → **87.0% (208/239)**, 95% CI [82.2%, 90.7%]. This is the lowest-effort, highest-confidence path. Already built: `finetune/compute_hybrid.py`.

**Path 2 — Hybrid + label fixes (with retraining)**:
1. Start from hybrid approach (already at 87.0%)
2. Fix ~16 wrong comp fuzzy corrections + ~15 worst AI_ML errors in training data
3. Retrain model with `_raw` fields (for chain-of-thought benefit)
4. Override tech/comp with regex at inference time → target **90%+**
5. Add ~100 contrastive examples for model's weak spots (seniority edge cases, loc edge cases) → target **92%+**

**Path 3 — Push for 95%** (requires test set audit):
6. Audit 50-100 test set labels (especially score 50-74 zone) — current "accuracy" may undercount true accuracy
7. Replace 212 NODE-biased generated jobs with diverse examples
8. After audit, recalibrate the target: if test set is ~91% correct, a model at 87% is actually ~96% against ground truth

**Important caveat**: The hybrid mixes *fields* (model loc/arr/sen + regex tech/comp), not final labels. This means it can break "both correct" jobs by combining wrong fields into a wrong score. The hybrid's measured 31 errors include ~13 new errors that neither standalone approach produces. Improving further requires fixing these field-mixing conflicts — primarily model loc errors (9 REMOTE→UNK) and regex tech errors (NODE misses).

---

## 2. Results Timeline

### All Evaluated Checkpoints

| Version | Model | Train Size | Best Ckpt | Label Acc† | All-Jobs Acc‡ | Parse Fail | Inv. Tok | Valid | Eval Set | Key Change |
|---------|-------|-----------|-----------|-----------|--------------|-----------|---------|-------|----------|------------|
| V7 | 0.5B | 642 | 2000 | **84.9%** | **75.3%** | 15 | 12 | 212/239 | 239-job test | Baseline, 10-field w/ _raw |
| V7 | 1.5B | 642 | 2000 | **85.4%** | **80.8%** | 0 | 13 | 226/239 | 239-job test | 3x model size |
| V8 | 0.5B | 641 | 1850 | 74.9% | 73.6% | 3 | 1 | 235/239 | 239-job test | Removed _raw fields |
| V9 | 0.5B | 1,010 | 1800 | 68.3% | 51.5% | 48 | 11 | 180/239 | 239-job test | +57% data, 50-char cap bug |
| V9 | 0.5B | 1,010 | 2000 | 62.9% | 44.8% | 63 | 6 | 170/239 | 239-job test | Later checkpoint, worse |
| V9 | 0.5B | 1,010 | 1225 | 52.7% | 49.4% | 5 | 10 | 224/239 | 239-job test | Parse fix confirmed but undertrained |
| V10 | 0.5B | 927 | 600 | 45.5% | 40.6% | 10 | 16 | 213/239 | 239-job test | New prompt format |
| V10 | 0.5B | 927 | 500 | 38.1% | 33.5% | 13 | 16 | 210/239 | 239-job test | Earlier checkpoint |
| V10 | 0.5B | 927 | 300 | 33.1% | 22.6% | 5 | 71 | 163/239 | 239-job test | Earliest checkpoint |
| V10 clean_raw | 0.5B | 927 | 300 | 37.1% | 27.6% | 3 | 58 | 178/239 | 239-job test | Cleaned _raw fields |
| V10 essential | 0.5B | 927 | 300 | 44.5% | 30.5% | 7 | 68 | 164/239 | 239-job test | Only loc/tech/comp _raw |
| V10 tokens | 0.5B | 641 | 300 | 25.7% | 25.5% | 0 | 2 | 237/239 | 239-job test | Token-only, V7 data size |
| V10 tokens | 0.5B | 641 | 600 | 49.6% | 49.0% | 0 | 3 | 236/239 | 239-job test | Token-only, more iters |
| V10 preprocessed | 0.5B | 927 | 600 | 38.8-43.5% | 33.9-44.8% | varies | varies | varies | 239-job test | Various noise reduction |
| V11 base | 0.5B | 900 | — | 0-65% | — | varies | varies | varies | 100-job valid | Full schema, massive parse failures |
| V11 tokens | 0.5B | 900 | 200 | 65-73% | — | varies | varies | varies | 100-job valid | Token-only, some parse issues |
| **V11c tokens** | **0.5B** | **900** | **600** | **71.0%** | **—** | **0** | **0** | **100/100** | **100-job valid** | **Best V11: teacher-filtered** |
| V11c tokens (V7 test) | 0.5B | 900 | 600 | **60.6%** | **59.8%** | 0 | 3 | 236/239 | **239-job test** | Same adapter, fair comparison to V7 |
| V11c tokens (hint) | 0.5B | 900 | 600 | 71.0% | 0 | 100/100 | 100-job valid | Hint helps tech (+19pp) but hurts other fields |
| V11c long | 0.5B | 900 | 500 | 64.2% | 5 | 95/100 | 100-job valid | More iterations, overfit |
| V11d tokens | 0.5B | 900 | 400 | 61.6% | 0 | 99/100 | 100-job valid | +261 synthetic, hurt accuracy |
| V11e tokens | 0.5B | 810 | 200 | 43.8% | 1 | 89/100 | 100-job valid | Aggressive dist caps, catastrophic |
| **Hybrid (V7+regex)** | **—** | **—** | **—** | **87.0%** | **0** | **239/239** | **239-job test** | **Model loc/arr/sen + regex tech/comp** |

### Per-Field Accuracy (Best Checkpoints)

| Version | loc | arr | sen | tech | comp | Label |
|---------|-----|-----|-----|------|------|-------|
| V7 0.5B | 95.8% | 78.3% | 89.6% | 70.3% | 71.7% | 84.9% |
| V7 1.5B | 97.3% | 91.2% | 92.0% | 70.4% | 77.9% | 85.4% |
| V8 0.5B | 91.5% | 79.1% | 84.7% | 60.4% | 54.9% | 74.9% |
| V9 best | 88.9% | 78.3% | 66.7% | 44.4% | 50.6% | 68.3% |
| V10 best | — | — | — | — | — | 45.5% |
| V11c tokens (100-job) | 60.0% | 67.0% | 61.0% | 41.0% | 48.0% | 71.0% |
| **V11c tokens (239-job)** | **60.6%** | **63.1%** | **59.7%** | **42.4%** | **40.3%** | **60.6%** |
| V11c tokens+hint (100-job) | 59.0% | 77.0% | 52.0% | 60.0% | 41.0% | 71.0% |
| **Deterministic baseline** | **94.1%** | **77.4%** | **29.3%** | **81.6%** | **90.4%** | **73.6%** |

*† Label Acc = accuracy on valid predictions only (denominator = Valid column). Parse failures and invalid-token predictions are excluded.*
*‡ All-Jobs Acc = correct / 239 (parse failures + invalid tokens counted as wrong). This is the production-relevant number.*

**Denominator methodology**: The eval script (`eval_student_v7.py`) excludes two categories from the accuracy denominator: (1) **parse failures** — model output is not valid JSON, and (2) **invalid tokens** — JSON is valid but contains token values outside the vocabulary (e.g. `LONDNO` instead of `IN_LONDON`), even after fuzzy correction (edit distance ≤ 2). Both are real model failures. The "Label Acc†" column reports accuracy on the remaining valid predictions; the "All-Jobs Acc‡" column counts all excluded predictions as wrong. **For production use, All-Jobs Acc is the honest number.** For model-vs-model comparison, Label Acc is consistent as long as all models use the same eval script. Per-field accuracies in the table below are computed on valid predictions only.

**IMPORTANT CAVEAT**: V7/V8/V9/V10 were evaluated on the 239-job V7 test set. V11 variants initially on a 100-job validation set. V11c was re-evaluated on the 239-job test set: **60.6%** (not 71%), confirming a ~24pp format penalty from removing `_raw` fields. The deterministic baseline uses regex/keyword matching only — no model inference. See Sections 12 and 16 for details.

---

## 3. What Helped

### 3.1 `_raw` fields as chain-of-thought scaffolding (+10pp valid-only, +1.7pp all-jobs)

V7 (with `_raw`) = 84.9% on valid vs V8 (without `_raw`) = 74.9% on valid. On all 239 jobs: V7 = 75.3% vs V8 = 73.6% — the gap narrows from 10pp to **1.7pp** because V7's advantage is partly offset by its higher parse failure rate (15 vs 3).

The `_raw` fields force the 0.5B model to extract evidence before classifying. The assistant response goes from ~90 chars (5 tokens) to ~430 chars (10 fields with verbatim text). This provides:
- Chain-of-thought reasoning in the output
- ~5x more gradient signal per example (with `mask_prompt=true`, loss is only on assistant response)
- Forced attention to specific parts of the JD

**Quantitative impact on gradient signal**: With 642 examples:
- Token-only: 642 × ~90 chars = ~57K chars of training signal
- With `_raw`: 642 × ~430 chars = ~275K chars of training signal

The 0.5B model with LoRA rank 16 has ~4M trainable parameters. Token-only format may not provide enough gradient diversity.

### 3.2 gpt-4.1-mini over gpt-4o-mini (10x fewer fuzzy corrections)

| Model | Fuzzy Correction Rate | Token Failure Rate |
|-------|----------------------|-------------------|
| gpt-4o-mini | 10.3% (196/1904) | 0.4% |
| gpt-4.1-mini | 0.3% (5/1798) | 0.1% |
| gpt-5-mini | 0.0% (0/200) | 0.0% |

V7 training data was labeled with gpt-4.1-mini, which is a key reason it's the cleanest dataset. gpt-4o-mini produces phantom tokens like `RANGE_35_44K` that get silently corrected to wrong values.

### 3.3 Teacher filtering (+6pp over unfiltered V9 data)

V11c teacher-filtered 811/1081 V9 jobs (kept only where V7 teacher model's predictions agreed with GPT labels) and got 71%. Raw V9 data got 68.3%. Filtering removes the noisiest labels.

However, teacher filtering has a circularity problem (see Section 12).

### 3.4 Sentence-level JD cleaning (0 parse failures)

V11 tokens variants achieved 0 parse failures by:
- Splitting JDs into sentences
- Dropping boilerplate (diversity statements, cookie notices, apply-now, etc.)
- Keeping only fact-bearing sentences (salary, tech, location, seniority keywords)
- Capping at 28 sentences

For a 0.5B model with 8K context, reducing input noise is beneficial for output format compliance.

### 3.5 Consistent preprocessing at train AND inference time

When preprocessing is applied at training time but not inference time (or vice versa), it creates a distribution shift. The model sees different input formats at train vs inference time. V11's approach of using the same `clean_text()` function at both stages avoided this trap.

---

## 4. What Didn't Help (or Actively Hurt)

### 4.1 More data without quality control (−16.6pp)

V7: 642 → V9: 1,010 (+57% more data) → accuracy dropped from 84.9% to 68.3%.

The extra data introduced:
- 50-char truncation bug (caused 48 parse failures — a pipeline bug, not a data quality issue)
- Distribution mismatch (good_fit 14.4% train vs 23.8% test)
- AI_ML labeling inconsistency between V7 and V9 teacher prompts (90 tech disagreements on overlapping jobs)

**Caveat**: The regression was partially caused by bugs, not just data quality. If the truncation bug were fixed and distribution matched, V9 might have performed closer to V7. We don't know because this was never tested.

### 4.2 Synthetic data (−9.4pp)

V11d added 261 template-based synthetic JDs → 61.6% vs V11c's 71.0%.

Problems with the synthetic data:
- Very short (~250 chars vs real median of 1,600 chars) — model learned "short JD = good_fit"
- 100% tracked tech, 0% OOS — no scope-gate examples
- Skewed toward high-comp (ABOVE_100K, RANGE_75_99K)
- Template patterns were too uniform

**Confound**: V11d also used balanced label distribution (30/30/40), so the harm could be from either the synthetic data or the balancing or both.

### 4.3 Removing `_raw` fields (−10 to −40pp)

Every version that removed `_raw` fields underperformed:
- V8 (no `_raw`): 74.9% vs V7 (with `_raw`): 84.9% → −10pp
- V10 (no `_raw`): 45.5% vs V7: 84.9% → −39.4pp
- V10 tokens: 49.6% vs V7: 84.9% → −35.3pp
- V11 tokens_c: 71.0% on different eval set (not directly comparable but pattern holds)

### 4.4 Bigger model (negligible accuracy gain)

On valid predictions: V7 0.5B: 84.9%, V7 1.5B: 85.4% → +0.5pp.
On all 239 jobs: V7 0.5B: **75.3%**, V7 1.5B: **80.8%** → **+5.5pp**.

The valid-only framing hides the 1.5B's real advantage: zero parse failures (vs 15) and fewer invalid tokens (13 vs 12). The 1.5B also improved arr (+12.9pp) and comp (+6.2pp) on valid predictions. Tech accuracy was identical (70.4% vs 70.3%).

**Revised conclusion**: Model size matters more than the valid-only comparison suggests. The 1.5B gets 13 more jobs right on the full test set — a meaningful improvement. The bottleneck is still data/labels for tech accuracy specifically, but overall production accuracy benefits substantially from the larger model.

### 4.5 More training iterations (−6.8pp)

V11c at iter 600: 71.0% → V11c_long at iter 500: 64.2%. More training caused overfitting. The model's optimal checkpoint is often surprisingly early relative to the configured iteration count.

Historical pattern:
- V7: trained 2000 iters, peak unclear (only final checkpoint evaluated)
- V5.1: peaked at iter 875 (training crashed at 890 via OOM)
- V9: best at iter 1800, worse at iter 2000

### 4.6 Balanced label distribution (neutral to negative)

V11d (30/30/40 balanced): 61.6% vs V11c (19/23/58 natural): 71.0%.

The natural distribution of V7 (24/29/47) outperformed forced balancing. This suggests the model needs to learn realistic base rates, not artificial ones.

### 4.7 Aggressive distribution caps (−27.2pp)

V11e capped NO_GBP at 15% and OUTSIDE_UK at 6% → 43.8%.

Removing "easy" negative examples destroyed the model's ability to learn base rates. Without enough bad_fit anchors, the model lost calibration entirely. "Easy" examples aren't noise — they're the foundation the model builds its decision boundaries on.

### 4.8 Hint injection at eval time (mixed results)

V11c tokens with hint: 71.0% (same overall as without hint), but field-level changes:
- tech: 41% → 60% (+19pp improvement)
- arr: 67% → 77% (+10pp)
- loc: 60% → 59% (−1pp)
- sen: 61% → 52% (−9pp)
- comp: 48% → 41% (−7pp)

Hints help the hardest field (tech) but confuse the model on easier fields where its own judgment was better.

### 4.9 Postprocessing at eval time (marginal)

Available corrections: currency → NO_GBP, non-UK → OUTSIDE_UK, AI term detection → add AI_ML. Results showed marginal improvement at best — postprocessing can't fix core model errors, only catch obvious misses.

---

## 5. Version-by-Version Breakdown

### V7 (Baseline — Best Result)

**Architecture**: 5 fields, 10 JSON keys (5 tokens + 5 `_raw` suffixes)
**Data**: 642 train / 71 valid, labeled with gpt-4.1-mini, teacher prompt v7
**Training**: 0.5B and 1.5B, rank 16, LR 2e-5, warmup 100, 2000 iters
**Format**: Smart truncation (protects first 300 + last 200 words, protects salary keyword windows)
**Result**: 84.9% (0.5B), 85.4% (1.5B)

**Why it worked**: Clean labels (0.3% fuzzy rate), `_raw` chain-of-thought, good diversity (real + some generated data), teacher and test set share same prompt/model.

### V8 (Output Simplification — Regression)

**Change**: Removed all `_raw` fields from assistant output (5 keys instead of 10)
**Data**: Same as V7 (641 train / 72 valid)
**Result**: 74.9% (−10pp from V7)

**Why it failed**: Removing `_raw` fields eliminated chain-of-thought AND reduced gradient signal by ~5x. The 0.5B model needs to "show its work" to classify correctly.

### V9 (More Data + Bugs — Major Regression)

**Changes**:
- Expanded to 1,010 train / 71 valid (6 batches merged from V7 + new labeling)
- 50-char hard cap on all `_raw` fields
- Synthetic jobs added (train-only)
- Restored `_raw` fields in output

**Result**: 68.3% best (iter 1800), 62.9% (iter 2000), 52.7% (iter 1225 with parse fix)

**Why it failed** (multiple compounding causes):
1. **50-char truncation bug** (Finding 41): `tech_raw` truncated mid-word → model dropped opening `"` before `tech` key → 48-63 parse failures per eval. 444 of 1010 training examples affected.
2. **Distribution mismatch** (Finding 42): good_fit 14.4% train vs 23.8% test (−9.4pp gap)
3. **AI_ML prompt drift**: V9 teacher prompt stricter on AI_ML → 90 tech disagreements with V7 labels on overlapping jobs → systematic under-prediction on V7-labeled test set
4. **Pipeline bug** (Finding 39): `job.location` vs `job_location` field name mismatch caused 83% UNK in early labeling runs

### V10 (Multiple Variants — Undertrained, Not Broken)

**Changes**:
- New prompt format with "Allowed tokens" + "Rules" section
- `filterTechRaw()` replaced 50-char cap (fixed V9's bug)
- 4 raw-mode options: all, none, essential, mixed
- Data built via `build-v10-training-set.ts` with distribution balancing

**Results**: 33.1-45.5% across all variants

**Root cause: RESOLVED — undertrained.** Investigation of training logs revealed:
- V10 was evaluated at checkpoints 300-600 only (~10 epochs for 927 jobs)
- V7 trained for 2000 iters (~50 epochs for 642 jobs) — 5x more training
- V10's accuracy was **monotonically increasing** at checkpoint 600 — it never plateaued
- The training was stopped early due to resource constraints, not convergence
- The "catastrophic 40pp regression" was simply a model that hadn't finished learning

**Evidence**: V10 checkpoint progression shows steady improvement:
- iter 300: 33.1% → iter 500: 38.1% → iter 600: 45.5% (upward trajectory)
- V10 tokens: iter 300: 25.7% → iter 600: 49.6% (same pattern)
- At this rate, V10 at 2000 iters might have reached 70-80%+

**V10 tokens variant** (V7 data size, token-only output): 49.6% at iter 600. Also undertrained — trajectory was still climbing.

**V10 preprocessed** (eval-time noise reduction): 38.8-43.5%. Three strategies tested on the undertrained model — results are not meaningful.

**Lesson**: Never compare models at different training stages. V10 vs V7 was a 10-epoch model vs a 50-epoch model — not a fair comparison of architectures or data.

### V11 (Teacher Filtering + Sentence Cleaning — Partial Recovery)

**Changes**:
- Teacher filtering: only V9 jobs where V7 student model predictions agree (811/1081 pass)
- Sentence-level JD cleaning (drop boilerplate, keep fact-bearing sentences, cap at 28)
- Rebuilt `_raw` fields from cleaned sentences
- Dual output format: full 10-field (`mlx/`) and token-only 5-field (`mlx_tokens/`)
- 5 data variants tested (v11, v11b, v11c, v11d, v11e)

**Results**:

| Variant | Train | Differentiator | Label Acc |
|---------|-------|---------------|-----------|
| v11 base (full schema) | 900 | 10-field output | 0-65% (massive parse failures) |
| v11 tokens | 900 | Token-only, natural dist | 65% (0 parse fails on full 100) |
| **v11c tokens** | **900** | **Different random selection** | **71% (best V11)** |
| v11d tokens | 900 | +261 synthetic, balanced | 61.6% |
| v11e tokens | 810 | Aggressive NO_GBP/OUTSIDE_UK caps | 43.8% |
| v11c tokens long | 900 | More iterations (800 vs 600) | 64.2% |

**Key V11 findings**:
- Full 10-field output caused massive parse failures (65-100 out of 100 jobs) — the sentence-cleaned `_raw` fields were apparently harder to generate than V7's verbatim `_raw` fields
- Token-only output achieved 0 parse failures but lower accuracy
- Teacher filtering provided moderate improvement over raw V9 data
- Hint injection helped tech (+19pp) but hurt other fields
- All V11 training runs stopped before reaching target iterations (resource constraints / OOM)

---

## 6. Data Pipeline Evolution

### Format Evolution

| Version | Output Fields | _raw Handling | JD Preprocessing |
|---------|--------------|---------------|-----------------|
| V7 | 10 (5 token + 5 _raw) | Unlimited length | Smart truncation (protect salary windows) |
| V8 | 5 (token only) | N/A (removed) | Same as V7 |
| V9 | 10 (5 token + 5 _raw) | 50-char hard cap (BUGGY) | Same as V7 + synthetic split |
| V10 | 10 (configurable) | `filterTechRaw()` (label-guided) | Same as V7 + distribution balancing |
| V11 | 10 or 5 (dual output) | Sentence-extracted | Sentence-level cleaning (28-sentence cap) |

### V7 Smart Truncation (format-for-mlx-v7.ts)
- Protects first 300 words and last 200 words
- Protects 100-word windows around salary keywords (£, $, salary, compensation, per annum)
- Removes largest contiguous unprotected region to fit maxTokens (default 7500)
- Inserts `[...]` marker at truncation point

### V9 50-Char Cap Bug (format-for-mlx-v9.ts)
```typescript
function cap50(s: string | null | undefined): string | null {
  if (!s) return null;
  return s.length > 50 ? s.slice(0, 50) : s;
}
```
Applied to ALL `_raw` fields. Truncated mid-word without checking token boundaries. Example: `"Python, React.js, TypeScript, PostgreSQL, Redis, Docker"` → `"Python, React.js, TypeScript, PosgreSQL, R"` (corrupted). 444 of 1010 training examples affected.

### V10 filterTechRaw (src/lib/filter-tech-raw.ts)
Label-guided extraction: only extracts tech terms matching labeled tokens. Case-insensitive, word-boundary-anchored. Returns comma-joined terms in reading order. Returns null for OOS. No hard character cap. Example: `tech_raw="Python, React.js, TypeScript, PostgreSQL"` with `tech=["REACT","JS_TS"]` → `"React.js, TypeScript"`.

### V11 Sentence-Level Cleaning (v11_preproc.py)
- Split JD into sentences
- Drop: diversity/inclusion, privacy policy, cookies, terms, how to apply, benefits, about-company
- Keep: salary/compensation, location, arrangement, tech, seniority, non-target tech
- Cap at 28 fact-bearing sentences
- Rebuild `_raw` fields from first matching sentence per field

### Data Volume by Version

| Version | Train | Valid | Total | Sources |
|---------|-------|-------|-------|---------|
| V7 | 642 | 71 | 713 | Real + generated, gpt-4.1-mini labels |
| V8 | 641 | 72 | 713 | Same as V7 |
| V9 | 1,010 | 71 | 1,081 | V7 + 6 new batches |
| V10 | 927 | 73 | 1,000 | Curated from V9 pool |
| V11 | 900 | 100 | 1,000 | Teacher-filtered V9 + V7 |
| V11b | 540 | 60 | 600 | Smaller, balanced labels |
| V11c | 900 | 100 | 1,000 | Different random selection |
| V11d | 900 | 100 | 1,000 | +261 synthetic, balanced |
| V11e | 810 | 90 | 900 | Aggressive NO_GBP/OUTSIDE_UK caps |

---

## 7. Prompt Evolution

### Student Prompts

| Version | Lines | Output Keys | Key Features |
|---------|-------|------------|--------------|
| student_v7.txt | 14 | 10 (with _raw) | Minimal: field definitions + template |
| student_v8.txt | 14 | 5 (token only) | Same as V7 but no _raw in template |
| student_v9.txt | 14 | 10 (with _raw) | Added "strict JSON" instruction, 50-char _raw cap |
| student_v10.txt | 21 | 5 (token only) | "Allowed tokens" section, explicit rules |
| student_v10_tokens.txt | 20 | 5 (token only) | Numbered rules, stronger token formatting |
| student_v11.txt | 13 | 10 (with _raw, missing sen_raw) | "Respond with JSON only" |
| student_v11_tokens.txt | 19 | 5 (token only) | Most structured: allowed tokens + 4 rules |

**Format ping-pong pattern**: V7(10) → V8(5) → V9(10) → V10(5) → V11(10, buggy) → V11_tokens(5). Never settled on one format.

### Teacher Prompts

| Version | Lines | Key Changes |
|---------|-------|------------|
| teacher_v7.txt | ~115 | Baseline: 5 sections, 3 examples, semantic rules |
| teacher_v8.txt | ~120 | Same as V7 but output format changed (no _raw) |
| teacher_v9.txt | ~120 | Added AI_ML coding tool exclusions (Copilot, Cursor, etc.), 50-char _raw caps, TC/OTE comp rules |
| teacher_tech_v9.txt | ~110 | Specialized tech-only extractor with 4-step approach, 10 examples, strongest AI_ML validation |

**Critical finding**: Teacher prompt changes between V7 and V9 caused 90+ tech disagreements on overlapping jobs, primarily on AI_ML. The test set uses V7 teacher labels, so V9-trained models are penalized for following V9's stricter AI_ML rules.

---

## 8. The Label Quality Problem

### 8.1 GPT Model Comparison

| Model | Fuzzy Rate | Failure Rate | Notes |
|-------|-----------|-------------|-------|
| gpt-4o-mini | 10.3% | 0.4% | Phantom tokens (RANGE_35_44K), high correction rate |
| gpt-4.1-mini | 0.3% | 0.1% | Much cleaner, V7 training data source |
| gpt-5-mini | 0.0% | 0.0% | Perfect adherence (tested on 200 jobs) |

### 8.2 Phantom Token Problem (Silent Comp Errors)

gpt-4o-mini generates non-existent comp tokens that get "corrected" by edit-distance fuzzy matching:

| Phantom Token | Fuzzy Match → | Correct Token | Count (V7+V9) |
|---------------|--------------|---------------|---------------|
| RANGE_35_44K | RANGE_45_54K (score 0) | BELOW_45K (score −30) | 45+ |
| RANGE_25_44K | RANGE_45_54K (score 0) | BELOW_45K (score −30) | 18+ |
| RANGE_25_34K | RANGE_45_54K (score 0) | BELOW_45K (score −30) | 5+ |

These are **semantically wrong corrections**. A job paying £25k-£34k should be BELOW_45K (score −30), not RANGE_45_54K (score 0). ~50 jobs are silently assigned wrong comp scores, potentially flipping labels.

### 8.3 AI_ML Inconsistency Across Versions

On 643 overlapping jobs between V7 and V9 training sets:
- **Tech agreement**: 81.0% (521/643)
- **90 of ~100 tech disagreements are AI_ML** — V7 teacher labeled it present, V9 teacher dropped it
- **Label agreement**: 92.7% (596/643) — many AI_ML disagreements don't flip labels

GPT-4.1-mini self-consistency on AI_ML edge cases: **52.5%** (Finding 40). This means relabeling the same job twice gives different AI_ML results about half the time for borderline cases.

### 8.4 Teacher Self-Consistency (V7 vs V8 Prompt)

Re-labeling 239 test jobs with V8 prompt (same model, gpt-4.1-mini):

| Field | Agreement | Notes |
|-------|-----------|-------|
| loc | 99.6% | Essentially perfect |
| sen | 98.3% | Very stable |
| comp | 96.7% | Stable |
| arr | ~97% | Stable |
| **tech** | **90.0%** | **24 disagreements, 10 are AI_ML drops** |

### 8.5 Empty Tech Array Default

172 corrections in V7 and 74 in V9 where GPT returned an empty tech array `[]` that was silently defaulted to `["OOS"]`. The teacher prompt says "never leave tech empty, use OOS" but GPT ignores this instruction ~10% of the time. Additionally, 25+ cases where GPT mixed OOS with real tokens (e.g., `["NODE", "OOS"]`) — the validation code strips OOS from mixed arrays.

### 8.6 Generated Data Bias

212 generated jobs in V7 training data (29.7% of total):
- **99.5% contain NODE** (211/212) vs ~41% in real data
- **0% are OOS** (0/212) vs ~24% in real data
- **50.9% contain AI_ML** (108/212) vs ~40% in real data
- **71.2% are REMOTE** (151/212) vs ~12% in real data
- **0% are UK_OTHER** vs ~32% in real data

This extreme bias teaches the model: "when uncertain, predict NODE + REMOTE."

### 8.7 Test Set Contamination

17 V9 training jobs appear in the V7 test set (239 jobs). ~7% contamination. Not catastrophic but inflates V9 eval scores slightly.

---

## 9. V7 Data Quality Audit

**This is the most important section of the analysis.** V7 was repeatedly described as having "clean" data. An actual audit reveals it is merely the "least dirty" — and the dirtiness directly explains the accuracy ceiling (84.9% on valid predictions, 75.3% on all jobs).

### 9.1 Overview

- **Total training jobs**: 713 (642 train + 71 valid)
- **Zero invalid tokens, zero OOS-mixed-with-real violations, zero empty tech arrays** — format is clean
- **But format compliance ≠ label correctness** — a perfectly formatted wrong label is still wrong

### 9.2 Generated Data (The Biggest Problem)

212 of 713 training jobs (29.7%) are generated. Their distribution is extreme:

| Metric | Generated (212) | Real (501) | Gap |
|--------|----------------|------------|-----|
| Contains NODE | **99.5%** (211/212) | ~41% | +58pp |
| Contains OOS | **0.0%** (0/212) | ~24% | −24pp |
| Contains AI_ML | 50.9% (108/212) | ~40% | +11pp |
| UK_OTHER location | 0.0% (0/212) | ~32% | −32pp |
| REMOTE location | 71.2% (151/212) | ~12% | +59pp |
| good_fit label | 45.8% (97/212) | 15.6% | +30pp |

The generated data is overwhelmingly NODE + REMOTE + good_fit. This teaches the model three false priors:
1. "When uncertain about tech → predict NODE"
2. "When uncertain about location → predict REMOTE"
3. "Short/simple JDs → good_fit" (generated JDs are shorter than real ones)

### 9.3 Tech Token Distribution

16 distinct tech combinations in training:

| Tech Combo | Count | % |
|-----------|-------|---|
| `["OOS"]` | 115 | 16.1% |
| `["AI_ML"]` | 82 | 11.5% |
| `["NODE", "JS_TS"]` | 75 | 10.5% |
| `["AI_ML", "NODE"]` | 65 | 9.1% |
| `["NODE"]` | 60 | 8.4% |
| `["AI_ML", "JS_TS", "NODE"]` | 56 | 7.9% |
| `["JS_TS", "NODE", "REACT"]` | 52 | 7.3% |
| `["JS_TS", "REACT"]` | 43 | 6.0% |
| `["AI_ML", "JS_TS", "NODE", "REACT"]` | 41 | 5.8% |
| `["JS_TS"]` | 40 | 5.6% |
| `["AI_ML", "JS_TS", "REACT"]` | 28 | 3.9% |
| `["AI_ML", "JS_TS"]` | 22 | 3.1% |
| `["REACT"]` | 11 | 1.5% |
| `["AI_ML", "NODE", "REACT"]` | 10 | 1.4% |
| `["AI_ML", "REACT"]` | 7 | 1.0% |
| `["NODE", "REACT"]` | 6 | 0.8% |

**Jobs with AI_ML: 311 (43.6%)** — nearly half of all training data includes AI_ML. Combined with 52.5% teacher self-consistency on AI_ML edge cases, this means ~40-50 AI_ML assignments may be wrong.

### 9.4 AI_ML Label Audit

Of 311 jobs with AI_ML in tech:
- **306 (98.4%)** have clear AI/ML keywords in tech_raw (machine learning, AI, LLM, NLP, deep learning, etc.)
- **3-4 (1.2%)** have NO obvious AI/ML keywords — AI_ML appears inferred from context
- **~40-50 (13-16%)** are edge cases where GPT would give different answers on re-labeling (52.5% self-consistency on borderline cases)

### 9.5 Comp Label Audit

| Comp Token | Count | % |
|-----------|-------|---|
| NO_GBP | 229 | 32.1% |
| RANGE_75_99K | 159 | 22.3% |
| RANGE_55_74K | 122 | 17.1% |
| ABOVE_100K | 66 | 9.3% |
| RANGE_45_54K | 48 | 6.7% |
| UP_TO_ONLY | 45 | 6.3% |
| BELOW_45K | 44 | 6.2% |

**Fuzzy correction damage**: Across labeling runs, 16 comp tokens were silently corrected by edit-distance matching:

| GPT Phantom Token | Mapped To | Should Be | Count | Score Error |
|-------------------|-----------|-----------|-------|-------------|
| RANGE_35_44K | RANGE_45_54K (0) | BELOW_45K (−30) | 13 | **+30 pts** |
| RANGE_25_34K | RANGE_45_54K (0) | BELOW_45K (−30) | 2 | **+30 pts** |
| RANGE_25_44K | RANGE_45_54K (0) | BELOW_45K (−30) | 1 | **+30 pts** |
| RANGE_65_74K | RANGE_55_74K (+5) | debatable | 2 | ±5 pts |
| RANGE_45_54K | RANGE_55_74K (+5) | debatable | 1 | +5 pts |
| RANGE_45_74K | RANGE_45_54K (0) | debatable | 1 | varies |

The first 3 rows (16 jobs) have a **+30 point score error** — these jobs are scored 30 points higher than they should be, potentially pushing bad_fit → maybe or maybe → good_fit.

### 9.6 Boundary Zone Analysis

**252 of 713 jobs (35.3%)** fall in the fragile score 50-74 zone:

| Score | Count | Label | Risk |
|-------|-------|-------|------|
| 50 | 48 | maybe | One −5pt error → bad_fit |
| 55 | 45 | maybe | Moderate risk |
| 60 | 50 | maybe | Moderate risk |
| 65 | 61 | maybe | One +5pt error → good_fit |
| 70 | 48 | good_fit | One −5pt error → maybe |

The 48 jobs at score 70 (barely good_fit) and 61 jobs at score 65 (one tech error from false good_fit) are the most fragile. A single AI_ML hallucination (+10 pts) at score 65 pushes to 75 → false good_fit.

### 9.7 Labeling Run Logs — Fuzzy Correction Summary

204 total fuzzy corrections across all labeling runs:

| Correction Type | Count |
|----------------|-------|
| Empty tech array `[]` → `["OOS"]` | 172 |
| Invalid comp token → nearest valid | 20 |
| OOS mixed with real tokens → OOS removed | 11 |
| Invalid tech element → nearest valid | 1 |

The 172 empty-array corrections indicate the teacher prompt wasn't explicit enough about always including `["OOS"]`. Later prompt fixes (after 2026-03-10T21:23) eliminated this — final runs show 0 fuzzy corrections.

### 9.8 Estimated Per-Field Label Accuracy

| Field | Estimated Accuracy | ~Wrong Labels | Evidence |
|-------|-------------------|---------------|----------|
| loc | ~99% | ~7 / 713 | GPT 99.6% self-consistent, clear textual signal |
| arr | ~97% | ~21 / 713 | "Flexible" ambiguity, but arr never affects score |
| sen | ~98% | ~14 / 713 | Title keywords usually clear |
| **tech** | **~93%** | **~50 / 713** | 3-4 clear errors + ~30 borderline AI_ML + ~15 co-occurrence |
| **comp** | **~95%** | **~35 / 713** | 16 fuzzy mismatches + ~19 boundary errors |

**Probability all 5 fields correct**: 0.99 × 0.97 × 0.98 × 0.93 × 0.95 ≈ **83%**

**Estimated label accuracy** (not all field errors flip labels): **~90-93%**

*Caveat: The 83% → 90-93% jump assumes ~78% of field errors don't flip labels (based on V7 1.5B diagnostics: 26/120 errors = 21.7% flip rate). But this flip rate was measured on the model's errors, not on the training data's errors. Comp errors from fuzzy correction are concentrated at score boundaries where they DO flip labels, so the training data's flip rate may be higher. The ~90-93% estimate could be as low as ~87% or as high as ~93%.*

### 9.9 Model Accuracy vs Data Accuracy

| Metric | Valid-Only | All 239 Jobs |
|--------|-----------|--------------|
| V7 data label accuracy (estimated) | ~90-93% | ~90-93% |
| V7 0.5B model label accuracy | 84.9% (180/212) | 75.3% (180/239) |
| V7 1.5B model label accuracy | 85.4% (193/226) | 80.8% (193/239) |
| Gap (0.5B) | ~6pp | ~16pp |
| Gap (1.5B) | ~6pp | ~10pp |

**The valid-only gap (~6pp) is explained by**: learning imperfection + generalization loss.
**The additional all-jobs gap (~10pp for 0.5B, ~4pp for 1.5B) is explained by**: parse failures (15 for 0.5B, 0 for 1.5B) and invalid tokens (12 for 0.5B, 13 for 1.5B). These are reliability failures, not accuracy failures.

**Conclusion**: The model is not failing to learn — it faithfully reproduces noisy data. The valid-only accuracy (~85%) is near the data quality ceiling (~91%). But in production, parse failures and invalid tokens drop the 0.5B to 75.3%. The 1.5B at 80.8% shows that model capacity reduces reliability failures. The hybrid at 87.0% sidesteps both problems by using regex fallback.

### 9.10 What "Clean Enough for 95%" Requires

| Target | Required Data Label Accuracy | Labels to Fix | Specific Actions |
|--------|------------------------------|---------------|------------------|
| 90% | ~95% data accuracy | ~31 labels | Fix 16 comp fuzzy corrections + 15 worst AI_ML |
| 93% | ~97% data accuracy | ~50 labels | Above + replace 212 biased generated jobs |
| 95% | ~98% data accuracy | ~70 labels + contrastive data | Above + audit 252 boundary-zone examples + 100 contrastive |

### 9.11 Cross-Version Data Label Accuracy vs Model Accuracy

The V7 audit above established that model accuracy tracks data accuracy. But V7 is only one version. This section applies the same analysis to every version — estimating data label accuracy (measured against the V7 test set standard) and comparing it to model accuracy to isolate what the gap is caused by.

#### The Comparison Table

| Version | Train | Data Label Acc (est.) | Model Acc (valid) | All-Jobs Acc | Gap (all-jobs) | Gap Explained By |
|---------|-------|-----------------------|-------------------|-------------|----------------|------------------|
| **V7** | 642 | **~91%** | **84.9%** | **75.3%** | 16pp | 6pp learning + 10pp parse/invalid (27 jobs) |
| **V7 1.5B** | 642 | **~91%** | **85.4%** | **80.8%** | 10pp | 6pp learning + 4pp invalid tokens (13 jobs) |
| **V8** | 641 | **~91%** | **74.9%** | **73.6%** | 17pp | 16pp no `_raw` + 1pp parse/invalid (4 jobs) |
| **V9** | 1,010 | **~85-88%** | **68.3%** | **51.5%** | 35-37pp | 48 parse + 11 invalid (bug) + distribution mismatch |
| **V10** | 927 | **~85-88%** | **45.5%** | **40.6%** | 45-47pp | **RESOLVED: Undertrained (600 iters vs V7's 2000)** |
| **V11c** | 900 | **~91-93%** | **60.6%** | **59.8%** | 31-33pp | No `_raw` fields (confirmed: 60.6% on 239-job V7 test set) |
| **V11d** | 900 | **~92-94%** | **61.6%** | — | — | Synthetic input distribution mismatch (100-job eval only) |
| **V11e** | 810 | **~90-92%** | **43.8%** | — | — | Base rate destruction (100-job eval only) |

#### V8: Same Data, Different Format → Data ~91%

V8 uses **the exact same 713 labeled jobs as V7**. Identical labels. The only change is the output format (5 tokens vs 10 fields with `_raw`). Data label accuracy is identical to V7 at ~91%.

The valid-only gap is 16pp (91% → 74.9%) and is entirely **format-driven**. On all 239 jobs, V8 scores 73.6% (same as regex), making the all-jobs gap 17pp. Removing `_raw` fields cut gradient signal by ~5x (430 chars → 90 chars per example with `mask_prompt=true`). However, V8 also has far fewer parse failures (3 vs 15) — so the `_raw` format costs reliability while gaining per-prediction accuracy.

#### V9: Different Teacher Prompt + Bugs → Data ~85-88%

V9 has 1,081 jobs labeled with the **V9 teacher prompt** (stricter AI_ML rules). The test set uses **V7 teacher prompt** labels. This creates a systematic mismatch.

Sources of label accuracy loss (vs V7 benchmark):
- V9 vs V7 AI_ML disagreement: 90 tech mismatches on 643 overlapping jobs (~7.3% tech disagreement)
- V9 vs V7 overall label agreement on overlap: 92.7% (596/643)
- 438 new V9 jobs labeled with V9 prompt: estimated same 7.3% label disagreement rate
- More fuzzy corrections than V7 (V9 used some gpt-4o-mini batches)
- Generated data NODE bias inherited from V7

**Estimated data label accuracy: ~85-88%** (vs V7 benchmark).

The 17-20pp gap (86% data → 68.3% model) breaks down as:
- ~3-6pp from worse data labels (V9 prompt drift causing AI_ML mismatch)
- ~12-15pp from 48 parse failures caused by 50-char truncation bug (engineering problem, not data)
- ~2-3pp from distribution mismatch (good_fit 14.4% train vs 23.8% test)

**Key realization**: If we remove parse failures, V9's accuracy on the 180 valid predictions is 68.3% × (239/180) ≈ **90.7% on valid predictions**. This is remarkably close to V7's 84.9% on 212 valid predictions. V9 may have learned its data nearly as well as V7 learned its data — the catastrophic-looking regression was primarily a pipeline bug (50-char truncation), not a data problem.

**Caveat on the 90.7% estimate**: This back-of-envelope calculation is misleading. 68.3% is computed on 180 valid predictions (denominator=180). Multiplying by 239/180 doesn't give "accuracy on valid predictions" — it gives "correct count as percentage of total." The actual valid-only accuracy is already 68.3%. What this estimate really shows is that 123 jobs are correct, same as if the model scored 51.5% on all 239 — highlighting that the valid-only framing inflates the gap between V7 and V9.

#### V10: Good Data, Undertrained → Data ~85-88%

V10 uses a curated subset of V9 data (927 jobs). Labels are inherited from V9 labeling.

**Estimated data label accuracy: ~85-88%** (same labels as V9, just filtered).

The 40-43pp gap (86% data → 45.5% model) is **RESOLVED: undertrained.**

V10 was evaluated at checkpoints 300-600 only (~10 epochs for 927 jobs). V7 trained for 2000 iters (~50 epochs for 642 jobs) — 5x more training. The model's accuracy was **monotonically increasing** at checkpoint 600, meaning it never plateaued or overfit. It simply wasn't given enough iterations.

| Checkpoint | Iters | Epochs | Accuracy | Trend |
|-----------|-------|--------|----------|-------|
| V10 best | 600 | ~10 | 45.5% | ↑ still climbing |
| V10 tokens | 600 | ~15 | 49.6% | ↑ still climbing |
| V7 (for reference) | 2000 | ~50 | 84.9% | converged |

**Key insight**: V10's "catastrophic failure" was comparing a 10-epoch model to a 50-epoch model. The data quality, prompt format, and architecture may be fine — we simply never trained long enough to find out. V10 is the one version where **insufficient training, not data quality, is definitively the main problem**.

#### V11c: Teacher-Filtered → Data ~91-93%

V11c teacher-filters V9 data (keeps only jobs where V7 student model agrees with labels) and includes V7 data directly.

| Data source | Jobs | Label accuracy (vs V7) |
|-------------|------|----------------------|
| V7 data (included directly) | ~300-400 | ~91% |
| V9 data (teacher-filtered) | ~500-600 | ~93-95% (worst disagreements removed) |
| **Combined** | **900** | **~91-93%** |

**Estimated data label accuracy: ~91-93%** (recovered to V7 level via filtering).

**UPDATED**: V11c tokens was re-evaluated on the 239-job V7 test set: **60.6% label accuracy** (not 71%). Per-field: loc 60.6%, arr 63.1%, sen 59.7%, tech 42.4%, comp 40.3%.

The 30-32pp gap (92% data → 60.6% model) is larger than V8's 16pp format penalty. The breakdown:
- ~16pp from no `_raw` fields (same as V8, well-established)
- ~8-10pp from sentence-level cleaning changing input distribution (V7 uses smart truncation; V11c uses sentence extraction — the model was trained on cleaned inputs but evaluated on cleaned inputs too, so this should be minimal, but the cleaning may be too aggressive)
- ~4-6pp from possible teacher filtering circularity + different data mix
- The 71% on 100-job valid vs 60.6% on 239-job test confirms **eval set matters enormously** — a 10.4pp discrepancy from eval set alone

**V11c's data is arguably as good as V7's**, but the format penalty without `_raw` is devastating — even worse than V8's 16pp, likely because the 0.5B model needs `_raw` more when input text has been sentence-cleaned (less context to work with).

#### V11d: Correct Labels, Wrong Inputs → Data ~92-94%

V11d adds 261 synthetic template jobs to V11c's base. The synthetic labels are deterministic and likely ~98% correct.

**Estimated data label accuracy: ~92-94%** (highest of any version due to clean synthetic labels).

**Yet the model scored only 61.6%.** This is paradoxical: V11d has the best data labels but the third-worst model accuracy.

The problem is **input distribution mismatch**, not label quality:
- Synthetic JDs: ~250 chars, template-based, always tracked tech, 0% OOS
- Real JDs: ~1,600 chars median, messy, diverse, ~24% OOS
- The model learned "short text → good_fit" from synthetic examples
- At eval time, all JDs are long/real → the shortcut fails

**V11d disproves the "fix the labels" recommendation in isolation.** Label accuracy is necessary but not sufficient — the input distribution must also match between training and eval.

#### V11e: Good Labels, Destroyed Distribution → Data ~90-92%

V11e aggressively caps NO_GBP at 15% and OUTSIDE_UK at 6%.

**Estimated data label accuracy: ~90-92%** (labels are fine, inherited from V7/V9 pool).

The 47-49pp gap (91% data → 43.8% model) is entirely from **distribution destruction**:
- Removing "easy" bad_fit examples (OUTSIDE_UK, NO_GBP) destroyed the model's base rate calibration
- Without enough clear-negative anchors, the model can't learn what a "clearly bad" job looks like
- Score thresholds drift without the full distribution to anchor them

#### Summary: The Penalty Decomposition

```
                    Data Label    Format    Pipeline   Distribution   Reliability    Total Gap
                    Accuracy      Penalty   Bug Pen.   Mismatch Pen.  (parse+inv)    (all-jobs)
──────────────────────────────────────────────────────────────────────────────────────────────
V7   (baseline)     ~91%          0pp       0pp        0pp            ~10pp (27)     ~16pp
V7   1.5B           ~91%          0pp       0pp        0pp            ~4pp (13)      ~10pp
V8   (no _raw)      ~91%         ~16pp      0pp        0pp            ~1pp (4)       ~17pp
V9   (bugs)         ~86%          0pp       ~13pp      ~3pp           ~20pp (59)     ~35pp
V10  (undertrained)  ~86%         unknown    0pp        0pp           ~5pp (26)      ~45pp†
V11c (filtered)     ~92%         ~24pp       0pp        0pp*          ~1pp (3)       ~32pp
V11d (synthetic)    ~93%         ~16pp       0pp       ~15pp          ~1pp (1)       ~32pp
V11e (caps)         ~91%         ~16pp       0pp       ~30pp          varies         ~47pp
──────────────────────────────────────────────────────────────────────────────────────────────
† V10 gap dominated by ~40pp undertrained convergence (only 10 epochs at 600 iters).
* V11c gap: 75.3% − 59.8% = 15.5pp on all-jobs (less extreme than valid-only 84.9% − 60.6% = 24pp).
  V11c's format penalty vs V8 is clearer on all-jobs: V8=73.6% vs V11c=59.8% → ~14pp difference,
  suggesting ~14pp from sentence cleaning + teacher filtering, on top of V8's ~16pp _raw penalty.
```

**Key revision (rev 7)**: Adding the Reliability column reveals a hidden factor. V7's `_raw` format helps accuracy but **hurts reliability** (27 failures = 10pp on all-jobs). V8 without `_raw` is less accurate but more reliable (4 failures = 1pp). The net effect: V7 all-jobs (75.3%) is only 1.7pp better than V8 (73.6%), not 10pp.

**Key findings:**

1. **Data label accuracy ranges from ~86% to ~93% across all versions** — a narrow 7pp spread.

2. **Model accuracy ranges from 43.8% to 84.9% on valid predictions** (or 40.6% to 80.8% on all jobs) — a 41pp spread. The model spread is **6x wider** than the data spread.

3. **Label accuracy explains at most ~7pp of the 41pp model spread.** The remaining ~34pp comes from:
   - Output format (with/without `_raw`): **~16-24pp** (V8's 16pp to V11c's 24pp)
   - Pipeline bugs (parse failures): **~13pp**
   - Distribution mismatch: **~5-30pp**
   - Training convergence (V10): **~40pp** (undertrained at 10 epochs)

4. **The corrected priority order for V12:**

| Priority | Fix | Expected Impact | Evidence |
|----------|-----|-----------------|----------|
| **1** | **Deploy hybrid (V7 model + regex)** | **+2.1pp (87.0%)** | **MEASURED — already built, no retraining** |
| 2 | **Keep `_raw` fields if retraining** | +16-24pp vs token-only | V8 vs V7 (same data, format change only) |
| 3 | **Audit test set labels** | Unknown but potentially large | 18 "both wrong" jobs may include wrong test labels |
| 4 | **Fix pipeline bugs** (no truncation, no parse failures) | +13pp | V9 parse failure analysis |
| 5 | **Match input distribution** (train ≈ test) | +5-30pp | V11e catastrophic failure |
| 6 | **Improve model seniority** (contrastive examples) | +2-3pp | Model's primary value-add field |
| 7 | **Fix data labels** (~31 surgical fixes) | +1-3pp | Matters less now that tech/comp use regex |

5. **The hybrid approach already achieves 87.0% with the existing V7 model, no retraining.** The foundation was right all along — and the fastest path forward is not to retrain but to supplement the model with regex where it's weaker.

---

## 10. The Tech Field: Why It's Uniquely Hard

### 10.1 It's the Only Multi-Label Array Field

| Field | Type | Possible Outputs | Evaluation |
|-------|------|-----------------|------------|
| loc | single token | 5 choices | exact match |
| arr | single token | 4 choices | exact match |
| sen | single token | 3 choices | exact match |
| comp | single token | 7 choices | exact match |
| **tech** | **array** | **~20+ combinations** | **exact array match** |

### 10.2 Exact-Match Compounds Errors Exponentially

From Finding 37, per-token accuracy is ~85%, but array accuracy degrades:

| Array Size | Accuracy (theoretical) | Accuracy (observed) |
|-----------|----------------------|-------------------|
| 1 token | 85% | 85% |
| 2 tokens | 72% | ~70% |
| 3 tokens | 61% | 55% |
| 4 tokens | 52% | 36-60% |

The model's per-token performance is decent — it's the evaluation metric that punishes partial correctness.

### 10.3 Teacher Self-Consistency Is Lowest

| Field | GPT Self-Agreement |
|-------|-------------------|
| loc | 99.6% |
| sen | 98.3% |
| arr | ~97% |
| comp | 96.7% |
| **tech** | **~90%** |
| **AI_ML specifically** | **52.5%** |

### 10.4 Co-Occurrence Bias

83.3% of REACT training examples also have JS_TS. The model learns `REACT → add JS_TS` as a shortcut. When a job mentions React without explicitly naming TypeScript/JavaScript, the model still adds JS_TS.

### 10.5 AI_ML Is Genuinely Ambiguous

Unlike other fields which have clear textual signals:
- loc: city/country names
- comp: salary numbers with £ signs
- arr: "remote", "hybrid", "on-site"
- sen: "Senior", "Lead", "Junior" in title

AI_ML's trigger is contextual: Is "Copilot" AI/ML? "Data-driven decisions"? "Automation"? Even humans would disagree. The teacher prompt tried to codify rules, but the 52.5% self-agreement shows the rules are inherently ambiguous.

### 10.6 Label Flip Impact of Tech Errors

From the V7 1.5B diagnostics:
- 69 total tech errors
- Only **15 caused label flips** (21.7%)
- 54 tech errors had no impact on the final label

Most tech errors occur on jobs where the score is far from label boundaries. The 15 that flip labels are concentrated in the score 65-74 zone.

### 10.7 What Fixes Would Actually Address

| Fix | Expected tech improvement | Notes |
|-----|--------------------------|-------|
| **Use regex instead of model** | **+11pp (81.6%)** | **MEASURED — already beats model, no training needed** |
| Deterministic comp/loc verification | 0pp | Doesn't touch tech |
| Remove generated data NODE bias | +2-3pp | Fewer false NODE predictions |
| AI_ML strict keyword validation | +3-5pp | Fewer hallucinated AI_ML |
| Contrastive NODE-without-JS_TS examples | +2-3pp | Breaks co-occurrence |
| Switch to per-token F1 eval | **Reframes** | Tech would appear ~82% not 70% |
| Drop AI_ML token entirely | **Eliminates** | Removes noisiest signal |

**Updated finding (rev 5)**: Regex already achieves 81.6% tech accuracy — matching the upper end of what model improvements could deliver with "all fixes." The most practical path is to use regex for tech at inference time, not to improve the model's tech prediction. However, regex has its own AI_ML false positive problem (4× OOS→AI_ML), so further regex tuning could push tech accuracy higher.

---

## 11. Training Data Distribution Analysis

### Label Distribution Across Versions

| Label | V7 Train | V9 Train | V7 Test | V11 (natural) | V11d (balanced) |
|-------|----------|----------|---------|---------------|-----------------|
| good_fit | 24.5% | 14.4% | 23.8% | 19% | 30% |
| maybe | 28.6% | 21.0% | 23.8% | 23% | 30% |
| bad_fit | 46.8% | 64.6% | 52.3% | 58% | 40% |

**V7's natural distribution (24/29/47) is closest to the test set (24/24/52).** V9's heavy bad_fit skew (64.6%) explains part of its regression.

### Tech Token Distribution (V7 Training Data)

- 212 generated jobs: 99.5% NODE, 0% OOS, 50.9% AI_ML
- Real jobs: ~41% NODE, ~24% OOS
- Overall: NODE is over-represented by ~2.4x due to generated data bias

### Compensation Distribution Issues

- V7/V9: 54.5-64% NO_GBP (score 0) — model defaults to NO_GBP when uncertain
- RANGE_55_74K is the most confused comp class (17 errors in V7 eval)
- Boundary salaries (£45k, £55k, £75k, £100k) are underrepresented

### Distribution Lessons

1. **Natural > Forced**: V7's natural distribution outperformed V11d's forced 30/30/40 balance
2. **"Easy" examples aren't noise**: V11e's removal of OUTSIDE_UK/NO_GBP caused catastrophic regression (−27pp)
3. **Match train to test**: The best results come when training distribution mirrors test distribution
4. **Generated data skews distributions dangerously**: 212 NODE-heavy generated jobs = 29.7% of training data with unrepresentative distribution

---

## 12. Label Accuracy vs Model Accuracy Relationship

### The Naive Assumption (Incorrect)

> "If labels are X% correct, the model can't exceed X%, so we need ~99% labels for 90% model accuracy"

### Why This Is Wrong

Models learn patterns, not individual labels. The relationship depends on the **type** of noise:

**Random noise**: Labels randomly wrong with no systematic pattern. The model sees 90 correct and 10 wrong examples of the same pattern, learns the majority, and can predict correctly even on mislabeled examples. Model accuracy can **exceed** label accuracy. Deep learning is robust to random noise up to ~20%.

**Systematic noise**: A specific rule is consistently wrong (e.g., AI_ML always labeled wrong for Copilot jobs). The model learns the wrong rule. Model accuracy **cannot** exceed label accuracy on those cases.

**Our noise is mostly systematic** (prompt-dependent AI_ML behavior, fuzzy correction biases), which is the dangerous kind.

### The Real Relationship

```
Model accuracy ≈ min(label_consistency, pattern_learnability)
```

- **Label consistency**: How often does the same input → same label? (~90% on tech, ~97% on other fields)
- **Pattern learnability**: Can a 0.5B model learn the decision boundary? (yes for simple fields, questionable for multi-token tech)

### The V7 Proof Point

The V7 data quality audit (Section 9) provides direct evidence:

| Metric | Valid-Only | All 239 Jobs |
|--------|-----------|--------------|
| V7 data label accuracy (estimated) | ~90-93% | ~90-93% |
| V7 0.5B model accuracy | 84.9% | 75.3% |
| Gap | ~6pp | ~16pp |

The valid-only gap (~6pp) is learning imperfection + generalization loss. The additional all-jobs gap (~10pp) comes from 27 parse/invalid-token failures. **The model is not failing to learn — it is faithfully reproducing its training data, including the errors.** But it also fails to produce valid output for 11.3% of jobs, which is a separate reliability problem. No training strategy change will break past ~91-93% on valid predictions without fixing the data. But reaching 90%+ on all jobs also requires reducing parse failures (use 1.5B) or using hybrid fallback (regex for unparseable output).

This explains why V8→V11 all failed: they were trying to solve a data quality problem with training strategy changes.

**Updated nuance (rev 5)**: The measured hybrid approach (87.0%) shows that even without fixing any data, bypassing the model on tech/comp achieves higher accuracy than the pure model. The "data is the ceiling" finding is correct for the model alone, but the hybrid circumvents this by using regex where the model is worse than deterministic rules. Additionally, model "accuracy" of 84.9% is measured against GPT test labels that are themselves ~91% correct. The true model accuracy against ground truth may be higher than reported (see Section 13.13).

### What We Actually Need for 90%+ Label Accuracy

**Updated (rev 5)**: The hybrid approach already achieves 87.0%. The path to 90%+ is now:
1. Deploy hybrid (already done: 87.0%)
2. Audit test set labels to determine if the gap is real
3. If real: improve model seniority accuracy + fix regex edge cases

From V7 diagnostics:
- 15 label flips from tech errors (mostly AI_ML)
- 6 from comp errors (fuzzy correction damage)
- 3 from loc errors
- 2 from sen errors
- 0 from arr errors (arr never affects score)

**To reach 90% on valid predictions**: Fix ~31 labels (16 comp fuzzy corrections + 15 worst AI_ML errors) → data accuracy ~95% → model valid-only accuracy ~90%.

**To reach 90% on all jobs**: Either (a) use 1.5B to eliminate parse failures (currently 80.8% → needs ~9pp from data fixes), or (b) use hybrid approach (already at 87.0% → needs ~3pp from regex/model improvements), or (c) both.

**To reach 95% on valid predictions**: Fix ~70 labels + add ~100 contrastive examples + replace 212 biased generated jobs → data accuracy ~98% → model accuracy ~93-95%.

The path to 90% on all jobs is **the hybrid** (already 87.0%). The path to 90% on valid predictions is **surgical data fixes**. Both together target 92%+.

---

## 13. Critique of This Analysis

### 13.1 Apples-to-Oranges Comparisons (Major Flaw)

V7/V8/V9/V10 were evaluated on the **239-job V7 test set**. V11 variants were evaluated on a **100-job validation set**. These are completely different evaluation sets. The accuracy numbers are NOT directly comparable. The "V7 remains unbeaten" narrative rests on comparing numbers from different benchmarks.

V11c at 71% on 100 jobs might perform very differently on the 239-job test set.

### 13.2 Confounding Variables

V9's regression was attributed to "more data = worse quality," but V9 also had the 50-char truncation BUG. If you remove the 48 parse failures and evaluate only valid predictions, V9's accuracy might be closer to V7. The root cause may be a pipeline bug, not data quality — very different problems with very different solutions.

Similarly, V8's −10pp was attributed to removing `_raw`, but V8 also had slightly different formatting code. No controlled experiment isolated just the `_raw` variable.

### 13.3 Ceiling Math Oversimplified

The claim that 90% label correctness = 90% model ceiling is incorrect. Models can generalize past noisy labels if the noise is random. The real ceiling depends on whether noise is random (beatable) or systematic (not beatable). AI_ML noise is partly systematic, making it harder but not a hard 90% cap.

### 13.4 Survivorship Bias in V7

V7 training and testing share the same teacher prompt, same model (gpt-4.1-mini), and same labeling biases. The student learned to mimic gpt-4.1-mini's biases, and the test set rewards those biases. V7 doesn't have 85% "correctness" — it has 85% agreement with gpt-4.1-mini's specific interpretation.

Diagnostic findings already identified 3-5 cases where the student model was correct and the teacher was wrong.

### 13.5 Teacher Filtering Circularity

V11's teacher filtering keeps only V9 jobs where the existing V7 student agrees with labels. This is "training on your own successes" — the model can never learn from cases it currently gets wrong, because those get filtered out. This creates confirmation bias in the data.

### 13.6 V10 Failure Under-Analyzed — **RESOLVED**

~~V10 dropped to 45.5%, which was dismissed as "different prompt format." But a 40pp drop requires deeper explanation. Was the prompt structure incompatible with Qwen2.5 tokenization? Was there a training/eval prompt mismatch? Was V10 evaluated with the wrong prompt? This was never investigated.~~

**Partially resolved**: V10 was undertrained. It was evaluated at iter 300-600 (~10 epochs) vs V7's 2000 iters (~50 epochs). Accuracy was monotonically increasing at the last evaluated checkpoint (45.5% at iter 600, up from 33.1% at iter 300). The "catastrophic failure" was a model that hadn't finished learning.

**What remains unknown**: We don't know what V10 converges to. It could plateau at 55% (if the new prompt format is genuinely incompatible) or reach 80%+ (if the format is fine and just needed more training). Without actually training V10 to 2000 iters, we've resolved *why it was low* but not *whether the approach works*. See updated Section 5 for details.

### 13.7 Deterministic Label Verifier Oversold

Comp is already 96.7% consistent from GPT. Fixing it gains ~3% on one field ≈ ~1pp overall. Loc is already 97-99% accurate. Tech (the bottleneck at 70%) is the field **least amenable to deterministic rules** — AI_ML is contextual and regex can't distinguish "AI as product domain" from "AI as dev tool."

### 13.8 Sample Size Noise

With 239 test jobs, 1pp = 2.4 jobs. The difference between 84.9% and 85.4% (V7 0.5B vs 1.5B) is 1 job. The V11 100-job eval is worse — 1pp = 1 job. Many conclusions are drawn from differences within the noise margin.

### 13.9 Missing Hyperparameter Analysis

The analysis didn't question whether training hyperparameters are optimal:
- LoRA rank 16: optimal for 0.5B? Same rank used for both 0.5B and 1.5B.
- LR 2e-5 with warmup 100: tuned or inherited from V5?
- Effective batch 16: with 642 examples = ~40 updates per epoch. Enough gradient diversity?
- `mask_prompt=true`: with 10-field output (~430 chars) vs 5-field (~90 chars), the loss landscape is completely different. Token-only format gets ~5x fewer gradient tokens per example.

### 13.10 Alternative Task Formulations Not Explored

Training the model to predict the final label directly (good_fit/maybe/bad_fit) — a 3-class classification — would avoid all the field error compounding and array exact-match penalties. This wasn't considered.

### 13.11 Deterministic Baseline — **MEASURED**

~~How well does pure regex (salary parsing, location matching, tech keywords) classify the 239 test jobs? This baseline would tell us the floor and the model's marginal value. Without it, we don't know if the model is adding 5% or 50% over simple rules.~~

**Resolution**: Built and measured. See Section 16 for full results. Summary:
- **Deterministic baseline: 73.6% label accuracy** on the 239-job V7 test set
- Regex beats the V7 model on **tech** (81.6% vs 70.3%) and **comp** (90.4% vs 71.7%)
- Model's primary value-add is **seniority** (89.6% vs 29.3%) — regex can't infer seniority from JD text
- Model also helps loc (95.8% vs 94.1%, marginal) and arr (78.3% vs 77.4%, marginal)
- **Hybrid approach MEASURED at 87.0%** (208/239), 95% CI [82.2%, 90.7%] — new high, no retraining
- 18 jobs wrong for both standalone approaches — but hybrid has 31 errors (field mixing creates ~13 new errors)
- Scripts: `finetune/deterministic_baseline.py`, `finetune/compute_hybrid.py`

### 13.12 Confidence Intervals Ignored — **FIXED**

With n=239, many conclusions in this analysis are within statistical noise:

| Comparison | Framing | Difference | Within 95% CI overlap? |
|-----------|---------|-----------|----------------------|
| V7 0.5B vs V7 1.5B | valid only | +0.5pp (84.9→85.4%) | **Yes** — 1 job |
| V7 0.5B vs V7 1.5B | **all 239** | **+5.5pp (75.3→80.8%)** | **Borderline** — 13 jobs |
| V7 vs V8 | valid only | +10.0pp (74.9→84.9%) | **No** — significant |
| V7 vs V8 | **all 239** | **+1.7pp (73.6→75.3%)** | **Yes** — 4 jobs |
| V7 model vs hybrid | all 239 | +11.7pp (75.3→87.0%) | **No** — significant |
| Regex vs V7 model | valid vs all | +11.3pp (73.6→84.9%) | **No** — but apples-to-oranges denominators |
| Regex vs V7 model | **all 239** | **+1.7pp (73.6→75.3%)** | **Yes** — within noise |

**Rule of thumb**: With n=239, differences <5pp are within noise. Only comparisons >8pp are statistically confident at 95%.

**Critical insight**: The denominator choice changes which comparisons are "significant." On valid-only, V7→V8 (removing `_raw`) looks like a 10pp disaster. On all-jobs, it's 1.7pp noise. The valid-only framing inflates the importance of `_raw` fields by hiding parse failures.

### 13.13 Test Set Quality Never Audited — **UNRESOLVED**

The 239-job V7 test set was labeled by gpt-4.1-mini with the V7 teacher prompt. If the test labels have the same ~91% accuracy as the training data (Section 9.8), then ~21-22 test labels are wrong. This means:

- A model achieving 85% agreement with the test set might actually be **~93-94% accurate** against ground truth
- Some "errors" from the model (and regex) may be correct predictions penalized by wrong test labels
- The "gap to 90%" might not exist at all if measured against human-verified labels
- The 18 "both wrong" jobs from the hybrid analysis might include cases where the test label is wrong

**This is the deepest unresolved question in the entire analysis.** A human audit of 50-100 test set jobs (especially in the score 50-74 zone) would either validate the 91% estimate or reveal the true accuracy is higher than reported.

### 13.14 AI_ML 52.5% Self-Consistency Used Too Broadly

The report frequently cites "52.5% AI_ML self-consistency" as evidence the entire AI_ML signal is coin-flip quality. But this was measured on **edge cases only** — the borderline AI_ML jobs. On clear cases:
- "Machine Learning Engineer" → AI_ML: ~99% consistent
- "Senior React Developer" (no ML) → no AI_ML: ~99% consistent
- "Full Stack + Copilot integration" → edge case: ~52.5% consistent

The 52.5% applies to maybe 30-50 of 311 AI_ML jobs (~10-16%), not all 311. The overall AI_ML accuracy is probably ~85-90%, not ~52.5%. The report sometimes implies the entire tech field's noise comes from AI_ML coin-flipping, which overstates the problem.

### 13.15 Section 9.8 All-Fields-Correct Math Gap

Section 9.8 computes: `P(all 5 fields correct) = 0.99 × 0.97 × 0.98 × 0.93 × 0.95 ≈ 83%` and then estimates label accuracy at ~90-93%. The 7-10pp jump from 83% → 90-93% assumes many field errors don't flip labels. This assumption is supported by V7 1.5B diagnostics showing only 26 label flips out of ~120 field errors (21.7%), but that was measured on the model's errors, not the training data's errors. The training data's label-flip rate could be different (e.g., comp errors from fuzzy correction are concentrated at specific score boundaries where they DO flip labels).

### 13.16 Section 13.7 Contradicted by Section 16 — **NOTED**

Section 13.7 says "Comp is already 96.7% consistent from GPT. Fixing it gains ~1pp overall." Section 16 shows regex beats model on comp by 19pp (90.4% vs 71.7%). These are about different things:
- 13.7: Fixing comp **labels** (data quality) → ~1pp because GPT already gets comp right
- Section 16: Using regex comp **predictions** instead of model predictions → +19pp

The resolution: **don't fix comp labels — bypass the model entirely on comp at inference time**. The model's comp errors aren't from bad training data; they're from the model being worse at salary parsing than regex. This is a task mismatch, not a data quality problem.

---

## 14. Key Lessons

### The Central Lesson

0. **The model already learned the data — the data is the ceiling.** V7 data is ~90-93% correct at the label level. V7 model achieved 84.9% on valid predictions (75.3% on all 239 jobs). The valid-only gap (~6pp) is learning imperfection; the all-jobs gap (~16pp) includes 27 jobs lost to parse failures and invalid tokens. Every V8→V11 experiment tried to solve a data quality problem with training strategy changes. That's why nothing worked. To improve the model, improve the data — and reduce parse failures (the 1.5B model's 80.8% all-jobs shows that model capacity helps here).

### Data Lessons

1. **Data quality > data quantity**: 642 examples (V7, ~91% correct labels) beat 1,010 examples (V9, lower quality + bugs). Every attempt to add data regressed.
2. **Bad data is worse than no data**: V11d's synthetic examples taught wrong shortcuts (short JD = good_fit).
3. **"Easy" examples aren't noise**: V11e's removal of OUTSIDE_UK/NO_GBP destroyed calibration (−27pp).
4. **Diversity > quality > quantity**: The hierarchy is coverage first, accuracy second, volume third.
5. **Generated data creates extreme bias**: 99.5% NODE in 212 generated jobs vs ~41% in real data. 0% OOS vs ~24% real.
6. **Match train distribution to test distribution**: V7's natural distribution (24/29/47) was closest to test (24/24/52) and performed best.
7. **V7 data is "least dirty," not "clean"**: ~50 wrong tech labels, ~35 wrong comp labels, 212 biased generated jobs = ~25-30% of examples with some issue.

### Label Lessons

7. **gpt-4.1-mini >> gpt-4o-mini**: 10x fewer fuzzy corrections, dramatically cleaner labels.
8. **Fuzzy correction masks semantic errors**: RANGE_35_44K → RANGE_45_54K instead of BELOW_45K silently corrupts ~50 labels.
9. **AI_ML is the noisiest signal**: 52.5% teacher self-agreement on edge cases. Either fix the rules or drop the token.
10. **Teacher prompt changes = label drift**: V7→V9 teacher prompt change caused 90 tech disagreements. Always relabel with the latest prompt or keep prompt/data in sync.
11. **Fix labels surgically, not wholesale**: Only ~26 label-flip cases matter. Fix those, not all 1000.

### Architecture Lessons

12. **`_raw` fields are load-bearing**: +10pp accuracy, ~5x more gradient signal. Don't remove them.
13. **Model size is not the bottleneck**: 0.5B and 1.5B have identical tech accuracy (70.4% vs 70.3%). The constraint is data, not capacity.
14. **Parse failures matter more for 0.5B**: 1.5B eliminates them (0 vs 15). Consider 1.5B if parse rate is critical.
15. **More iterations ≠ better**: V11c_long overfit (64.2% < 71.0%). Optimal checkpoint is often early.
16. **Consistent preprocessing is critical**: Apply the same transforms at training AND inference time. Any mismatch creates distribution shift.

### Pipeline Lessons

17. **Pipeline bugs can look like data problems**: V9's 50-char cap bug caused a 16.6pp regression that was initially attributed to data quality.
18. **SIGPIPE destroys files**: Never pipe write scripts through head/tail.
19. **Validate the full pipeline**: Check inputs, outputs, and transforms — not just final results.
20. **When models of different sizes fail identically**: The bottleneck is data, not capacity.
21. **Temperature=0 ≠ determinism for GPT**: Multiple runs needed before trusting labeling consistency.

### Evaluation Lessons

22. **Don't compare across different eval sets**: V7 (239 jobs) vs V11 (100 jobs) numbers are not comparable.
23. **Tech exact-array-match may be too strict**: Per-token F1 would show tech is ~82%, not 70%.
24. **Sample sizes matter**: With 239 test jobs, 1pp = 2.4 jobs. Many "findings" are within noise margin.
25. **Agreement ≠ correctness**: The model is measured against GPT labels, not ground truth. Some "errors" are the model being right.

---

## 15. Recommended Training Plan for V12

### Phase 1: Measure What We Have (Before Any Changes)

1. **Run V11c tokens on the 239-job V7 test set** to get a fair comparison with V7.
2. **Measure deterministic baseline**: Build regex classifiers for all 5 fields, evaluate on the 239-job test set. This establishes the floor.
3. **Analyze V7 training data at the boundary**: Identify the ~50-80 examples in score 50-74 zone and audit their labels manually.

### Phase 2: Fix Labels (No Retraining Yet)

4. **Fix fuzzy comp corrections**: Replace edit-distance matching with a semantic lookup table:
   - RANGE_35_44K → BELOW_45K (not RANGE_45_54K)
   - RANGE_25_44K → BELOW_45K
   - RANGE_25_34K → BELOW_45K

5. **Validate AI_ML assignments**: Use strict keyword list — only trigger AI_ML for: "machine learning", "AI", "artificial intelligence", "NLP", "computer vision", "deep learning", "LLM", "neural network" as core job requirements (not developer tools like Copilot/Cursor/Codeium).

6. **Remove or replace 212 NODE-biased generated jobs**: Either drop entirely or replace with deterministically-labeled template jobs covering diverse tech combos including OOS.

7. **Remove 17 contaminated test jobs** from training data.

### Phase 3: Build V12 Training Set

8. **Start from V7's 642 verified examples** (after Phase 2 fixes).
9. **Add ~100-150 targeted contrastive examples**:
   - 30× NODE without JS_TS (Node.js backend, no TypeScript mentioned)
   - 30× REACT without JS_TS (React UI, no explicit TS/JS)
   - 20× OOS (pure Python/Java/Go/Rust stacks)
   - 20× AI_ML clear positive cases (ML engineer roles)
   - 20× AI_ML clear negative cases (mentions AI tools but not ML role)
   - 20× comp boundary cases (£45k, £55k, £75k, £100k thresholds)
   - 10× loc edge cases (remote-UK vs remote-global)
10. **Label contrastive examples deterministically** where possible (template-based, known-correct labels).
11. **Maintain V7-style distribution**: ~24% good_fit, ~24% maybe, ~52% bad_fit.
12. **Keep `_raw` fields** in training format.
13. **Apply sentence-level JD cleaning** at training time AND use the same cleaning at inference time.
14. **Target size: ~750-800 high-quality examples**.

### Phase 4: Train and Evaluate

15. Train 0.5B with V7 hyperparameters (rank 16, LR 2e-5, warmup 100, 2000 iters).
16. Evaluate at every 50 checkpoints on the 239-job V7 test set.
17. If parse failures > 5, also train 1.5B.
18. **Target: 88%+ label accuracy** (would be a new high).

### Phase 0: Deploy Hybrid (ALREADY MEASURED — 87.0%)

This should be done FIRST, before any retraining:

19. **Use existing V7 model + regex hybrid** — already measured at **87.0% (208/239)**, 95% CI [82.2%, 90.7%]
20. Scripts already built: `finetune/deterministic_baseline.py` (regex) + `finetune/compute_hybrid.py` (combination)
21. Integrate into inference pipeline: model predicts all 5 fields, regex overrides tech + comp
22. Parse failure fallback: if model output is unparseable, use regex for all fields

### Phase 5: Beyond 87% — Targeted Improvements

After deploying the hybrid baseline:
23. **Audit 50-100 test set labels** to determine if the gap to 90% is real or an artifact of noisy test labels
24. Train model on **all 5 fields** (with `_raw` for gradient benefit) but use regex overrides at inference
25. Focus model improvement on **seniority** (model's primary value-add: +60pp over regex)
26. **Measured ceiling so far**: 87.0% (208/239). Improving requires fixing 31 hybrid errors — most are field-mixing conflicts (model loc + regex tech/comp combine into wrong score)

### Phase 5b: Focused Model Training

If hybrid accuracy needs further improvement:
27. Train model on **only sen** + use `_raw` for chain-of-thought — less to learn, higher accuracy on the field that matters
28. The model's marginal value over regex is concentrated in seniority. Optimizing seniority accuracy yields the highest ROI.
29. Consider 1.5B for seniority-focused training (eliminates parse failures).

### Alternative: Simplify the Task

If per-field prediction remains stuck:
- Train model to predict **final label directly** (good_fit/maybe/bad_fit)
- 3-class classification, no array exact-match, no error compounding
- Loses interpretability but maximizes label accuracy
- Could reach 90%+ more easily
- But loses the ability to explain WHY a job is good/bad fit

---

## 16. Deterministic Baseline & Hybrid Approach

### 16.1 The Question

How well does pure regex/keyword matching classify the 239 test jobs? This establishes the floor — what a model needs to beat to justify its existence.

### 16.2 Deterministic Baseline Results

**Script**: `finetune/deterministic_baseline.py`
**Eval set**: 239-job V7 test set (same as all V7/V8/V9/V10 evals)
**Result**: **73.6% label accuracy** (176/239 jobs correct)

Per-field accuracy:

| Field | Deterministic | V7 Model (0.5B) | V7 Model (1.5B) | Winner |
|-------|--------------|-----------------|-----------------|--------|
| **loc** | **94.1%** | 95.8% | 97.3% | Model (marginal) |
| **arr** | **77.4%** | 78.3% | 91.2% | Model (marginal for 0.5B) |
| **sen** | **29.3%** | 89.6% | 92.0% | **Model (+60pp)** |
| **tech** | **81.6%** | 70.3% | 70.4% | **Regex (+11pp)** |
| **comp** | **90.4%** | 71.7% | 77.9% | **Regex (+13-19pp)** |
| **Label** | **73.6%** | 84.9%† | 85.4% | Model (+11pp) |

*† V7 0.5B label accuracy is 84.9% on 212 valid predictions. On all 239 jobs (counting 27 parse failures as wrong), it's 75.3%. All field accuracies are computed on valid predictions only.*

**Note on methodology**: Both model and regex accuracy are measured against GPT-4.1-mini labels on the V7 test set — not against human-verified ground truth. If the test labels are ~91% correct (see Section 9.8), then some "errors" from both regex and model may actually be correct predictions penalized by wrong labels.

### 16.3 What This Reveals

**The model's value is concentrated in one field: seniority.** Regex beats the model on tech and comp — the two fields with the most training effort. This is a surprising and important finding.

**Why regex beats the model on tech**:
- Regex matches keyword lists deterministically: "Node.js" → NODE, "React" → REACT, "TypeScript" → JS_TS
- No co-occurrence bias — regex doesn't assume React implies TypeScript
- Exact keyword matching is actually what the tech classification task needs

**But regex has its own tech errors** (44 total, 81.6% accuracy):
- **AI_ML false positives**: 4× OOS→AI_ML (regex catches "AI" in non-ML contexts), 2× REACT→REACT+AI_ML, 2× NODE+JS_TS→NODE+JS_TS+AI_ML — regex over-triggers on "AI" keyword
- **NODE misses**: 3× NODE+REACT+JS_TS+AI_ML→REACT+JS_TS+AI_ML (missed "Node.js" mention), 2× NODE+JS_TS→OOS (missed completely)
- **JS_TS misses**: 3× NODE+JS_TS→NODE (TypeScript not caught by regex pattern)

The claim "no hallucination" is partially wrong — regex DOES add AI_ML when it shouldn't (broad `\bai\b` pattern). Regex tech errors are different in kind from model tech errors (regex misses keywords; model hallucinates co-occurrences), but neither is error-free.

**Why regex beats the model on comp**:
- Salary parsing with regex is essentially solved: find £XX,XXX patterns, compute midpoints, classify into bins
- No fuzzy correction errors — regex maps directly to bins
- Handles edge cases (range, per annum, "up to") programmatically
- The comp field is inherently a parsing task, not a classification task

**Why the model crushes regex on seniority**:
- Seniority requires contextual understanding of job descriptions
- "Lead" in "Lead Developer" vs "Lead generation" vs "Technical Lead"
- Implied seniority from "10+ years experience," "managing a team," "reporting to CTO"
- This is genuine NLP — keyword matching hits only obvious cases (Senior, Junior, Lead in title)

**Why the model slightly wins on loc/arr**:
- Location has some contextual clues beyond city names (e.g., "UK-based clients" implying UK)
- Arrangement inference from context (e.g., "flexible working" → could be HYBRID or REMOTE)
- But regex handles the clear cases (city names, "Remote" keyword) well enough

### 16.4 The Hybrid Approach — MEASURED

**Script**: `finetune/compute_hybrid.py`

Combine the best of each — model loc/arr/sen + regex tech/comp — and compute scores on ALL 239 test jobs:

| Field | Source | Accuracy (on valid model predictions) |
|-------|--------|---------------------------------------|
| loc | Model | 95.8% |
| arr | Model | 78.3% |
| sen | Model | 89.6% |
| tech | **Regex** | **82.1%** |
| comp | **Regex** | **89.6%** |

**MEASURED hybrid label accuracy: 87.0% (208/239)**, 95% CI [82.2%, 90.7%].

For the 27 jobs where the model produced unparseable output, the hybrid falls back to regex-only predictions.

#### Comparison (all on 239-job V7 test set)

| Method | Correct | Total | Accuracy | 95% CI |
|--------|---------|-------|----------|--------|
| V7 model only (valid) | 180 | 212 | 84.9% | [79.5%, 89.1%] |
| V7 model (all, failures=wrong) | 180 | 239 | 75.3% | [69.5%, 80.3%] |
| Regex only | 176 | 239 | 73.6% | [67.7%, 78.8%] |
| **Hybrid A (model+regex)** | **208** | **239** | **87.0%** | **[82.2%, 90.7%]** |

**Note**: The model's commonly reported 84.9% excludes 27 jobs with parse failures. On the full test set, the model only gets 75.3% — barely above regex (73.6%). The hybrid's 87.0% is on the FULL test set, including parse failure fallback. This is a fairer comparison.

#### Per-Label Hybrid Accuracy

| Label | Correct | Total | Accuracy | 95% CI |
|-------|---------|-------|----------|--------|
| good_fit | 43 | 57 | 75.4% | [62.9%, 84.8%] |
| maybe | 44 | 57 | 77.2% | [64.8%, 86.2%] |
| bad_fit | 121 | 125 | 96.8% | [92.1%, 98.7%] |

bad_fit classification is excellent (96.8%). The remaining errors are concentrated in good_fit and maybe, where scores are near the 50/70 boundaries.

#### Error Correlation — Model vs Regex

| | Regex correct | Regex wrong |
|---|---|---|
| **Model correct** | 140 | 40 |
| **Model wrong** | 14 | **18** |

- **18 jobs** are wrong for BOTH model and regex (7.5% of test set) — but NOT a hard ceiling (see below)
- **40 jobs** only the model gets right — these are where the model adds value (mostly seniority)
- **14 jobs** only regex gets right — these are where regex saves the hybrid

The hybrid captures the 14 regex-only wins alongside the 140 where both agree. The 40 model-only wins are partially retained (model still handles loc/arr/sen).

**Important**: The 18 "both wrong" jobs are NOT a hard ceiling for the hybrid. The hybrid mixes *fields* from different approaches (model loc/arr/sen + regex tech/comp), not final labels. This means the hybrid can fix some "both wrong" jobs (by combining correct fields) but also *break* "both correct" jobs (by mixing wrong fields into a wrong score). In practice, the hybrid has 31 errors (239−208), not 18 — field mixing introduces ~13 new errors that neither standalone approach produces.

#### Production Integration Note

The hybrid approach requires building an inference pipeline that:
1. Runs the V7 model for loc/arr/sen (loads adapter, runs inference)
2. Runs regex for tech/comp (fast, no GPU needed)
3. Combines predictions and computes scores via the scoring formula
4. Falls back to regex-only when model output is unparseable

This is a software engineering task, not a training experiment. The inference pipeline already exists — it just needs the regex override step added.

### 16.5 Implications for V12

1. **The model's marginal value over regex is primarily seniority (+60pp)**. For tech and comp, the model adds noise to what regex does cleanly.

2. **87.0% is achieved with NO retraining** — just combining existing V7 model + regex. This is the immediate win.

3. **A V12 "model" could be just regex + a seniority classifier**. Train the model on sen (+ loc/arr for `_raw` gradient benefit), override tech/comp with regex at inference.

4. **The hybrid has 31 errors, not 18** — field mixing creates ~13 new errors that neither standalone approach produces. To push past 87.0% requires: (a) fixing model loc errors (9 REMOTE→UNK cases where "remote" appears in JD text but not job_location), (b) fixing regex tech errors (NODE misses, AI_ML false positives), and (c) auditing test set labels (some of the 31 errors may have wrong golden labels).

5. **Parse failure handling is critical**: V7 model has 27 parse failures (11.3% of test set). The hybrid masks this by falling back to regex, but ideally the model should produce valid output for all jobs. Using the 1.5B model (0 parse failures) with hybrid could potentially reach 88-89%.

### 16.6 All Methods Compared (239-job V7 Test Set)

| Method | Denominator | Label Acc | 95% CI | loc | arr | sen | tech | comp |
|--------|-------------|-----------|--------|-----|-----|-----|------|------|
| **Hybrid (V7+regex)** | **239** | **87.0%** | **[82.2%, 90.7%]** | 95.8% | 78.3% | 89.6% | 82.1% | 89.6% |
| V7 1.5B (valid only) | 226 | 85.4% | [80.2%, 89.5%] | 97.3% | 91.2% | 92.0% | 70.4% | 77.9% |
| V7 0.5B (valid only) | 212 | 84.9% | [79.5%, 89.1%] | 95.8% | 78.3% | 89.6% | 70.3% | 71.7% |
| V7 1.5B (all 239) | **239** | **80.8%** | [75.3%, 85.3%] | — | — | — | — | — |
| V7 0.5B (all 239) | **239** | **75.3%** | [69.5%, 80.3%] | — | — | — | — | — |
| V8 0.5B (all 239) | **239** | **73.6%** | [67.7%, 78.9%] | — | — | — | — | — |
| Deterministic regex | **239** | **73.6%** | [67.7%, 78.8%] | 94.1% | 77.4% | 29.3% | 81.6% | 90.4% |
| V11c tokens (all 239) | **239** | 60.6% | [54.3%, 66.5%] | 60.6% | 63.1% | 59.7% | 42.4% | 40.3% |

**Key takeaway: V8 all-jobs = regex all-jobs = 73.6%.** Removing `_raw` fields made the model no better than regex on a production basis (V7→V8 drops from 75.3% to 73.6%, matching regex exactly). The "10pp _raw advantage" (84.9% vs 74.9%) is mostly a denominator artifact — V7 excludes 27 jobs (11.3%) vs V8 excluding 4 (1.7%).

**V11c tokens is worse than regex on every field**, but this reflects V11c's specific configuration (no `_raw`, sentence-cleaned V9 data, teacher-filtered, 600 iters) — not a general indictment of models. **V7 with `_raw` beats regex on 3/5 fields** (loc, arr, sen). The lesson is about the V11c configuration, not models vs regex in general.

**V7 model vs hybrid**: On valid predictions, the CIs overlap [79.5%, 89.1%] vs [82.2%, 90.7%] — within noise. But on all 239 jobs, the hybrid clearly wins: 87.0% vs 75.3% (+11.7pp), because the hybrid handles parse failures gracefully via regex fallback.

---

## 17. Appendix: File Reference

### Prompts
| File | Purpose |
|------|---------|
| prompts/teacher_v7.txt | V7 teacher — 5 sections, 3 examples, semantic rules |
| prompts/teacher_v8.txt | V8 teacher — same as V7 but no _raw in output |
| prompts/teacher_v9.txt | V9 teacher — AI_ML exclusions, 50-char caps, TC rules |
| prompts/teacher_tech_v9.txt | Tech-only extractor — 4-step approach, 10 examples |
| prompts/student_v7.txt | V7 student — 14 lines, 10-field output |
| prompts/student_v8.txt | V8 student — 14 lines, 5-field output |
| prompts/student_v9.txt | V9 student — 14 lines, 10-field, 50-char caps |
| prompts/student_v10.txt | V10 student — 21 lines, 5-field, "Allowed tokens" |
| prompts/student_v10_tokens.txt | V10 tokens — 20 lines, numbered rules |
| prompts/student_v11.txt | V11 student — 13 lines, 10-field (missing sen_raw) |
| prompts/student_v11_tokens.txt | V11 tokens — 19 lines, strongest token rules |

### Formatting Scripts
| File | Purpose |
|------|---------|
| src/cli/format-for-mlx-v7.ts | V7 MLX formatter — smart truncation |
| src/cli/format-for-mlx-v8.ts | V8 MLX formatter — token-only output |
| src/cli/format-for-mlx-v9.ts | V9 MLX formatter — 50-char cap (BUGGY) |
| src/cli/format-for-mlx-v10.ts | V10 MLX formatter — filterTechRaw, raw modes |
| src/cli/format-for-mlx-v10-tokens.ts | V10 tokens formatter |
| src/cli/format-for-mlx-v11.ts | V11 MLX formatter — dual output |
| src/cli/build-v10-training-set.ts | V10 data builder — distribution balancing |
| src/cli/build-v11.ts | V11 data builder — teacher filtering + sentence cleaning |
| src/cli/clean-raw-fields-v10.ts | Post-labeling _raw field condensing |
| src/cli/preprocess-eval-noise.ts | Eval-time JD noise reduction |
| src/lib/filter-tech-raw.ts | Label-guided tech_raw extraction |

### Training Configs
| File | Purpose |
|------|---------|
| finetune/lora_config_v7.yaml | V7 0.5B — rank 16, LR 2e-5, 2000 iters |
| finetune/lora_config_v7_1.5B.yaml | V7 1.5B — same hyperparams |
| finetune/lora_config_v8.yaml | V8 config |
| finetune/lora_config_v9.yaml | V9 config |
| finetune/lora_config_v10.yaml | V10 config |
| finetune/lora_config_v10_quick.yaml | V10 quick config |
| finetune/lora_config_v11.yaml | V11 base — 400 iters |
| finetune/lora_config_v11_tokens.yaml | V11 tokens — varies by variant |
| finetune/lora_config_v11_tokens_1p5b.yaml | V11 1.5B — 50 iters (short test) |

### Eval Scripts
| File | Purpose |
|------|---------|
| finetune/eval_student_v7.py | V7 eval — standard 5-field scoring |
| finetune/eval_student_v11.py | V11 eval — clean/hint/postprocess options |
| finetune/compare_evals.py | Side-by-side eval comparison |
| finetune/deterministic_baseline.py | Regex classifier baseline — 73.6% label acc |
| finetune/compute_hybrid.py | Hybrid accuracy computation — 87.0% measured |
| finetune/v11_preproc.py | V11 preprocessing (clean, hint, postprocess) |
| finetune/test_v11_preproc.py | Unit tests for V11 preprocessing |

### Inference Scripts
| File | Purpose |
|------|---------|
| finetune/infer_v7_teacher.py | Run V7 teacher on labeled data |
| finetune/infer_v7_teacher_tokens.py | Generate synthetic labels for unlabeled data |
| scripts/label_with_gpt.py | Sequential GPT labeling |
| scripts/label_with_gpt_async.py | Async concurrent GPT labeling |

### Data
| File | Purpose |
|------|---------|
| data/v7/test_labeled.jsonl | 239 test jobs (chmod 444, locked) |
| data/v7/train_labeled.jsonl | 713 labeled training jobs |
| data/v7/mlx/{train,valid}.jsonl | 642/71 MLX chat format |
| data/v8/mlx/{train,valid}.jsonl | 641/72 MLX chat format |
| data/v9/train_labeled.jsonl | 1081 labeled jobs (6 batches) |
| data/v9/mlx/{train,valid}.jsonl | 1010/71 MLX chat format |
| data/v10/mlx/{train,valid}.jsonl | 927/73 standard |
| data/v10/mlx_clean_raw/{train,valid}.jsonl | 927/73 clean_raw variant |
| data/v10/mlx_essential/{train,valid}.jsonl | 927/73 essential variant |
| data/v10_tokens_v7/mlx/{train,valid}.jsonl | 641/72 V7-sized tokens |
| data/v11/mlx/{train,valid}.jsonl | 900/100 full schema |
| data/v11/mlx_tokens/{train,valid}.jsonl | 900/100 token-only |
| data/v11b/mlx_tokens/{train,valid}.jsonl | 540/60 balanced smaller |
| data/v11c/mlx_tokens/{train,valid}.jsonl | 900/100 different selection |
| data/v11d/mlx_tokens/{train,valid}.jsonl | 900/100 +synthetic balanced |
| data/v11e/mlx_tokens/{train,valid}.jsonl | 810/90 aggressive caps |

### Adapters
| Directory | Model | Best Checkpoint | Notes |
|-----------|-------|----------------|-------|
| finetune/adapters_v7/ | 0.5B | iter 2000 | 84.9% label acc |
| finetune/adapters_v7_1.5B/ | 1.5B | iter 2000 | 85.4% label acc |
| finetune/adapters_v8/ | 0.5B | iter 1850 | 74.9% |
| finetune/adapters_v9/ | 0.5B | iter 1800 | 68.3% |
| finetune/adapters_v10/ | 0.5B | — | Not evaluated |
| finetune/adapters_v10_quick/ | 0.5B | iter 600 | 45.5% |
| finetune/adapters_v11/ | 0.5B | — | 0% (100 parse failures) |
| finetune/adapters_v11_tokens/ | 0.5B | iter 200 | 65% |
| finetune/adapters_v11_tokens_b/ | 0.5B | iter 25 | Not fully evaluated |
| finetune/adapters_v11_tokens_c/ | 0.5B | iter 600 | 71% (best V11) |
| finetune/adapters_v11_tokens_c_long/ | 0.5B | iter 500 | 64.2% |
| finetune/adapters_v11_tokens_d/ | 0.5B | iter 400 | 61.6% |
| finetune/adapters_v11_tokens_e/ | 0.5B | iter 200 | 43.8% |

### Eval Results
| Directory | Eval Set | Versions |
|-----------|----------|----------|
| eval_results/adapters_v7/ | 239-job test | V7 0.5B |
| eval_results/adapters_v7_1.5B/ | 239-job test | V7 1.5B |
| eval_results/adapters_v8/ | 239-job test | V8 |
| eval_results/adapters_v9/ | 239-job test | V9 |
| eval_results/adapters_v10_quick/ | 239-job test | V10 + preprocessed |
| eval_results/adapters_v10_clean_raw_quick/ | 239-job test | V10 clean_raw |
| eval_results/adapters_v10_essential_quick/ | 239-job test | V10 essential |
| eval_results/adapters_v10_tokens_v7_quick/ | 239-job test | V10 tokens |
| eval_results_v11/ | 100-job valid + 239-job test | All V11 variants (incl. V11c on V7 test: 60.6%) |

### Diagnostic Documents
| File | Purpose |
|------|---------|
| V6_DIAGNOSTIC_FINDINGS.md | 44 findings with root causes and fixes |
| V6_STUDENT_TRAINING_PLAN.md | Execution plan with 17 steps and status markers |
| docs/v10_tokens_soft_eval_regression_audit_2026-03-12.md | V10 regression analysis |
| docs/V6_V11_DEEP_ANALYSIS_2026-03-13.md | This document |

---

## Changelog

- 2026-03-13 (rev 7b): **Consistency pass — all sections updated to all-jobs framing.** Sections 3.1, 9.9, 9.10, 9.11, 12, 14 updated to show both valid-only and all-jobs numbers. Penalty decomposition table expanded with Reliability column (parse+invalid failures). Key revisions: (1) Section 3.1 heading now says "+10pp valid / +1.7pp all-jobs". (2) Section 9.9 table now has Valid-Only and All-Jobs columns with both 0.5B and 1.5B numbers. (3) Section 9.11 cross-version table expanded with All-Jobs Acc column. (4) V8 paragraph notes reliability vs accuracy trade-off. (5) Section 12 proof point updated with dual-framing table. (6) Section 14 central lesson updated with all-jobs gap (16pp) and 1.5B context. (7) Section 9.10 targets updated to distinguish valid-only vs all-jobs paths to 90%.
- 2026-03-13 (rev 7a): **Denominator audit — "84.9%" reframed.** Investigation found V7's 84.9% excludes 27 jobs from the denominator (15 parse failures + 12 invalid tokens). All-Jobs Acc column added to results table with verified numbers from eval summary JSONs (all pass sanity check: valid + parse + inv_tok = 239). Key corrections: (1) V7 0.5B = 75.3% on all jobs, V7 1.5B = 80.8%. (2) V7→V8 `_raw` advantage is 1.7pp on all-jobs (not 10pp on valid-only) — parse failures absorb most of the gap. (3) V8 all-jobs = regex = 73.6% (model without `_raw` ties regex in production). (4) 1.5B advantage is 5.5pp on all-jobs (not 0.5pp on valid-only) — mostly from zero parse failures. (5) V9 "90.7% on valid" estimate flagged as misleading math. (6) CI comparison table expanded with all-jobs vs valid-only framing — shows denominator choice determines which comparisons are "significant." (7) V10 iter 500 parse/inv_tok numbers corrected (were swapped). (8) V10 iter 300 parse_fail corrected (5, not 14; inv_tok 71, not 62). (9) Methodology note added explaining eval script denominator conventions. No contamination found (zero train/test overlap by job ID or JD text). Parse failure bias check: excluded jobs are disproportionately easy (70.4% bad_fit, 1.5B scores 91.7% on them) — 84.9% is not inflated by excluding hard jobs, but 75.3% is the production-relevant number.
- 2026-03-13 (rev 6): **Method review — 2 bugs fixed.** (1) Error correlation table corrected: "Model correct, regex correct" was 166 (wrong), now 140. Bug: code subtracted `model_wrong_regex_right` instead of `regex_wrong_model_right`. Quadrants now sum correctly: 140+40+14+18=212. (2) "Theoretical max 92.5%" claim removed throughout (executive summary, Section 13.11, 15, 16.4, 16.5). The 18 "both wrong" standalone jobs are NOT a ceiling for the hybrid because the hybrid mixes *fields* not labels — and field mixing creates ~13 NEW errors (hybrid has 31 errors total, not 18). Reframed as measured ceiling of 87.0% with actionable breakdown of the 31 errors.
- 2026-03-13 (rev 5): **Comprehensive critique fixes — 15 issues resolved.** (1) Hybrid accuracy MEASURED at **87.0% (208/239)**, 95% CI [82.2%, 90.7%] — replaces the estimated "87-90%" throughout. Script: `compute_hybrid.py`. (2) V7 per-field numbers reconciled to run 2 (044934): loc=95.8%, arr=78.3%, sen=89.6%, tech=70.3%, comp=71.7%. Run 1 numbers were from a different eval with 98 parse failures — mixed-run data removed. (3) Central finding reframed: "model learned easy parts, failed on hard parts — regex does those better." (4) Confidence intervals added throughout — with n=239, differences <5pp are within noise. (5) Penalty decomposition table updated: V10 = undertrained, V11c = 24pp gap (measured), stale footnote removed. (6) V11c "worse than regex on every field" reframed as V11c-specific configuration problem, not general model indictment — V7 with `_raw` beats regex on 3/5 fields. (7) V10 resolution noted as partial — we know WHY but not whether it converges. (8) Test set quality critique added (Section 13.13): 21-22 test labels likely wrong, true accuracy may be higher. (9) AI_ML 52.5% scoped to edge cases only (Section 13.14). (10) Section 9.8 all-fields math gap acknowledged. (11) Section 13.7 vs Section 16 contradiction resolved (Section 13.16). (12) Regex error patterns added to Section 16.3 (AI_ML false positives, NODE misses). (13) Production integration note added to hybrid section. (14) V12 training plan reordered: deploy hybrid FIRST, then audit test set, then retrain. (15) Error correlation analysis added: 18 hard-floor jobs, theoretical max ~92.5%. Model true accuracy on all 239 = 75.3% (27 parse failures counted), not 84.9%.
- 2026-03-13 (rev 4): **Gap-closing findings — V10, V11c, deterministic baseline.** Three critical gaps closed: (1) V10 root cause RESOLVED — undertrained at 600 iters (~10 epochs), not broken architecture. Accuracy was monotonically increasing. Updated Section 5 and 13.6. (2) V11c tokens evaluated on 239-job V7 test set: **60.6%** (not 71%). 10.4pp discrepancy from eval set difference. Updated comparison tables, Section 9.11, per-field table. (3) Deterministic regex baseline **measured at 73.6%** — regex beats V7 model on tech (81.6% vs 70.9%) and comp (90.4% vs 72.4%). Model's value-add is primarily seniority (+63pp). Added new Section 16 (Deterministic Baseline & Hybrid Approach) with full results, hybrid recommendation (~87-90%), and implications. Updated executive summary, V12 training plan (Phase 5 now measured hybrid approach), and Section 13.11. V11c without `_raw` is worse than pure regex on every field — strongest evidence yet that `_raw` is non-negotiable. Renumbered Appendix to Section 17.
- 2026-03-13 (rev 3): **Cross-version data accuracy analysis** — Added Section 9.11 comparing estimated data label accuracy vs model accuracy for ALL versions (V7-V11e). Key finding: data label accuracy spans only 7pp (86-93%) while model accuracy spans 41pp (43.8-84.9%). Label quality explains at most 7pp of the gap. The remaining 34pp comes from: output format penalty (~16pp), pipeline bugs (~13pp), and distribution mismatch (~5-30pp). Corrected the V12 priority order: keeping `_raw` fields (+16pp) and fixing distribution (+5-30pp) are more important than fixing labels (+3-5pp). Added V9 reanalysis showing parse-failure-adjusted accuracy is ~90.7% on valid predictions.
- 2026-03-13 (rev 2): **Major correction** — Added Section 9 (V7 Data Quality Audit) with full token distribution analysis, generated data audit (212 jobs, 99.5% NODE), AI_ML label audit (3-4 clear errors, ~40-50 borderline), comp fuzzy correction audit (16 semantically wrong corrections), boundary zone analysis (252 jobs in score 50-74). Corrected executive summary: V7 data is ~90-93% correct at label level, not "clean." Model accuracy (84.9%) ≈ data accuracy (91%) minus learning penalty. Corrected generated job count from 228 → 212 throughout. Added "model already learned the data" as central finding. Updated Section 12 with V7 proof point. Updated Section 14 Key Lessons with corrected framing.
- 2026-03-13 (rev 1): Initial comprehensive analysis covering V6→V11 training journey

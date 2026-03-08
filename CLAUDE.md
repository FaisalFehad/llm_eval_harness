# Student Model Training Project — V5 Semantic Tokens

## Project Overview

Distilling GPT-4o-mini into a local Qwen2.5-0.5B-Instruct-4bit model for job-fit scoring. The model classifies LinkedIn job postings into semantic tokens across 4 fields. Code computes the numeric score and label deterministically. GPT-4o-mini is the single source of truth for all labeling — never question its labels.

## Current Status

Best result: **83.9% label accuracy** (v5.1, checkpoint 875, 895 training jobs).
Target: **90-93%** (0.5B ceiling).
Currently in **V6 pre-training analysis** — fixing data quality issues before retraining.

### V6 Tracking Documents (IMPORTANT — always keep these updated)

These two files are our map and history. Every finding, decision, and verification goes here:

| Document | Purpose |
|----------|---------|
| `V6_STUDENT_TRAINING_PLAN.md` | The execution plan — what to do, in what order, with status markers |
| `V6_DIAGNOSTIC_FINDINGS.md` | The history — every problem found, how found, impact, decision, lessons |

**Rules**: After completing any analysis or making any decision, update BOTH documents. The plan tracks what's done/todo. The findings track what we learned and why. Never declare something "fixed" without a verification step in the plan. Claims about V6 fixing something are hypotheses until verified by re-labeling.

### What's Been Done (V6 Pre-Training)
- V5.1 diagnostics complete: generalization gap, token frequency, boundary zones, checkpoint stability
- V6 teacher prompt written (`prompts/teacher_v6.txt`) — engineering gate, COMP ignore rules, AI/ML keywords
- 20 findings documented (see `V6_DIAGNOSTIC_FINDINGS.md`)
- Spec violations vs spec gaps analysis complete — found 24 issues in 4 categories (14 spec gaps, 7 title mismatches, 2 spec violations, 1 V6-still-wrong)
- Execution order defined (12 steps with status tracking)
- Post-relabeling verification checklist created (Step 5.5)

### What's In Progress (V6 Execution Order)
1. ✅ Lock rules (teacher_v6.txt)
2. ✅ Classify spec violations vs gaps
3. ⬜ **BLOCKER: Investigate title mismatch pipeline bug** (7 jobs where stored title ≠ labeled title)
4. ⬜ Deduplicate training data
5. ⬜ Re-label all training data with V6 prompt
6. ⬜ Re-label eval set with V6 prompt
7. ⬜ Verify re-labeling (checklist in Step 5.5)
8. ⬜ Prune trivially easy bad_fit
9. ⬜ Generate contrastive data (140 jobs)
10. ⬜ Pre-train gates
11. ⬜ Train (1000 iters, fresh LoRA)
12. ⬜ Evaluate and decide

## Architecture

### Semantic Token Vocabulary

**loc (4 tokens):**
| Token | Score |
|-------|-------|
| LONDON_OR_REMOTE | 25 |
| UK_OTHER | 10 |
| OUTSIDE_UK | -50 |
| MISSING | 0 |

**role (3 tokens):**
| Token | Score |
|-------|-------|
| SENIOR_PLUS | 25 |
| MID_LEVEL | 15 |
| NO_SENIORITY | 0 |

**tech (8 tokens):**
| Token | Score |
|-------|-------|
| NONE | 0 |
| JS_TS | 5 |
| NODE | 10 |
| NODE_JS_TS | 15 |
| AI_ML | 10 |
| JS_TS_AI_ML | 15 |
| NODE_AI_ML | 20 |
| NODE_JS_TS_AI_ML | 25 |

**comp (6 tokens):**
| Token | Score |
|-------|-------|
| NO_GBP | 0 |
| UP_TO_ONLY | 0 |
| BELOW_45K | -30 |
| RANGE_55_74K | 5 |
| RANGE_75_99K | 15 |
| ABOVE_100K | 25 |

### Score and Label Computation (in code, NOT in model)
```python
score = max(0, min(100, loc + role + tech + comp))
if score >= 70: label = "good_fit"
elif score >= 50: label = "maybe"
else: label = "bad_fit"
```

### Student Prompt (used in training data user message AND at inference)
```
Classify this job. Output JSON only.

Title: {{job_title}}
Location: {{job_location}}
Description: {{jd_text}}

Valid tokens:
loc: LONDON_OR_REMOTE | UK_OTHER | OUTSIDE_UK | MISSING
role: SENIOR_PLUS | MID_LEVEL | NO_SENIORITY
tech: NONE | JS_TS | NODE | NODE_JS_TS | AI_ML | JS_TS_AI_ML | NODE_AI_ML | NODE_JS_TS_AI_ML
comp: NO_GBP | UP_TO_ONLY | BELOW_45K | RANGE_55_74K | RANGE_75_99K | ABOVE_100K

{"reasoning":"...","loc":"...","role":"...","tech":"...","comp":"..."}
```

### Model Output Example
```json
{"reasoning":"loc: 'London, England' → London found → LONDON_OR_REMOTE. role: 'Senior' in title → SENIOR_PLUS. tech: node.js found, typescript found, react ignored → NODE_JS_TS. comp: '£85,000-£120,000' midpoint £102.5k → ABOVE_100K.","loc":"LONDON_OR_REMOTE","role":"SENIOR_PLUS","tech":"NODE_JS_TS","comp":"ABOVE_100K"}
```

### Evaluation Rules
- Score ONLY on loc, role, tech, comp tokens and code-computed label
- NEVER evaluate reasoning content
- Invalid tokens count as wrong for that field

## Key Files

| File | Purpose |
|------|---------|
| data/v5/train_800.jsonl | Current training set (895 jobs, 806 train / 89 valid) |
| data/v5/eval_150_golden.jsonl | Locked eval set (150 jobs, 50/50/50 by label, SHA-256 recorded) |
| data/v5/full_pool.jsonl | Full labeled pool (~1750 jobs) |
| prompts/teacher_v5.txt | V5 teacher prompt (superseded by V6) |
| prompts/teacher_v6.txt | **V6 teacher prompt** — engineering gate, COMP ignore rules, AI/ML keywords |
| prompts/student_v5.txt | Student prompt for training and inference |
| V6_STUDENT_TRAINING_PLAN.md | **V6 execution plan** — 12-step sequence with status tracking |
| V6_DIAGNOSTIC_FINDINGS.md | **V6 findings history** — 20 findings with decisions and lessons |
| finetune/eval_student.py | Eval script (fixed adapter loading bug — copies checkpoint to adapters.safetensors) |
| finetune/lora_config_v5.yaml | LoRA training config |
| finetune/adapters_v5.1/ | Current best adapters (875 checkpoint) |
| src/cli/label-jobs.ts | Labels jobs with GPT-4o-mini teacher prompt |
| src/cli/analyze-student-eval.ts | Generates detailed eval reports |
| src/cli/build-datasets.ts | Builds train/valid/eval splits |
| src/cli/generate-synthetic-recipes.ts | Generates synthetic jobs from recipes |
| src/cli/audit-training-data.ts | **Data quality audit** — duplicates, contamination, token validation, title mismatches |
| TRAINING_PLAN_V5.md | The full V5 plan |
| TRAINING_PLAN_ITERATIONS.md | Decision log from V1→V5 |
| CONTRASTIVE_GENERATION_PROMPTS.md | Generation prompts for the 10 contrastive batches |
| SCRAPING_INSTRUCTIONS_PATCH_V5.md | Scraping instructions for comp/tech data |

## Training Config

| Parameter | Value |
|-----------|-------|
| Model | mlx-community/Qwen2.5-0.5B-Instruct-4bit |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.1 |
| Learning rate | 5e-5 |
| Batch size | 1 (changed from 2 to prevent OOM) |
| Grad accumulation | 16 (effective batch = 16) |
| Iters | 1000 (for next round) |
| Warmup | 50 |
| Eval every | 50 |
| Save every | 50 |
| mask_prompt | true |
| grad_checkpoint | true |

## Known Problems and Patterns

### Comp Degradation Pattern
Comp accuracy DECREASES in later training iterations while loc/role improves:
```
Checkpoint 450: comp 77.3%, role 90.8%
Checkpoint 600: comp 65.4%, role 94.7%
Checkpoint 850: comp 78.5%, role 92.6%
```
The model trades comp accuracy for role/role accuracy in later iters because the training set is 63% bad_fit (where comp rarely matters for label). The model learns a shortcut: perfect loc/role on easy jobs rather than learning comp discrimination on hard jobs.

### ABOVE_100K Bias (v5.0, partially fixed in v5.1)
V5.0 had 36 of 46 comp errors defaulting to ABOVE_100K. Caused by synthetic good_fit recipes heavily featuring ABOVE_100K. Partially fixed in v5.1 by adding more RANGE_75_99K and RANGE_55_74K data. V5.1 comp errors are scattered rather than biased to one token.

### Invalid Token Invention (fixed)
V5.0 had 12 invalid tokens (NODE_JS_TS_PG, Golang_GRPC, etc). Fixed by adding valid token list to student prompt. V5.1 has only 1 invalid token.

### Parse Failures on Garbage Locations (fixed)
V5.0 had 5 parse failures from JDs with company descriptions pasted into location field. Fixed by adding 20 garbage-location training examples. V5.1 has 0 parse failures.

### Never Continue Training from Existing Adapter
Training on top of an existing adapter produces much worse results than fresh LoRA from base. The old weights fight the new data. Always train fresh.

### V5.1 Eval Results (875 checkpoint — current best)
```
Label accuracy:   125/149 = 83.9%
Valid outputs:    149/150

Field token accuracy:
  loc:   138/149 = 92.6%
  role:  138/149 = 92.6%
  tech:  108/149 = 72.5%
  comp:  117/149 = 78.5%

Per-label breakdown:
  good_fit: 38/49 = 78%
  maybe:    39/50 = 78%
  bad_fit:  48/50 = 96%

Confusion Matrix:
  Golden\Pred   good_fit  maybe  bad_fit
  good_fit           38     11       0
  maybe               5     39       6
  bad_fit             0      2      48

Top comp errors: scattered (no single dominant bias)
Top tech errors: AI_ML→NONE (7), JS_TS→NONE (6), NODE_JS_TS→JS_TS (4)
```

### 24 Label Errors Breakdown
- good_fit→maybe: 11 (model under-scores good jobs)
- maybe→good_fit: 5 (model over-scores borderline)
- maybe→bad_fit: 6 (model under-scores borderline)
- bad_fit→maybe: 2 (model over-scores bad jobs)
- good_fit→bad_fit: 0
- bad_fit→good_fit: 0

## Contrastive Training Batch Design

140 jobs across 10 batches. Each contrastive set takes a base JD and creates variants by changing ONLY one field. Variants must be created programmatically (regex/string replacement) not via GPT rephrasing.

| Batch | Purpose | Count |
|-------|---------|-------|
| A | Comp amount (same JD, salary changes) | 15 |
| B | Tech presence (same JD, tech stack changes) | 15 |
| C | Location + Remote formats | 19 |
| D | AI/ML required vs mentioned vs nice-to-have | 15 |
| E | Role titles + edge cases (Snr, III, Founding) | 20 |
| F | Comp format + currency (GBP vs USD vs TC/OTE) | 15 |
| G | Garbage location fields → MISSING | 10 |
| H | Double ambiguity (tech AND comp change) | 6 |
| J | Red herrings and traps | 25 |

Batch I (format variance) was removed — GPT-4o-mini labeling may produce inconsistent tokens for reformatted text.

Full generation prompts are in CONTRASTIVE_GENERATION_PROMPTS.md.

## Pruning Criteria

Remove trivially easy bad_fit jobs where ALL of:
- loc = OUTSIDE_UK or MISSING
- role = NO_SENIORITY
- tech = NONE
- comp = NO_GBP

These teach the model nothing it doesn't already know (96% bad_fit accuracy). Removing ~100-150 rebalances the dataset so the model spends more time on boundary cases.

## Critical Rules
1. GPT-4o-mini is the source of truth — never question its labels
2. Temperature=0 for ALL labeling (consistency > creativity)
3. Temperature=0.7 for generating synthetic JD text (variety needed)
4. Fresh LoRA from base for every training round — never continue from existing adapter
5. Eval scores only on field tokens and code-computed label — never on reasoning
6. Contrastive variants must be created programmatically, not by GPT rephrasing
7. Eval set is locked after creation — `build-datasets.ts` auto-applies chmod 444. V5 eval set (eval_150_golden.jsonl) was **lost on 2026-03-07** — V6 will build a new one
8. Check the teacher prompt temperature setting — if the original 895 labels used temperature > 0, that may explain inconsistencies
9. **NEVER test write-capable scripts with default output paths** — always pass explicit `--eval-output /tmp/test_eval.jsonl --train-output /tmp/test_train.jsonl` or similar safe paths when testing scripts that produce output files
10. **NEVER pipe write-capable scripts through `head`, `tail`, or `| head -N`** — SIGPIPE can kill the process after files are truncated but before data is written, zeroing them. Redirect to a file instead: `command > /tmp/output.txt`
11. **Before running any script that takes `--input`**, verify the input file exists and is non-empty: `wc -l <file>` first

## Data Pipeline
```
Scrape/Generate raw jobs → Label with GPT-4o-mini (temperature=0) → Validate tokens → Build train/valid/eval splits → Format as chat JSONL → Train LoRA → Eval on locked eval set
```

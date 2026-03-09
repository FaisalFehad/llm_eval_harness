# Student Model Training Project — V5 Semantic Tokens

## Project Overview

Distilling GPT-4o-mini into a local Qwen2.5-0.5B-Instruct-4bit model for job-fit scoring. The model classifies LinkedIn job postings into semantic tokens across 6 fields (V7 architecture). Code computes the numeric score and label deterministically via a scoring translation layer. GPT-4o-mini is the single source of truth for all labeling — never question its labels.

## Current Status

Best result: **83.9% label accuracy** (v5.1, checkpoint 875, 895 training jobs).
Target: **90-93%** (0.5B ceiling).
Currently in **V7 pre-training** — V7 teacher prompt designed (6 fields, richer tokens, semantic rules). Next: re-label all data with V7 prompt.

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
- **V7 teacher prompt written** (`prompts/teacher_v7.txt`) — 6 fields, semantic rules, richer token vocabulary (Finding 28)
- 28 findings documented (see `V6_DIAGNOSTIC_FINDINGS.md`)
- Spec violations vs spec gaps analysis complete — found 24 issues in 4 categories (14 spec gaps, 7 title mismatches, 2 spec violations, 1 V6-still-wrong)
- Title mismatch investigated (Finding 24) — pipeline is clean, GPT adherence issue, pre/post-label audit catches it
- Audit tooling built (`audit-training-data.ts`) — 6 critical + 7 warning checks, clean mode, pipeline gate
- Validation pipeline built — pre-label mode, post-label checks, per-run logging, Promptfoo V6 config (28 edge case tests)
- Execution order defined (13 steps with status tracking)
- Post-relabeling verification checklist created (Step 5.5)
- Failed V6 experiment: resumed from V5.1 adapter with only 60 H/I/J jobs → 63.5% (invalid — violates fresh LoRA rule)
- Distribution analysis complete — all NODE gaps filled with 2,543 real + 46 synthetic jobs

### What's In Progress (V6 Execution Order)
1. ✅ Lock rules (teacher_v6.txt → superseded by teacher_v7.txt)
2. ✅ Classify spec violations vs gaps
3. ✅ Build audit tooling + deduplicate training data
4. ✅ Build validation pipeline (pre-label audit, post-label checks, Promptfoo, logging)
4b. ✅ Distribution analysis & gap filling (2,589 jobs in `data/v6/scraped_clean_for_labeling.jsonl`)
4c. ✅ V7 teacher prompt architecture (6 fields, semantic rules, richer tokens — Finding 28)
5. ⬜ **NEXT: Create V7 versions of scripts/prompts** (new files alongside V6 — non-destructive. See Critical Rule #12)
6. ⬜ Re-label all training data with V7 prompt
7. ⬜ Build new eval set (V5 eval set was lost — see Critical Rules #7)
8. ⬜ Verify re-labeling (checklist in Step 5.5)
9. ⬜ Prune trivially easy bad_fit
10. ⬜ Generate contrastive data (140 jobs, batches A-H, J)
11. ⬜ Verify contrastive variants
12. ⬜ Pre-train gates
13. ⬜ Train (1000 iters, fresh LoRA)
14. ⬜ Evaluate and decide

## Architecture

### Semantic Token Vocabulary (V7 — 6 fields)

**location (5 tokens):**
| Token | Score |
|-------|-------|
| IN_LONDON | 25 |
| FULLY_REMOTE | 25 |
| UK_OTHER | 10 |
| OUTSIDE_UK | -50 |
| UNKNOWN | 0 |

**work_arrangement (4 tokens):**
| Token | Score |
|-------|-------|
| REMOTE | 0 |
| HYBRID | 0 |
| IN_OFFICE | 0 |
| UNKNOWN | 0 |
*(Informational — scores 0 today, could be scored later without re-labeling)*

**scope (2 tokens):**
| Token | Score |
|-------|-------|
| IN_SCOPE | *(gate — use seniority score)* |
| OUT_OF_SCOPE | 0 *(forces seniority=0)* |

**seniority (3 tokens):**
| Token | Score |
|-------|-------|
| LEVEL_3 | 25 |
| LEVEL_2 | 15 |
| LEVEL_1 | 0 |

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
# V7 scoring translation layer
loc_score = V7_LOCATION_SCORES[pred['location']]  # IN_LONDON=25, FULLY_REMOTE=25, etc.
role_score = 0 if pred['scope'] == 'OUT_OF_SCOPE' else V7_SENIORITY_SCORES[pred['seniority']]
tech_score = V7_TECH_SCORES[pred['tech']]
comp_score = V7_COMP_SCORES[pred['comp']]

score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
if score >= 70: label = "good_fit"
elif score >= 50: label = "maybe"
else: label = "bad_fit"
```

### Student Prompt (V7 — used in training data user message AND at inference)
```
Classify this job. Output JSON only.

location: IN_LONDON | FULLY_REMOTE | UK_OTHER | OUTSIDE_UK | UNKNOWN
work_arrangement: REMOTE | HYBRID | IN_OFFICE | UNKNOWN
scope: IN_SCOPE | OUT_OF_SCOPE
seniority: LEVEL_1 | LEVEL_2 | LEVEL_3
tech: NONE | JS_TS | NODE | NODE_JS_TS | AI_ML | JS_TS_AI_ML | NODE_AI_ML | NODE_JS_TS_AI_ML
comp: NO_GBP | UP_TO_ONLY | BELOW_45K | RANGE_55_74K | RANGE_75_99K | ABOVE_100K

Title: {{job_title}}
Location: {{job_location}}
Description: {{jd_text}}

{"location_reason":"...","location":"...","work_arrangement_reason":"...","work_arrangement":"...","scope_reason":"...","scope":"...","seniority_reason":"...","seniority":"...","tech_reason":"...","tech":"...","comp_reason":"...","comp":"..."}
```

### Model Output Example
```json
{"location_reason":"London -> IN_LONDON","location":"IN_LONDON","work_arrangement_reason":"hybrid, 3 days in office -> HYBRID","work_arrangement":"HYBRID","scope_reason":"Senior Software Engineer -> IN_SCOPE","scope":"IN_SCOPE","seniority_reason":"Senior -> LEVEL_3","seniority":"LEVEL_3","tech_reason":"node.js, typescript; react ignored -> NODE_JS_TS","tech":"NODE_JS_TS","comp_reason":"£85k-£120k mid £102.5k -> ABOVE_100K","comp":"ABOVE_100K"}
```

### Evaluation Rules
- Score ONLY on 6 token fields (location, work_arrangement, scope, seniority, tech, comp) and code-computed label
- NEVER evaluate reasoning content or _reason fields
- Invalid tokens count as wrong for that field
- V7 tokens are translated to scores via translation layer before label computation

### Comparing V7 Results to V5.1 Historical Baselines

**Label accuracy**: Directly comparable. V7 translation layer produces the same scores and uses the same label thresholds (≥70=good_fit, ≥50=maybe, <50=bad_fit) as V5.1. V5.1 best = **83.9%** (checkpoint 875).

**Field-level accuracy**: Compare at the **score level**, not the token level, since V7 has different tokens:
| V5.1 field | V5.1 baseline | V7 equivalent | How to compare |
|------------|---------------|---------------|----------------|
| loc (92.6%) | 4 tokens | location (5 tokens) | Map V7 location → loc_score (IN_LONDON/FULLY_REMOTE→25, UK_OTHER→10, OUTSIDE_UK→-50, UNKNOWN→0) and compare score accuracy |
| role (92.6%) | 3 tokens | scope + seniority | Compute role_score = 0 if OUT_OF_SCOPE else seniority_score, compare score accuracy |
| tech (72.5%) | 8 tokens | tech (8 tokens) | Same tokens — directly comparable |
| comp (78.5%) | 6 tokens | comp (6 tokens) | Same tokens — directly comparable |

**Caveat**: The V5 eval set was lost (2026-03-07). V7 will use a new eval set, so numbers are approximate comparisons, not exact apples-to-apples. The comparison shows whether the model is in the same performance range, not whether a specific percentage point changed.

## Key Files

| File | Purpose |
|------|---------|
| data/v5/train_860.jsonl | Original V5.1 training data (895 jobs, chmod 444). train_800.jsonl was lost. |
| data/v5/eval_150_golden.jsonl | **LOST on 2026-03-07** — V6 will build a new eval set |
| data/v5/all_labeled_pool.jsonl | Full labeled pool (1522 jobs, chmod 444) |
| prompts/teacher_v5.txt | V5 teacher prompt (superseded by V6) |
| prompts/teacher_v6.txt | V6 teacher prompt (superseded by V7) |
| prompts/teacher_v7.txt | **V7 teacher prompt** — 6 fields, semantic rules, richer token vocabulary, 5 examples |
| prompts/student_v5.txt | V5 student prompt (superseded by V7) |
| prompts/student_v6.txt | V6 student prompt (superseded by V7) |
| data/v6/scraped_clean_for_labeling.jsonl | **V6 new data** — 2,589 jobs (2,543 real + 46 synthetic NODE/NODE_AI_ML variants) ready for labeling |
| data/v6/synthetic/node_variants.jsonl | 32 NODE tech-swap variants (from real Python/Java/Go backend jobs) |
| data/v6/synthetic/node_ai_ml_variants.jsonl | 14 NODE_AI_ML tech-swap variants |
| V6_STUDENT_TRAINING_PLAN.md | **V6 execution plan** — 13-step sequence with status tracking |
| V6_DIAGNOSTIC_FINDINGS.md | **V6/V7 findings history** — 28 findings with decisions and lessons |
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
- location = OUTSIDE_UK or UNKNOWN
- scope = OUT_OF_SCOPE or seniority = LEVEL_1
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
12. **Non-destructive versioning** — Never modify existing versioned files (scripts, prompts, configs). Create new versions alongside them (e.g., `student_v7.txt` not editing `student_v6.txt`, `semantic_tokens_v7.py` not editing `semantic_tokens.py`). Previous versions must remain runnable so we can reproduce old results. The training plan (`V6_STUDENT_TRAINING_PLAN.md`) is the source of truth — only amend (add notes, strikeouts, status updates), never delete content.

## Data Pipeline
```
Scrape/Generate raw jobs → Label with GPT-4o-mini (temperature=0) → Validate tokens → Build train/valid/eval splits → Format as chat JSONL → Train LoRA → Eval on locked eval set
```

## Common Commands

```bash
# Label jobs with teacher prompt
npx tsx src/cli/label-jobs.ts --input <input.jsonl> --output <output.jsonl> --prompt prompts/teacher_v7.txt

# Audit training data (pre-label or post-label)
npx tsx src/cli/audit-training-data.ts --input <file.jsonl> [--pre-label] [--eval-set <eval.jsonl>]
npx tsx src/cli/audit-training-data.ts --input <file.jsonl> --clean --output <clean.jsonl> [--remove-trivial]

# Build train/valid/eval splits
npx tsx src/cli/build-datasets.ts --input <labeled.jsonl> --train-output <train.jsonl> --eval-output <eval.jsonl>

# Format for MLX training (student prompt needs V7 update)
npx tsx src/cli/format-for-mlx.ts --input <train.jsonl> --output-dir <dir> --prompt prompts/student_v7.txt

# Train LoRA (fresh from base — NEVER resume from existing adapter)
python3 -m mlx_lm.lora --config finetune/lora_config_v7.yaml

# Evaluate checkpoint (needs V7 scoring translation)
python3 finetune/eval_student.py --adapter <adapter.safetensors> --test-file <eval.jsonl> --prompt prompts/student_v7.txt

# Run Promptfoo teacher prompt tests
npx promptfoo eval -c configs/promptfoo_teacher_v7.yaml
```

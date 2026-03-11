# Student Model Training Project — V7 Semantic Tokens

## Project Overview

Distilling gpt-4.1-mini into a local Qwen2.5 student model for job-fit scoring. The model classifies LinkedIn job postings into semantic tokens across 5 fields. Code computes the numeric score and label deterministically via a scoring translation layer (`finetune/semantic_tokens_v7.py`). gpt-4.1-mini is the single source of truth for all labeling — never question its labels.

## Current Status

**V7 training complete. Two models evaluated. Tech accuracy bottleneck under investigation.**

| Model | Label Acc | Tech | Comp | Loc | Parse Fail |
|-------|-----------|------|------|-----|------------|
| 0.5B (2000 iters) | **84.9%** | 70.3% | 71.7% | 95.8% | 15 |
| 1.5B (2000 iters) | **85.4%** | 70.4% | 77.9% | 97.3% | 0 |
| V5.1 baseline | 83.9% | 72.5% | 78.5% | 92.6% | 0 |

**Next step**: Fix tech accuracy (~70% for both models). See Finding 36 in `V6_DIAGNOSTIC_FINDINGS.md`.

### Tracking Documents

| Document | Purpose |
|----------|---------|
| `V6_STUDENT_TRAINING_PLAN.md` | Execution plan with status markers |
| `V6_DIAGNOSTIC_FINDINGS.md` | All findings, decisions, rationale (38 findings) |

**Rules**: After any analysis or decision, update BOTH documents. Never declare something "fixed" without verification.

## Architecture

### Semantic Token Vocabulary (V7 — 5 fields, 10 JSON keys)

Field names: `loc`, `arr`, `sen`, `tech`, `comp`. Each has a `_raw` suffix for verbatim JD text.

Design decisions documented in Findings 28-35 of `V6_DIAGNOSTIC_FINDINGS.md`.

**loc** (5 tokens): IN_LONDON=25, REMOTE=25, UK_OTHER=10, OUTSIDE_UK=-50, UNK=0

**arr** (4 tokens, informational, all score 0): REMOTE, HYBRID, IN_OFFICE, UNK

**sen** (3 tokens): LEVEL_3=25, LEVEL_2=15, LEVEL_1=0

**tech** (array of individual tokens, additive scoring + scope gate):
- Valid tokens: NODE=10, REACT=5, JS_TS=5, AI_ML=10, OOS=0
- Tech is an **array** (e.g. `["NODE", "JS_TS"]`), NOT a combo string
- `["OOS"]` = scope gate, forces `role_score = 0`
- Rules: never mix OOS with real tokens, never empty (use `["OOS"]`), untracked tech in `tech_raw` only

**comp** (7 tokens): NO_GBP=0, UP_TO_ONLY=0, BELOW_45K=-30, RANGE_45_54K=0, RANGE_55_74K=5, RANGE_75_99K=15, ABOVE_100K=25

### Score and Label Computation (in code, NOT in model)
```python
loc_score = LOCATION_MAP[pred["loc"]]
is_oos = "OOS" in pred["tech"] or len(pred["tech"]) == 0
role_score = 0 if is_oos else SENIORITY_MAP[pred["sen"]]
tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP[t] for t in pred["tech"])
comp_score = COMP_MAP[pred["comp"]]
score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
# >=70 good_fit, >=50 maybe, <50 bad_fit
```

### Student Prompt (`prompts/student_v7.txt`)
```
Classify this job. Output JSON only.
loc: IN_LONDON | REMOTE | UK_OTHER | OUTSIDE_UK | UNK
arr: REMOTE | HYBRID | IN_OFFICE | UNK
sen: LEVEL_3 | LEVEL_2 | LEVEL_1
tech: array of [NODE, REACT, JS_TS, AI_ML] or [OOS]
comp: NO_GBP | UP_TO_ONLY | BELOW_45K | RANGE_45_54K | RANGE_55_74K | RANGE_75_99K | ABOVE_100K
Title: {{job_title}}
Location: {{job_location}}
Description: {{jd_text}}
{"loc_raw":"...","loc":"...","arr_raw":"...","arr":"...","sen_raw":"...","sen":"...","tech_raw":"...","tech":[],"comp_raw":"...","comp":"..."}
```

### Evaluation Rules
- Score ONLY on 5 token fields and code-computed label — NEVER `_raw` content
- Invalid tokens count as wrong
- Tech uses exact array match (all tokens must match)

## Key Files

### V7 (current)
| File | Purpose |
|------|---------|
| prompts/teacher_v7.txt | Teacher prompt — 5 fields, semantic rules, 3 examples |
| prompts/student_v7.txt | Student prompt — training + inference |
| finetune/semantic_tokens_v7.py | Token vocab, scoring, validation, fuzzy matching |
| finetune/eval_student_v7.py | Eval script |
| finetune/compare_evals.py | Side-by-side eval comparison |
| finetune/lora_config_v7.yaml | 0.5B training config |
| finetune/lora_config_v7_1.5B.yaml | 1.5B training config |
| finetune/adapters_v7/ | 0.5B adapters (50-2000) |
| finetune/adapters_v7_1.5B/ | 1.5B adapters (50-2000) |
| data/v7/mlx/{train,valid}.jsonl | MLX chat-format (642 train, 71 valid) |
| data/v7/test_labeled.jsonl | 239 test jobs (locked, chmod 444) |
| data/v7/train_labeled.jsonl | 713 labeled training jobs |
| src/cli/label-jobs-v7.ts | Labeling with gpt-4.1-mini |
| src/cli/format-for-mlx-v7.ts | MLX formatter |
| src/cli/audit-training-data-v7.ts | Data quality audit |

### Previous Versions (for reproducibility)
| File | Purpose |
|------|---------|
| prompts/{teacher,student}_{v5,v6}.txt | Old prompts |
| finetune/eval_student.py | V5/V6 eval |
| finetune/lora_config_v5.yaml | V5.1 config |
| finetune/adapters_v5.1/ | V5.1 best (875 ckpt, 83.9%) |
| data/v5/{train_860,all_labeled_pool}.jsonl | V5 data (chmod 444) |
| src/cli/{label-jobs,audit-training-data,build-datasets,format-for-mlx}.ts | V5/V6 scripts |

## Training Config

Both V7 models use identical hyperparameters. Rationale for V5→V7 changes in Finding 36.

| Parameter | Value |
|-----------|-------|
| Rank/Alpha/Dropout | 16 / 32 / 0.05 |
| LR / Warmup | 2e-5 / 100 |
| Batch / Grad accum | 1 / 16 (effective=16) |
| Iters | 2000 |
| max_seq_length | 8192 |
| Eval every / Save every | 25 / 50 |
| mask_prompt / grad_checkpoint | true / true |

V5.1 historical: dropout=0.1, LR=5e-5, warmup=50, iters=1000, eval_every=50. Config: `lora_config_v5.yaml`.

## Critical Rules
1. gpt-4.1-mini is the source of truth — never question its labels
2. Temperature=0 for labeling, 0.7 for synthetic JD generation
3. Fresh LoRA from base every round — never resume from existing adapter
4. Eval on token fields + code-computed label only — never `_raw` fields
5. Contrastive variants: programmatic only, not GPT rephrasing
6. Eval set locked (chmod 444). V5 eval lost 2026-03-07, V7 has new 239-job test set
7. **NEVER** test write scripts with default output paths — use explicit safe paths
8. **NEVER** pipe write scripts through `head`/`tail` — SIGPIPE destroys files
9. Verify input files exist before running scripts: `wc -l <file>`
10. **Non-destructive versioning** — new versions alongside old, never modify existing versioned files

## Data Pipeline
```
Scrape/Generate → Pre-label audit → Label (gpt-4.1-mini, temp=0) → Post-label audit → Split → Format MLX → Train → Eval
```

## Common Commands

```bash
# Label
npx tsx src/cli/label-jobs-v7.ts --input <in.jsonl> --output <out.jsonl>

# Audit
npx tsx src/cli/audit-training-data-v7.ts --input <file.jsonl> [--pre-label] [--eval-set <eval.jsonl>]

# Format for MLX
npx tsx src/cli/format-for-mlx-v7.ts --input <train.jsonl> --output-dir <dir> --prompt prompts/student_v7.txt

# Train (fresh from base — NEVER resume)
python3 -m mlx_lm.lora --config finetune/lora_config_v7.yaml      # 0.5B
python3 -m mlx_lm.lora --config finetune/lora_config_v7_1.5B.yaml  # 1.5B

# Eval
python3 finetune/eval_student_v7.py --adapter finetune/adapters_v7/<ckpt>.safetensors --test-file data/v7/test_labeled.jsonl --prompt prompts/student_v7.txt

# Compare evals
python3 finetune/compare_evals.py <summary1.json> <summary2.json>
```

### V5/V6 Commands (for reproducing old results)
```bash
npx tsx src/cli/label-jobs.ts --input <in.jsonl> --output <out.jsonl> --prompt prompts/teacher_v5.txt
python3 finetune/eval_student.py --adapter finetune/adapters_v5.1/<ckpt>.safetensors --test-file <eval.jsonl> --prompt prompts/student_v5.txt
```

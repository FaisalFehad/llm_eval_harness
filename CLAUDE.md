# Student Model Training Project — V13 Hybrid Pipeline

## Project Overview

Distilling gpt-4.1-mini into a local Qwen3-0.6B student model for job-fit scoring. The model classifies LinkedIn job postings into semantic tokens across 5 fields. A **hybrid pipeline** combines regex (loc/tech/comp) with model inference (sen/arr) for maximum accuracy. Score and label computed deterministically in code via `finetune/semantic_tokens_v7.py`.

## Current Status

**V13.1 in progress — 1.5B Qwen2.5 training. Best 0.6B still V13 at 97.9%.**

| Model | Hybrid Acc | Sen | Arr | Tech | Comp | Parse Fail |
|-------|-----------|-----|-----|------|------|------------|
| **0.6B Qwen3 (V13, iter 1500)** | **97.9%** | **86.6%** | 72.8% | 88.3% | 95.8% | 19* |
| 0.6B Qwen3 (V13.1 corrective, iter 200) | 97.5% | 84.9% | 76.2% | **90.4%†** | **96.2%†** | 13 |
| 0.6B Qwen3 (V12.1) | 97.5% | 84.5% | 80.3% | 87.9% | 95.4% | 0 |
| 1.5B Qwen2.5 (V13.1, in training) | — | — | — | — | — | — |
| 1.5B Qwen2.5 (V12.1) | 98.3% | 90.0% | 90.4% | 87.9% | 95.4% | 8 |

*Parse failures fall back to regex in hybrid pipeline — hybrid accuracy unaffected.
†V13.1 regex (tech=90.4%, comp=96.2%) is an improvement over V13 (88.3%, 95.8%), but error interdependency on Job 14 means the 0.6B corrective retrain doesn't beat V13's 97.9% overall.

**Production model**: Qwen3-0.6B-4bit, adapter `finetune/adapters_v13_0.6B/0001500_adapters.safetensors`

**Next step**: V13.1 1.5B training in progress. V13.1 0.6B corrective retrain peaked at 97.5% (iter 200, all 6 remaining errors are irreducible sen boundary cases). V13.1 regex: tech=90.4% (+2.1pp), comp=96.2% (+0.4pp).

### Tracking Documents

| Document | Purpose |
|----------|---------|
| `docs/V13_PLAN.md` | V13 + V13.1 execution plan |
| `docs/V12_IMPLEMENTATION_PROGRESS.md` | Full V12 history — all phases, models, comparisons |
| `V6_DIAGNOSTIC_FINDINGS.md` | Pre-V12 findings and decisions (38 findings) |

## Architecture

### V12 Hybrid Pipeline (regex + model)

```
Job → Model inference (all 5 fields) → Regex override (loc/tech/comp) → Score → Label
```

| Field | Source | Accuracy | Notes |
|-------|--------|----------|-------|
| loc | **Regex** | 100% | UK city list + non-UK indicators |
| arr | **Model** | 72.8% | Description-based, no regex; arr tokens all score 0 so errors are invisible to hybrid |
| sen | **Model** | 86.6% | Title first, JD description as fallback; regex fallback on parse failure |
| tech | **Regex** | 88.3% | Pattern matching on tech keywords |
| comp | **Regex** | 95.8% | Salary parsing with midpoint calculation |

Regex code: `finetune/deterministic_baseline_v13.py`
Hybrid evaluator: `finetune/compute_hybrid_v13.py`

### Semantic Token Vocabulary (V7 — 5 fields, 10 JSON keys)

Field names: `loc`, `arr`, `sen`, `tech`, `comp`. Each has a `_raw` suffix for verbatim JD text.

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

### Evaluation Rules
- Score ONLY on 5 token fields and code-computed label — NEVER `_raw` content
- Invalid tokens count as wrong
- Tech uses exact array match (all tokens must match)
- **Always eval against audited test set**: `data/v12/test_labeled_audited.jsonl` (not `data/v7/test_labeled.jsonl`)

## Key Files

### V13.1 (current — 1.5B in training)
| File | Purpose |
|------|---------|
| finetune/deterministic_baseline_v13_1.py | **V13.1 regex** (tech=90.4%, comp=96.2% — use for V13.1+ evals) |
| finetune/compute_hybrid_v13_1.py | V13.1 hybrid evaluator (V13.1 regex + model) |
| finetune/deterministic_baseline_v13.py | V13 regex classifier (production — use with V13 model) |
| finetune/compute_hybrid_v13.py | V13 hybrid evaluator (V13 regex + model) |
| finetune/deterministic_baseline_v12_1.py | V12.1 regex classifier (previous) |
| finetune/compute_hybrid_v12_1.py | V12.1 hybrid evaluator (previous) |
| finetune/eval_student_v7.py | Model-only eval script |
| finetune/semantic_tokens_v7.py | Token vocab, scoring, validation, fuzzy matching |
| finetune/compare_evals.py | Side-by-side eval comparison |
| prompts/student_v13_1.txt | **Student prompt — V13.1 (corrective + 1.5B training)** |
| prompts/student_v13.txt | Student prompt — V13 (production 0.6B) |
| prompts/teacher_v7.txt | Teacher prompt — gpt-4.1-mini labeling |

### Configs & Adapters
| File | Purpose |
|------|---------|
| finetune/lora_config_v13_1_1.5B.yaml | **V13.1 1.5B training config (in training)** |
| finetune/adapters_v13_1_1.5B/ | V13.1 1.5B adapters (in training) |
| finetune/lora_config_v13_1_0.6B.yaml | V13.1 0.6B corrective retrain config |
| finetune/adapters_v13_1_0.6B/ | V13.1 0.6B adapters (best: iter 200, 97.5%) |
| finetune/lora_config_v13_0.6B.yaml | V13 0.6B training config (production) |
| finetune/adapters_v13_0.6B/ | **V13 0.6B adapters (best: iter 1500, 97.9%)** |
| finetune/lora_config_v12_qwen3_0.6B.yaml | V12 0.6B training config (previous) |
| finetune/adapters_v12_qwen3_0.6B/ | V12 0.6B adapters (best: iter 1400) |
| finetune/adapters_v12/ | V12.1 1.5B adapters (best: iter 2000, 98.3%) |

### Data
| File | Purpose |
|------|---------|
| data/v12/test_labeled_audited.jsonl | **239 test jobs — USE THIS for all V12+ evals** |
| data/v13_1/mlx/{train,valid}.jsonl | **V13.1 MLX training data (774 train / 86 valid)** |
| data/v13_1/train_merged.jsonl | V13.1 training data (860 jobs: 842 V13 + 18 contrastive) |
| data/v13/train_merged.jsonl | V13 training data (842 jobs: 790 original + 52 contrastive) |
| data/v13/mlx/{train,valid}.jsonl | V13 MLX training data (757 train / 85 valid) |
| data/v13/contrastive.jsonl | 52 contrastive examples (37 sen + 15 loc) |
| data/v12/train_labeled.jsonl | 790 original training jobs |

### Eval Results
| Directory | Purpose |
|-----------|---------|
| eval_results/v13_1_sweep/ | **V13.1 0.6B sweep (8 checkpoints, iter 50–400, best: iter 200 at 97.5%)** |
| eval_results/v13_sweep/ | V13 0.6B sweep (9 checkpoints, iter 1500–1900, best: iter 1500 at 97.9%) |
| eval_results/v13_think_compare/ | Thinking experiment results (think ON/OFF comparison) |
| eval_results/v12_1/ | V12.1 hybrid results (previous production) |
| eval_results/v12_qwen3_0.6B/ | V12 0.6B results |

### Previous Versions (for reproducibility)
| File | Purpose |
|------|---------|
| finetune/adapters_v7/ | V7 0.5B adapters |
| finetune/adapters_v7_1.5B/ | V7 1.5B adapters |
| finetune/adapters_v5.1/ | V5.1 best (875 ckpt, 83.9%) |
| prompts/{teacher,student}_{v5,v6}.txt | Old prompts |
| finetune/eval_student.py | V5/V6 eval |

## Training Config

### V13.1 1.5B (in training)
| Parameter | Value |
|-----------|-------|
| Model | mlx-community/Qwen2.5-1.5B-Instruct-4bit |
| Optimizer | AdamW (weight_decay=0.01) |
| Rank/Alpha/Dropout | 16 / 32 / 0.05 |
| LR / Warmup | 2e-5 / 100 |
| Batch / Grad accum | 1 / 16 (effective=16) |
| Iters | 2000 (fresh from base) |
| max_seq_length | 8192 |
| Eval every / Save every | 100 / 200 |
| mask_prompt / grad_checkpoint | true / true |
| Training data | 860 jobs (842 V13 + 18 contrastive) |

### V13.1 0.6B corrective (done)
| Parameter | Value |
|-----------|-------|
| Model | mlx-community/Qwen3-0.6B-4bit |
| Resume from | V13 iter 1500 |
| LR / Warmup | 2e-6 (10x lower) / 50 |
| Iters | 400 (best: iter 200 at 97.5%) |
| Training data | 860 jobs (842 V13 + 18 contrastive) |
| Best checkpoint | iter 200 (97.5%, 13 parse fails) |
| Remaining errors | 6, all pure sen L2/L3 boundary (irreducible) |

### V13 0.6B (production)
| Parameter | Value |
|-----------|-------|
| Model | mlx-community/Qwen3-0.6B-4bit |
| Optimizer | AdamW (weight_decay=0.01) |
| Rank/Alpha/Dropout | 16 / 32 / 0.05 |
| LR / Warmup | 2e-5 / 100 |
| Batch / Grad accum | 1 / 16 (effective=16) |
| Iters | 1900 (stopped at plateau; planned 3000) |
| max_seq_length | 8192 |
| Eval every / Save every | 25 / 50 |
| mask_prompt / grad_checkpoint | true / true |
| Best checkpoint | iter 1500 (97.9% hybrid) |
| Training data | 842 jobs (790 original + 52 contrastive) |

## Critical Rules
1. gpt-4.1-mini is the source of truth — never question its labels (exception: clear arithmetic errors or data augmentation artifacts)
2. Temperature=0 for labeling, 0.7 for synthetic JD generation
3. **Corrective retraining allowed** — resume from best adapter with 10x lower LR. Fresh-from-base for major data/prompt changes only.
4. Eval on token fields + code-computed label only — never `_raw` fields
5. **Always eval against audited test set** (`data/v12/test_labeled_audited.jsonl`)
6. Eval set locked (chmod 444). V5 eval lost 2026-03-07, V7 has 239-job test set
7. **NEVER** test write scripts with default output paths — use explicit safe paths
8. **NEVER** pipe write scripts through `head`/`tail` — SIGPIPE destroys files
9. Verify input files exist before running scripts: `wc -l <file>`
10. **Non-destructive versioning** — new versions alongside old, never modify existing versioned files

## Common Commands

```bash
# ── V13.1 Hybrid Eval (current workflow) ──────────────────────────

# Model-only eval (V13.1 1.5B — use student_v13_1.txt prompt)
.venv/bin/python3 finetune/eval_student_v7.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter finetune/adapters_v13_1_1.5B/<ITER>_adapters.safetensors \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v13_1.txt \
  --output-dir eval_results/v13_1_1.5B/ \
  --save-predictions

# Hybrid accuracy (V13.1 regex + model predictions)
.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions <predictions.jsonl> \
  --v12 \
  --output <output.json>

# V13.1 regex-only baseline
.venv/bin/python3 finetune/deterministic_baseline_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl

# ── V13 Hybrid Eval (production 0.6B) ─────────────────────────────

# Model-only eval (V13 — use student_v13.txt prompt)
.venv/bin/python3 finetune/eval_student_v7.py \
  --model mlx-community/Qwen3-0.6B-4bit \
  --adapter finetune/adapters_v13_0.6B/0001500_adapters.safetensors \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v13.txt \
  --output-dir eval_results/v13/ \
  --save-predictions

# Checkpoint sweep (score multiple adapters at once)
.venv/bin/python3 finetune/sweep_v13.py --iters 1500 1600 1800 1900 --skip-existing

# ── Training ─────────────────────────────────────────────────────

# V13.1 1.5B fresh from base
.venv/bin/python3 -m mlx_lm.lora --config finetune/lora_config_v13_1_1.5B.yaml 2>&1 | tee training_v13_1_1.5B.log

# V13 0.6B fresh from base
.venv/bin/python3 -m mlx_lm.lora --config finetune/lora_config_v13_0.6B.yaml 2>&1 | tee training_v13.log

# ── Data Pipeline ────────────────────────────────────────────────

# Label
npx tsx src/cli/label-jobs-v7.ts --input <in.jsonl> --output <out.jsonl>

# Audit
npx tsx src/cli/audit-training-data-v7.ts --input <file.jsonl> [--pre-label] [--eval-set <eval.jsonl>]

# Format for MLX
npx tsx src/cli/format-for-mlx-v7.ts --input <train.jsonl> --output-dir <dir> --prompt prompts/student_v13.txt

# Compare evals
.venv/bin/python3 finetune/compare_evals.py <summary1.json> <summary2.json>
```

## Gotchas

- **Qwen3 needs `--model` flag**: `eval_student_v7.py` defaults to Qwen2.5-0.5B. Pass `--model mlx-community/Qwen3-0.6B-4bit` for 0.6B evals
- **Audited vs original test set**: `data/v12/test_labeled_audited.jsonl` has 3 golden corrections from V12 Phase 0B. Always use this for V12+ evals
- **Prompt is baked into training data**: Fine-tuned models can't use different prompts at inference — the prompt is part of the learned input distribution
- **One model at a time**: M1 16GB can only run one model for eval (GPU memory exclusive). OOM if two run concurrently
- **Thermal throttling**: Sustained evals (30+ min) slow ~2x on M1. Budget 15-20 min per 239-job eval
- **MLX LoRA resume**: Loads adapter weights but resets optimizer state (Adam m/v) and iter counter. Warmup replays from zero. Works fine in practice.
- **V13.1 regex is better but not always paired with V13.1 model**: V13.1 regex (tech=90.4%, comp=96.2%) is strictly better than V13 regex. However, V13 model + V13.1 regex = 97.5% (worse than V13 model + V13 regex = 97.9%) due to error interdependency on Job 14 (OTE comp error was accidentally cancelling a tech error). Always pair V13.1 regex with V13.1 model.
- **V13 regex imports**: `compute_hybrid_v13.py` imports from `deterministic_baseline_v13.py`; `compute_hybrid_v13_1.py` imports from `deterministic_baseline_v13_1.py` — keep pairs in `finetune/`
- **Model-only scores are misleading for hybrid**: Model-only tech dropped 20pp V12→V13 but hybrid tech improved — regex overrides model for loc/tech/comp so model-only accuracy on those fields is irrelevant. Always compare hybrid-to-hybrid, never model-only-to-hybrid.
- **`eval_student_v7.py` extra flags**: `--no-think` disables Qwen3 thinking mode (faster but more parse failures); `--max-tokens N` raises token budget (3000+ causes 55s/job on M1 — impractical). Keep defaults.

## Data Pipeline
```
Scrape/Generate → Pre-label audit → Label (gpt-4.1-mini, temp=0) → Post-label audit → Split → Format MLX → Train → Eval (hybrid)
```

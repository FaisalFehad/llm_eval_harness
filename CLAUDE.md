# Student Model Training Project — V13 Hybrid Pipeline

## Project Overview

Distilling gpt-4.1-mini into a local Qwen3-0.6B student model for job-fit scoring. The model classifies LinkedIn job postings into semantic tokens across 5 fields. A **hybrid pipeline** combines regex (loc/tech/comp) with model inference (sen/arr) for maximum accuracy. Score and label computed deterministically in code via `finetune/semantic_tokens_v7.py`.

## Current Status

**V14 COMPLETE. Best model: Qwen3-4B at 98.7% hybrid (new all-time record). Mac deployment: Q6_K GGUF at 98.3% hybrid / 83.7% model-only (3.1 GB).**

| Model | Format | Hybrid | Model-only | Parse | Size | Speed (hardware) |
|-------|--------|--------|-----------|-------|------|-----------------|
| **Qwen3-4B V14 (step 800)** | bfloat16 HF | **98.7%** | 86.2% | 5 | ~8 GB | — (Lambda only) |
| Qwen3-4B V14 F16 GGUF | GGUF F16 | 98.7% | 86.2% | 8 | 7.5 GB | 0.7s/job (Lambda GH200) |
| **Qwen3-4B V14 Q6_K GGUF** | GGUF Q6_K | **98.3%** | **83.7%** | 10 | **3.1 GB** ✅ Mac | 0.7s/job (Lambda GH200) / ❓ Mac untested |
| Qwen3-4B V14 MLX 6-bit | MLX 6-bit | TBD | TBD | TBD | ~3.1 GB ⏳ eval running | ~25s/job (Mac M1, thinking ON) |
| Qwen3-4B V14 Q4_K_M GGUF | GGUF Q4_K_M | 97.9% | 62.3% | 51 | 2.3 GB | 0.7s/job (Lambda GH200) / ❓ Mac untested |
| Qwen3-4B V14 MLX 4-bit | MLX 4-bit | TBD | TBD | TBD | ~2.3 GB ⏳ pending | ❓ untested |
| Qwen3-0.6B V14 (step 2000) | HF 4-bit | 97.9% | 49.4% | 83 | 335 MB |
| Qwen2.5-1.5B V14 (step 1200) | HF 4-bit | 96.7% | 41.4% | 116 | 839 MB |
| V13 0.6B MLX (iter 1500) | MLX 4-bit | 97.9% | — | 19 | 335 MB |
| V13.1 1.5B MLX (iter 1800) | MLX 4-bit | 97.5% | — | 36 | 839 MB |
| V12.1 1.5B MLX (iter 2000) | MLX 4-bit | 98.3% | — | 8 | 839 MB |

**Production model (Mac)**: `~/qwen3_4B_v14_Q6_K.gguf` (3.1 GB, downloaded)
**HuggingFace archive**: `FF-01/qwen3-4b-v14` (private) — all GGUFs + merged_v14_4B. Lambda deleted 2026-03-19.
**Next steps**: MLX 4-bit and 6-bit eval pending (download `merged_v14_4B` from HF → convert on Mac). V15 priorities in `docs/V14_IMPLEMENTATION_PROGRESS.md`.

### Tracking Documents

| Document | Purpose |
|----------|---------|
| `docs/V14_IMPLEMENTATION_PROGRESS.md` | **V14 full history — 4B training, sweep, quantization comparison, GGUF debugging** |
| `docs/V14_REPRODUCTION_GUIDE.md` | **Step-by-step reproduction guide — all commands to redo any V14 step** |
| `docs/lambda_logs/` | All Lambda training/eval/conversion logs + imatrix calibration data |
| `docs/V13_1_IMPLEMENTATION_PROGRESS.md` | V13.1 full history — regex, 0.6B corrective, 1.5B training, error analysis |
| `docs/V13_PLAN.md` | V13 execution plan (V13 phases 1–4 complete) |
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

### V14 (current — complete)
| File | Purpose |
|------|---------|
| finetune/train_v14.py | V14 training script (Unsloth + HuggingFace, Lambda only) |
| finetune/run_v14_training.sh | Full training queue — all 3 models (0.6B, 1.5B, 4B) |
| finetune/sweep_v14.py | V14 checkpoint sweep script |
| finetune/eval_student_v14_gguf.py | GGUF inference eval (llama-cpp-python) |
| finetune/eval_student_v14.py | HF bfloat16 inference eval (Lambda) |
| prompts/student_v14.txt | **V14 student prompt** |
| finetune/adapters_v14_4B/checkpoint-800/ | **Best 4B checkpoint (98.7% hybrid)** |
| finetune/adapters_v14_0.6B/checkpoint-2000/ | Best 0.6B checkpoint (97.9% hybrid) |
| finetune/adapters_v14/checkpoint-1200/ | Best 1.5B checkpoint (96.7% hybrid) |
| ~/merged_v14_4B/ | Merged 4B HF model (~7.6 GB) — source for MLX/GGUF conversion |
| ~/qwen3_4B_v14_Q6_K.gguf | **Mac deployment model (3.1 GB, 98.3% hybrid)** |
| ~/qwen3_4B_v14_Q4_K_M.gguf | Smaller option (2.3 GB, 97.9% hybrid, model-only 62%) |
| ~/qwen3_4B_v14_f16.gguf | Full precision GGUF (7.5 GB, 98.7% hybrid) |
| data/v14/train.jsonl | V14 training data (774 jobs) |
| data/v14/valid.jsonl | V14 validation data (86 jobs) |
| eval_results/v14_sweep_4B_fixed/ | 4B fixed sweep results (best: step 800) |
| eval_results/v14_gguf_Q6_K/ | Q6_K GGUF eval results |
| eval_results/v14_gguf_Q4_K_M/ | Q4_K_M GGUF eval results |
| docs/lambda_logs/imatrix.dat | IQ2 calibration data (for future IQ quantization) |

### V13.1 (current — complete)
| File | Purpose |
|------|---------|
| finetune/deterministic_baseline_v13_1.py | **V13.1 regex** (tech=90.4%, comp=96.2% — use for V13.1+ evals) |
| finetune/compute_hybrid_v13_1.py | V13.1 hybrid evaluator (V13.1 regex + model) |
| finetune/sweep_v13_1_1.5B.py | **V13.1 1.5B checkpoint sweep script** |
| finetune/sweep_v13_1.py | V13.1 0.6B checkpoint sweep script |
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
| finetune/lora_config_v13_1_1.5B.yaml | **V13.1 1.5B training config** |
| finetune/adapters_v13_1_1.5B/ | **V13.1 1.5B adapters (best: iter 1800, 97.5%)** |
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
| eval_results/v14_sweep_4B_fixed/ | **V14 4B fixed sweep (best: step 800, 98.7% hybrid)** |
| eval_results/v14_sweep_0.6B/ | V14 0.6B sweep (best: step 2000, 97.9% hybrid) |
| eval_results/v14_sweep_1.5B/ | V14 1.5B sweep (best: step 1200, 96.7% hybrid) |
| eval_results/v14_gguf_f16/ | 4B F16 GGUF eval (98.7% hybrid, 86.2% model-only) |
| eval_results/v14_gguf_Q6_K/ | 4B Q6_K GGUF eval (98.3% hybrid, 83.7% model-only) |
| eval_results/v14_gguf_Q4_K_M/ | 4B Q4_K_M GGUF eval (97.9% hybrid, 62.3% model-only) |
| eval_results/v14_mlx4bit/ | 4B MLX 4-bit eval ⏳ in progress |
| eval_results/v14_mlx6bit/ | 4B MLX 6-bit eval ⏳ in progress |
| eval_results/v13_1_1.5B_sweep/ | **V13.1 1.5B sweep (10 checkpoints, iter 200–2000, best: iter 1800 at 97.5%)** |
| eval_results/v13_1_sweep/ | V13.1 0.6B sweep (8 checkpoints, iter 50–400, best: iter 200 at 97.5%) |
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

### V13.1 1.5B (complete — best: iter 1800)
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
| Best checkpoint | iter 1800 (97.5% hybrid, sen=90.8%, parse=36) |
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
# ── V14 GGUF Eval (Mac — llama-cpp-python) ────────────────────────

# Q6_K eval (Mac deployment target)
.venv/bin/python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v14_Q6_K.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_gguf_Q6_K_mac

# Hybrid score GGUF predictions
.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions <predictions.jsonl> --v12

# ── V14 MLX Eval (Mac) ────────────────────────────────────────────

# Convert merged HF model to MLX (one-time, run from Mac)
.venv/bin/python3 -m mlx_lm convert \
  --hf-path ~/merged_v14_4B --mlx-path ~/qwen3_4B_v14_mlx4bit -q --q-bits 4
.venv/bin/python3 -m mlx_lm convert \
  --hf-path ~/merged_v14_4B --mlx-path ~/qwen3_4B_v14_mlx6bit -q --q-bits 6

# Eval MLX model
.venv/bin/python3 finetune/eval_student_v7.py \
  --model ~/qwen3_4B_v14_mlx4bit \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_mlx4bit --save-predictions

# ── V13.1 1.5B Hybrid Eval ─────────────────────────────────────────

# Model-only eval (V13.1 1.5B best checkpoint — iter 1800)
.venv/bin/python3 finetune/eval_student_v7.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter finetune/adapters_v13_1_1.5B/0001800_adapters.safetensors \
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

# Checkpoint sweep (1.5B — all 10 checkpoints)
.venv/bin/python3 finetune/sweep_v13_1_1.5B.py --skip-existing

# Checkpoint sweep (0.6B corrective)
.venv/bin/python3 finetune/sweep_v13_1.py --skip-existing

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

- **V14 GGUF: no `chat_format` override**: Remove `chat_format="chatml"` from `Llama()` — it overrides the GGUF's embedded Jinja template, producing garbage output. Let llama-cpp-python auto-detect from GGUF metadata.
- **V14 GGUF: use `/no_think` not pre-fill**: Disable Qwen3 thinking via `/no_think` in system prompt. Pre-filling assistant messages in `create_chat_completion` closes the turn with `<|im_end|>` → zero output.
- **V14 `{{` pre-fill bug**: Original sweep had `{{` in eval script causing 60–70% parse failures. Fixed sweep results in `eval_results/v14_sweep_4B_fixed/`. Never use `v14_sweep_4B/`.
- **Q4_K_M model-only ceiling**: 62.3% due to schema hallucination in weights (wrong field names like `loc_field`). Not fixable at inference — minimum viable quantization for fine-tuned JSON schema is ~6-bit (Q6_K).
- **MLX is Apple Silicon only**: Cannot convert or run MLX models on Lambda (NVIDIA). Conversion must happen on Mac.
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
- **V13.1 1.5B parse failures (36/239 = 15%)**: Qwen2.5-1.5B with V13.1 prompt generates verbose preambles on long JDs before the JSON, consuming the token budget. V12.1 1.5B had only 8. This costs ~3 hybrid labels. V14 fix: add "Begin response with `{`" to prompt end.
- **Val loss ≠ hybrid accuracy for V13.1 1.5B**: Iter 1400 has the best mid-run val loss (0.142) but only 95.0% hybrid due to 67 parse failures. Iter 1800 (val 0.148) beats it with 36 parse failures. Always run the full sweep — don't pick checkpoint by val loss alone.

## Data Pipeline
```
Scrape/Generate → Pre-label audit → Label (gpt-4.1-mini, temp=0) → Post-label audit → Split → Format MLX → Train → Eval (hybrid)
```

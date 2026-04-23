# V15 Runbook — Step-by-Step Execution Commands

All scripts are built and ready. Execute in order.

---

## Pre-requisite

```bash
export OPENAI_API_KEY=sk-...   # Needed for Steps 2-3
```

---

## Step 1: OOS Downsampling (DONE ✅)

Already executed. Output: `data/v15/train_rebalanced.jsonl` (597 examples, OOS 25.1%)

```bash
# To re-run if needed:
python3 finetune/downsample_oos_v15.py \
  --input data/v14/train.jsonl \
  --keep 150 \
  --output data/v15/train_rebalanced.jsonl \
  --report data/v15/oos_downsample_report.json
```

---

## Step 2: Generate Synthetic JDs (needs OPENAI_API_KEY)

```bash
# Dry run first to verify plan:
python3 finetune/generate_v15_data.py \
  --output data/v15/synthetic_raw.jsonl \
  --dry-run

# Full generation (~155 jobs, ~$0.50 in API costs):
python3 finetune/generate_v15_data.py \
  --output data/v15/synthetic_raw.jsonl \
  --model gpt-4.1-mini

# Verify output:
wc -l data/v15/synthetic_raw.jsonl
head -1 data/v15/synthetic_raw.jsonl | python3 -m json.tool
```

---

## Step 3: Label Synthetic Data (needs OPENAI_API_KEY)

```bash
# Label with V15 teacher prompt:
npx tsx src/cli/label-jobs-v7.ts \
  --input data/v15/synthetic_raw.jsonl \
  --output data/v15/synthetic_labeled.jsonl \
  --prompt prompts/teacher_v15.txt \
  --model gpt-4.1-mini \
  --concurrency 10

# Audit labels:
npx tsx src/cli/audit-training-data-v7.ts \
  --input data/v15/synthetic_labeled.jsonl \
  --eval-set data/v12/test_labeled_audited.jsonl

# Verify line count:
wc -l data/v15/synthetic_labeled.jsonl
```

### Manual Audit Checklist
- [ ] Every "Anywhere" job has loc=REMOTE
- [ ] Every Next.js job does NOT have NODE in tech
- [ ] Every secondary AI/ML job has AI_ML in tech
- [ ] Every £60k-£80k range has comp=RANGE_55_74K
- [ ] Every USD/CAD salary has comp=NO_GBP
- [ ] Every "Up to £X" has comp=UP_TO_ONLY

---

## Step 4: Format Synthetic Data for Training

```bash
# Convert labeled data to chat format (same format as training data):
npx tsx src/cli/format-for-mlx-v7.ts \
  --input data/v15/synthetic_labeled.jsonl \
  --output-dir data/v15/synthetic_chat \
  --prompt prompts/student_v15.txt \
  --valid-pct 0 \
  --max-tokens 7500
```

The output will be at `data/v15/synthetic_chat/train.jsonl` (all examples in train, no valid split — we'll split later in merge).

---

## Step 5: Merge Rebalanced + Synthetic Data

```bash
python3 finetune/merge_v15_data.py \
  --base data/v15/train_rebalanced.jsonl \
  --augment data/v15/synthetic_chat/train.jsonl \
  --output-dir data/v15/ \
  --valid-ratio 0.1 \
  --seed 42

# Verify:
wc -l data/v15/train.jsonl data/v15/valid.jsonl
```

Expected: ~680 train / ~75 valid (597 rebalanced + ~155 synthetic, minus 10% valid)

---

## Step 5b: Format for MLX Training

The merge output (`data/v15/train.jsonl`, `data/v15/valid.jsonl`) is already in chat format.
MLX expects the data directory to contain `train.jsonl` and `valid.jsonl` in chat format.

```bash
# Copy merged data to MLX training directory:
mkdir -p data/v15/mlx
cp data/v15/train.jsonl data/v15/mlx/train.jsonl
cp data/v15/valid.jsonl data/v15/mlx/valid.jsonl

# Verify:
wc -l data/v15/mlx/train.jsonl data/v15/mlx/valid.jsonl
```

---

## Step 6: Set Up MLX Environment (M5 Pro Max 128GB)

```bash
# Create venv if not already done:
python3 -m venv .venv
source .venv/bin/activate

# Install MLX and dependencies:
pip install mlx mlx-lm transformers datasets

# Verify MLX:
.venv/bin/python3 -c "import mlx; import mlx_lm; print('MLX ready')"
```

---

## Step 7: Train with MLX LoRA (M5 Pro Max — local)

```bash
# Training: Qwen3-4B with MLX LoRA (~1-2 hours on M5 Pro Max)
.venv/bin/python3 -m mlx_lm.lora \
  --config finetune/lora_config_v15_4B.yaml \
  2>&1 | tee training_v15.log
```

**Config**: `finetune/lora_config_v15_4B.yaml`
- Model: `mlx-community/Qwen3-4B-4bit` (auto-downloads from HuggingFace)
- 1400 iters, save every 100, eval every 100
- LoRA: rank=16, alpha=32, dropout=0.05
- Effective batch: 16 (batch=1, grad_accum=16)

With 128GB, you could also try 8-bit: `mlx-community/Qwen3-4B-8bit` (better gradient fidelity, ~8GB model).
Update the `model:` line in the YAML if available.

---

## Step 8: Checkpoint Sweep (MLX eval)

```bash
# Sweep all checkpoints against test set:
.venv/bin/python3 finetune/sweep_v13_1_1.5B.py \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-dir finetune/adapters_v15_4B \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v15.txt \
  --output-dir eval_results/v15_sweep_4B \
  --skip-existing

# Or eval a single checkpoint manually:
.venv/bin/python3 finetune/eval_student_v7.py \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter finetune/adapters_v15_4B/0000800_adapters.safetensors \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v15.txt \
  --output-dir eval_results/v15_sweep_4B \
  --save-predictions

# Hybrid scoring:
.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v15_sweep_4B/*.predictions.jsonl \
  --v12
```

### Success Gates
- [ ] Hybrid accuracy ≥ 98.3% (match V14 Q6_K)
- [ ] Model-only accuracy ≥ 88% (meaningful improvement over 83.7%)
- [ ] Parse failures ≤ 10
- [ ] No regression on REMOTE jobs (non-Anywhere)

---

## Step 9: Convert Best Model to GGUF + MLX 6-bit

```bash
BEST_ITER=???  # Replace with actual best iter from sweep

# MLX fuse (merges LoRA into base model):
.venv/bin/python3 -m mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path finetune/adapters_v15_4B \
  --save-path ~/merged_v15_4B_mlx \
  --de-quantize  # Fuse back to full precision for GGUF conversion

# Convert to MLX 6-bit (Mac deployment):
.venv/bin/python3 -m mlx_lm convert \
  --hf-path ~/merged_v15_4B_mlx \
  --mlx-path ~/qwen3_4B_v15_mlx6bit \
  -q --q-bits 6

# Convert to GGUF (requires llama.cpp):
python3 llama.cpp/convert_hf_to_gguf.py ~/merged_v15_4B_mlx --outtype f16 --outfile ~/qwen3_4B_v15_f16.gguf
./llama.cpp/llama-quantize ~/qwen3_4B_v15_f16.gguf ~/qwen3_4B_v15_Q6_K.gguf Q6_K
```

---

## Step 10: Final Eval (MLX 6-bit + GGUF Q6_K)

```bash
# MLX 6-bit eval:
.venv/bin/python3 finetune/eval_student_v7.py \
  --model ~/qwen3_4B_v15_mlx6bit \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v15.txt \
  --output-dir eval_results/v15_mlx6bit \
  --save-predictions

# GGUF Q6_K eval:
.venv/bin/python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v15_Q6_K.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v15.txt \
  --output-dir eval_results/v15_gguf_Q6_K

# Hybrid scoring for both:
.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v15_mlx6bit/*.predictions.jsonl --v12

.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v15_gguf_Q6_K/*.predictions.jsonl --v12
```

---

## Step 11: Compare V14 vs V15

```bash
.venv/bin/python3 finetune/compare_evals.py \
  eval_results/v14_gguf_Q6_K/hybrid_step0000800.json \
  eval_results/v15_sweep_4B/hybrid_best.json
```

---

## Files Created

| File | Purpose |
|------|---------|
| `prompts/student_v15.txt` | V15 student prompt (optimized) |
| `prompts/teacher_v15.txt` | V15 teacher prompt (Anywhere + Next.js fixes) |
| `finetune/downsample_oos_v15.py` | OOS downsampling script |
| `finetune/generate_v15_data.py` | Synthetic JD generation script |
| `finetune/merge_v15_data.py` | Data merge script |
| `finetune/lora_config_v15_4B.yaml` | **MLX LoRA training config (M5 Pro Max)** |
| `finetune/train_v15.py` | Training script (Lambda — legacy, not needed) |
| `finetune/sweep_v15.py` | Checkpoint sweep script |
| `finetune/eval_student_v15.py` | HF eval script |
| `data/v15/train_rebalanced.jsonl` | OOS-downsampled V14 data (597 examples) |
| `data/v15/oos_downsample_report.json` | Downsampling report |
| `docs/V15_PLAN.md` | Full implementation plan |
| `docs/V15_RUNBOOK.md` | This file |

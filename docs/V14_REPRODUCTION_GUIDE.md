# V14 Full Reproduction Guide

Everything needed to reproduce any step of the V14 pipeline from scratch.
All commands are exact — copy-paste ready.

---

## Overview

V14 trains three model sizes (0.6B, 1.5B, 4B) on Lambda GH200 using PyTorch + Unsloth,
then quantizes the best 4B checkpoint to GGUF format for Mac deployment.

```
Data prep (Mac) → Train (Lambda GH200) → Sweep checkpoints (Lambda)
→ Merge best 4B (Lambda) → Quantize to GGUF (Lambda) → Download to Mac
→ Convert to MLX (Mac) → Final eval (Mac/Lambda)
```

---

## Step 0: Lambda Instance Setup

**Instance**: Lambda GH200 (NVIDIA Grace Hopper, 97.8 GB VRAM, 192 GB RAM)
**SSH**: `ssh -i "/Users/faisal/Downloads/Apollo .pem" ubuntu@<INSTANCE_IP>`

```bash
# Clone repo
git clone <repo_url> ~/ai_eval_harness
cd ~/ai_eval_harness

# Install Python deps (Unsloth for fast LoRA training)
pip install unsloth
pip install -r requirements.txt  # or install manually as needed

# Build llama.cpp (for GGUF conversion + quantization)
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j8
cd ~/ai_eval_harness

# Install llama-cpp-python (for GGUF eval)
CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir
```

---

## Step 1: Data Preparation (Mac)

Training data lives at `data/v14/` — already formatted for HuggingFace training.

```bash
# Verify data
wc -l data/v14/train.jsonl data/v14/valid.jsonl
# Expected: ~774 train, ~86 valid

# If rebuilding from scratch: format V13.1 data for HF format
# (V14 uses same training data as V13.1, just different format)
npx tsx src/cli/format-for-mlx-v7.ts \
  --input data/v13_1/train_merged.jsonl \
  --output-dir data/v14 \
  --prompt prompts/student_v14.txt \
  --format hf   # HuggingFace format, not MLX
```

---

## Step 2: Training (Lambda)

All three models train sequentially via `finetune/run_v14_training.sh`.

### 2a. Qwen2.5-1.5B

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 finetune/train_v14.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir data/v14 \
  --output-dir finetune/adapters_v14 \
  --batch-size 32 \
  --grad-accum 1 \
  --max-steps 2000 \
  --max-seq-len 6144 \
  --save-every 200 \
  --eval-every 100 \
  2>&1 | tee training_v14_1.5B.log
```

### 2b. Qwen3-0.6B

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 finetune/train_v14.py \
  --model Qwen/Qwen3-0.6B \
  --data-dir data/v14 \
  --output-dir finetune/adapters_v14_0.6B \
  --batch-size 32 \
  --grad-accum 1 \
  --max-steps 2000 \
  --max-seq-len 6144 \
  --save-every 200 \
  --eval-every 100 \
  2>&1 | tee training_v14_0.6B.log
```

### 2c. Qwen3-4B

```bash
# Note: Qwen3.5-4B failed with OOM (vision encoder too large).
# Use Qwen3-4B (pure text, standard attention).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 finetune/train_v14.py \
  --model Qwen/Qwen3-4B \
  --data-dir data/v14 \
  --output-dir finetune/adapters_v14_4B \
  --batch-size 8 \
  --grad-accum 4 \
  --max-steps 2000 \
  --max-seq-len 6144 \
  --save-every 200 \
  --eval-every 100 \
  2>&1 | tee training_v14_4B.log
```

**Training times on GH200:**
- 1.5B: ~40 min
- 0.6B: ~20 min
- 4B: ~2.5 hrs

---

## Step 3: Checkpoint Sweep (Lambda)

Evaluates all saved checkpoints to find the best one. Uses V14 hybrid pipeline.

### 3a. Sweep 1.5B

```bash
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir finetune/adapters_v14 \
  --results-dir eval_results/v14_sweep_1.5B \
  2>&1 | tee sweep_v14_1.5B.log
# Best: step 1200 at 96.7% hybrid (116 parse fails)
```

### 3b. Sweep 0.6B

```bash
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen3-0.6B \
  --adapter-dir finetune/adapters_v14_0.6B \
  --results-dir eval_results/v14_sweep_0.6B \
  2>&1 | tee sweep_v14_0.6B.log
# Best: step 2000 at 97.9% hybrid (83 parse fails)
```

### 3c. Sweep 4B (fixed — no pre-fill bug)

```bash
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen3-4B \
  --adapter-dir finetune/adapters_v14_4B \
  --results-dir eval_results/v14_sweep_4B_fixed \
  2>&1 | tee sweep_v14_4B_fixed.log
# Best: step 800 at 98.7% hybrid (5 parse fails)
```

**CRITICAL — the `{{` pre-fill bug**: The original sweep used `{{` assistant pre-fill
which caused ~60% parse failures. The fixed sweep removes this. Always use the
`v14_sweep_4B_fixed` results, not `v14_sweep_4B`.

---

## Step 4: Merge Best 4B Checkpoint (Lambda)

Merges LoRA adapter weights into the base model to produce a standalone HF model.

```bash
python3 finetune/train_v14.py \
  --merge-only \
  --model Qwen/Qwen3-4B \
  --adapter finetune/adapters_v14_4B/checkpoint-800 \
  --output-dir finetune/merged_v14_4B

# Result: ~/ai_eval_harness/finetune/merged_v14_4B/ (~7.6 GB)
```

---

## Step 5: GGUF Conversion (Lambda)

Converts the merged HF model to GGUF format for llama.cpp inference.

### 5a. Convert to F16 GGUF (full precision baseline)

```bash
python3 ~/llama.cpp/convert_hf_to_gguf.py \
  finetune/merged_v14_4B \
  --outfile ~/qwen3_4B_v14_f16.gguf \
  --outtype f16 \
  2>&1 | tee ~/gguf_convert.log
# Result: 7.5 GB
```

### 5b. Generate imatrix (calibration data for IQ quantization)

```bash
# Generate calibration text from training data
python3 -c "
import json
lines = []
with open('data/v14/train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        for msg in d.get('messages', []):
            lines.append(msg['content'])
with open('imatrix_calibration.txt', 'w') as f:
    f.write('\n'.join(lines))
"

~/llama.cpp/build/bin/llama-imatrix \
  -m ~/qwen3_4B_v14_f16.gguf \
  -f ~/imatrix_calibration.txt \
  -o ~/imatrix.dat \
  --chunks 128 \
  -ngl 80
# Result: imatrix.dat (~3.7 MB)
```

### 5c. Quantize all variants

```bash
# Q6_K — Mac deployment target (3.1 GB, 98.3% hybrid)
~/llama.cpp/build/bin/llama-quantize \
  ~/qwen3_4B_v14_f16.gguf \
  ~/qwen3_4B_v14_Q6_K.gguf \
  Q6_K

# Q4_K_M — smaller Mac option (2.3 GB, 97.9% hybrid)
~/llama.cpp/build/bin/llama-quantize \
  ~/qwen3_4B_v14_f16.gguf \
  ~/qwen3_4B_v14_Q4_K_M.gguf \
  Q4_K_M \
  2>&1 | tee ~/quantize_Q4_K_M.log

# Q2_K — broken for fine-tuned models (1.6 GB, 0% accuracy)
~/llama.cpp/build/bin/llama-quantize \
  ~/qwen3_4B_v14_f16.gguf \
  ~/qwen3_4B_v14_Q2_K.gguf \
  Q2_K

# IQ2_XXS — broken for fine-tuned models (1.2 GB, 0% accuracy)
# Uses imatrix calibration data
~/llama.cpp/build/bin/llama-quantize \
  --imatrix ~/imatrix.dat \
  ~/qwen3_4B_v14_f16.gguf \
  ~/qwen3_4B_v14_IQ2_XXS.gguf \
  IQ2_XXS \
  2>&1 | tee ~/gguf_quant.log
```

---

## Step 6: GGUF Evaluation (Lambda)

Uses `finetune/eval_student_v14_gguf.py` with llama-cpp-python.

**Critical settings** (learned from debugging — see V14_IMPLEMENTATION_PROGRESS.md):
- No `chat_format` override — use GGUF's embedded Jinja template
- `/no_think` in system prompt (Qwen3 language-level signal, not pre-fill)
- `max_tokens=600`

```bash
# F16
python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v14_f16.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_gguf_f16 \
  --n-gpu-layers -1 \
  2>&1 | tee eval_gguf_f16.log

# Q6_K
python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v14_Q6_K.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_gguf_Q6_K \
  --n-gpu-layers -1 \
  2>&1 | tee eval_gguf_Q6_K.log

# Q4_K_M
python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v14_Q4_K_M.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_gguf_Q4_K_M \
  --n-gpu-layers -1 \
  2>&1 | tee eval_gguf_Q4_K_M.log

# Run hybrid scorer on each predictions file
python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v14_gguf_Q6_K/<timestamp>.predictions.jsonl \
  --v12 \
  --output eval_results/v14_gguf_Q6_K/hybrid.json
```

---

## Step 7: Get Models to Mac

**Lambda instance has been deleted (2026-03-19). All models are on HuggingFace.**

### 7a. Download from HuggingFace (preferred — resumable, no Lambda needed)

```bash
# Install CLI if needed
pip install huggingface_hub

# Production model (Q6_K — Mac deployment target)
huggingface-cli download FF-01/qwen3-4b-v14 \
  qwen3_4B_v14_Q6_K.gguf --local-dir ~/

# All GGUFs
huggingface-cli download FF-01/qwen3-4b-v14 \
  qwen3_4B_v14_f16.gguf \
  qwen3_4B_v14_Q6_K.gguf \
  qwen3_4B_v14_Q4_K_M.gguf \
  qwen3_4B_v14_Q2_K.gguf \
  qwen3_4B_v14_IQ2_XXS.gguf \
  --local-dir ~/

# Merged HF model (for MLX conversion — 7.6 GB)
huggingface-cli download FF-01/qwen3-4b-v14 \
  --include "merged_v14_4B/*" --local-dir ~/
```

HF token required (write-access token for private repo). Adapters are in the git repo at `finetune/adapters_v14*/` — no download needed.

### 7b. If re-running on a NEW Lambda instance (SCP route)

Use `rsync -avP` (resumable) instead of bare SCP for large files:

```bash
# GGUF files
rsync -avP -e "ssh -i '/path/to/key.pem'" \
  ubuntu@<IP>:~/qwen3_4B_v14_Q6_K.gguf ~/

# Merged HF model (for MLX conversion)
rsync -avP -e "ssh -i '/path/to/key.pem'" \
  ubuntu@<IP>:~/ai_eval_harness/finetune/merged_v14_4B/ ~/merged_v14_4B/
```

---

## Step 8: MLX Conversion (Mac)

Converts the merged HF model to MLX format for Apple Silicon inference.
MLX is Apple Silicon only — cannot run on Lambda.

```bash
cd /Users/faisal/Code/automation/ai_eval_harness

# 4-bit MLX (~2.3 GB)
.venv/bin/python3 -m mlx_lm convert \
  --hf-path ~/merged_v14_4B \
  --mlx-path ~/qwen3_4B_v14_mlx4bit \
  -q --q-bits 4

# 6-bit MLX (~3.1 GB)
.venv/bin/python3 -m mlx_lm convert \
  --hf-path ~/merged_v14_4B \
  --mlx-path ~/qwen3_4B_v14_mlx6bit \
  -q --q-bits 6
```

---

## Step 9: MLX Evaluation (Mac)

```bash
cd /Users/faisal/Code/automation/ai_eval_harness

# MLX 4-bit
.venv/bin/python3 finetune/eval_student_v7.py \
  --model ~/qwen3_4B_v14_mlx4bit \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_mlx4bit \
  --save-predictions

# MLX 6-bit
.venv/bin/python3 finetune/eval_student_v7.py \
  --model ~/qwen3_4B_v14_mlx6bit \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_mlx6bit \
  --save-predictions

# Hybrid score each predictions file
.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v14_mlx4bit/<timestamp>.predictions.jsonl \
  --v12

.venv/bin/python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v14_mlx6bit/<timestamp>.predictions.jsonl \
  --v12
```

---

## Step 10: GGUF Inference on Mac (Production)

Run the Q6_K GGUF locally via llama-cpp-python:

```bash
# Install llama-cpp-python on Mac (Metal backend)
CMAKE_ARGS="-DGGML_METAL=ON" pip install llama-cpp-python --no-cache-dir

# Eval (same script as Lambda — it auto-detects Metal)
.venv/bin/python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v14_Q6_K.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_gguf_Q6_K_mac \
  --n-gpu-layers -1
```

---

## Key Files Reference

| File | Location | Purpose |
|------|----------|---------|
| `finetune/train_v14.py` | Mac/Lambda | Training script (Unsloth + HF) |
| `finetune/run_v14_training.sh` | Mac/Lambda | Full training queue (all 3 models) |
| `finetune/sweep_v14.py` | Mac/Lambda | Checkpoint sweep |
| `finetune/eval_student_v14_gguf.py` | Mac/Lambda | GGUF inference eval |
| `finetune/eval_student_v7.py` | Mac | MLX inference eval |
| `finetune/compute_hybrid_v13_1.py` | Mac | Hybrid scoring (V13.1 regex) |
| `prompts/student_v14.txt` | Mac | V14 student prompt |
| `data/v14/train.jsonl` | Mac | 774 training examples |
| `data/v14/valid.jsonl` | Mac | 86 validation examples |
| `data/v12/test_labeled_audited.jsonl` | Mac | 239 test jobs (locked, eval only) |
| `finetune/adapters_v14_4B/checkpoint-800/` | Mac | Best 4B checkpoint |
| `finetune/merged_v14_4B/` | Mac (~/merged_v14_4B) | Merged 4B model (7.6 GB) |
| `imatrix.dat` | Mac (docs/lambda_logs/) | IQ2 calibration data |

---

## Results Summary

| Model | Size | Hybrid | Model-Only | Parse Fails |
|-------|------|--------|------------|-------------|
| 4B F16 GGUF | 7.5 GB | **98.7%** | 86.2% | 8 |
| 4B Q6_K GGUF | 3.1 GB | **98.3%** | 83.7% | 10 |
| 4B MLX 6-bit | ~3.1 GB | TBD | TBD | TBD |
| 4B Q4_K_M GGUF | 2.3 GB | 97.9% | 62.3% | 51 |
| 4B MLX 4-bit | ~2.3 GB | TBD | TBD | TBD |
| 0.6B HF 4-bit | 335 MB | 97.9% | 49.4% | 83 |
| 1.5B HF 4-bit | 839 MB | 96.7% | 41.4% | 116 |

**Mac deployment target**: `~/qwen3_4B_v14_Q6_K.gguf` (3.1 GB) via llama-cpp-python.
MLX 4-bit/6-bit results pending (in-progress eval).

---

## Gotchas

1. **`{{` pre-fill bug**: Original sweep used assistant pre-fill that caused 60–70% parse failures.
   Fixed by removing pre-fill entirely. Always use `v14_sweep_4B_fixed` results.
2. **Qwen3.5-4B OOM**: Vision encoder + hybrid attention exceeds 94.5 GB VRAM. Use Qwen3-4B only.
3. **`chat_format="chatml"` collision**: Overrides GGUF's Jinja template → garbage output.
   Remove entirely from `Llama()` constructor.
4. **`/no_think` not pre-fill**: Use `/no_think` in system prompt, not assistant message pre-fill.
   `create_chat_completion` closes pre-filled turns with `<|im_end|>` → zero output.
5. **Explicit field list hurts accuracy**: Adding field list to system prompt drops hybrid
   97.9% → 91.6% by shifting model attention from JD content to schema. Keep prompt minimal.
6. **Q2 destroys fine-tuning**: At ≤2.7 BPW, LoRA deltas are rounded away entirely.
   Minimum viable quantization is Q4 (4.95 BPW); Q6 recommended.
7. **One model at a time on M1**: 16 GB RAM; running two models concurrently causes OOM.
8. **MLX requires Apple Silicon**: Cannot run MLX conversion or eval on Lambda (NVIDIA).

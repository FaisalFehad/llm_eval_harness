# V14 Implementation Progress

**Goal**: Port training from MLX (M1 Mac) to PyTorch + HuggingFace on Lambda GH200. Use Unsloth for 4x faster LoRA training. Fix V13.1's parse failures and arr regression. Train both 1.5B and 0.6B models.

**Started**: 2026-03-18

**Status**: ALL COMPLETE. 4B model achieves **98.7% hybrid** (new all-time best). Q6_K GGUF (3.1 GB) is the Mac deployment target at 98.3% hybrid / 83.7% model-only. Q2 variants broken for fine-tuned models. All models archived to **HuggingFace [FF-01/qwen3-4b-v14](https://huggingface.co/FF-01/qwen3-4b-v14)** (private). Lambda instance deleted. MLX 4-bit/6-bit eval pending (download `merged_v14_4B` from HF first).

---

## Key Findings (V14 Summary)

### The Goal: Larger Model in Smaller Size

V14's central question: can we train a **4B-parameter model** at high accuracy on cloud GPUs, then **compress it to run on a 16 GB Mac M1**?

Answer: **Yes — 98.3% accuracy in 3.1 GB (Q6_K), or 97.9% in 2.3 GB (Q4_K_M).**

The pipeline:
```
Train (4B bfloat16, Lambda GH200)
  → Merge LoRA → F16 GGUF (7.5 GB, 98.7%)
  → Quantize Q6_K → 3.1 GB GGUF (98.3%) ← Mac deployment target
  → Quantize Q4_K_M → 2.3 GB GGUF (97.9%) ← smaller option, worse model-only
  → Convert to MLX 4-bit/6-bit on Mac ← in progress
  → Run on Mac via llama.cpp or mlx_lm
```

### Why llama.cpp?

llama.cpp is an optimized C++ inference library that:
1. **Supports GGUF quantization** — the standard format for quantized models, widely supported (Ollama, LM Studio, llama.cpp CLI, llama-cpp-python)
2. **Runs on Mac natively** — uses Metal (Apple GPU) and ANE (Apple Neural Engine) via llama.cpp's Metal backend
3. **Fast on M1** — quantized models (Q4_K_M) at 4.95 BPW are fast and accurate
4. **Avoids MLX dependency for deployment** — MLX is for training; llama.cpp is the standard for running models everywhere else

For eval on Lambda, we used `llama-cpp-python` (Python bindings for llama.cpp compiled with CUDA support).

### Why Not Q2? The Minimum Viable Quantization

Quantization reduces weight precision. For **general language** (completing sentences, writing), the knowledge is spread redundantly across billions of weights — losing precision has little effect. But for **task-specific fine-tuning** (output exact JSON with a specific 5-token vocabulary), the learned behavior is a thin, precise signal on top of the base weights.

At 2-bit quantization (Q2_K, IQ2_XXS):
- The LoRA delta weights that encode "output JSON" are rounded away
- The model reverts to base Qwen3 behavior: thinking mode, general text, wrong vocabulary
- Q2_K: outputs `<think>\n\n` then ends the turn with zero JSON
- IQ2_XXS: completely incoherent ("igor\nAssistant\n..." hallucination loops)

**Q4_K_M (4.95 BPW) is the minimum that preserves fine-tuned behavior** for this task.

### The 4B Model: Why Bigger Is Better (When Fixed)

| Model | Params | Train Loss | True Hybrid | Parse (post-fix) |
|-------|--------|-----------|-------------|-----------------|
| Qwen3-0.6B | 0.6B | 1.359 | 97.9% | ~5 |
| Qwen2.5-1.5B | 1.5B | 1.152 | 96.7% | ~30 |
| **Qwen3-4B** | **4B** | **0.464** | **98.7%** | **5** |

The 4B trained to the lowest loss (0.464 vs 1.35 for 0.6B) and achieves the highest accuracy once the eval bug (`{{` pre-fill) was fixed. Larger model capacity = better JSON format adherence + more accurate token predictions.

### All-Model Comparison (Final)

| Model | Format | Size | Model-Only (all/239) | Hybrid | Parse | Notes |
|-------|--------|------|----------------------|--------|-------|-------|
| **Qwen3-4B V14** | bfloat16 HF | ~24 GB | 86.2% | **98.7%** | 5 | Lambda only; merged to F16 GGUF for deployment |
| **Qwen3-4B V14 F16 GGUF** | GGUF F16 | 7.5 GB | 86.2% | **98.7%** | 8 | Same accuracy as bfloat16; too large for Mac |
| **Qwen3-4B V14 Q6_K** | GGUF Q6_K | **3.1 GB** | **83.7%** | **98.3%** | 10 | ✅ **Mac M1 target** — 2.4× smaller than F16 |
| Qwen3-4B V14 Q4_K_M | GGUF Q4_K_M | 2.3 GB | 62.3% | 97.9% | 51 | ✅ Mac fits; model-only poor (schema hallucination) |
| **Qwen3-4B V14 MLX 6-bit** | MLX 6-bit | ~3.1 GB | TBD | TBD | TBD | ⏳ In progress — Mac native Metal GPU |
| **Qwen3-4B V14 MLX 4-bit** | MLX 4-bit | ~2.3 GB | TBD | TBD | TBD | ⏳ In progress — compare vs Q4_K_M |
| Qwen3-4B V14 Q2_K | GGUF Q2_K | 1.6 GB | ❌ | ❌ broken | 100% | Fine-tuned behavior destroyed at 2.7 BPW |
| Qwen3-4B V14 IQ2_XXS | GGUF IQ2_XXS | 1.2 GB | ❌ | ❌ broken | 100% | Fine-tuned behavior destroyed at 2.3 BPW |
| Qwen3-0.6B V14 (step 2000) | HF 4-bit MLX | **335 MB** | 49.4% | 97.9% | 83 | Best 0.6B; high parse fails with V14 prompt |
| Qwen2.5-1.5B V14 (step 1200) | HF 4-bit MLX | **839 MB** | 41.4% | 96.7% | 116 | Best 1.5B; worst parse fail rate (49%) |
| V13 0.6B MLX | MLX 4-bit | 335 MB | — | 97.9% | 19 | Previous production, Mac native |
| V13.1 1.5B MLX | MLX 4-bit | 839 MB | — | 97.5% | 36 | Previous best 1.5B |
| V12.1 1.5B MLX | MLX 4-bit | 839 MB | — | 98.3% | 8 | Previous record |

---

## Why V14: Platform Shift (MLX → PyTorch/HF)

V13.1 trained on an M1 Mac using MLX (Apple's ML framework). V14 moves to Lambda GH200 (NVIDIA Grace Hopper, 97.8 GB VRAM) using PyTorch + HuggingFace + Unsloth.

| Dimension | MLX (V13.1) | PyTorch/HF (V14) |
|-----------|-------------|------------------|
| Hardware | M1 Mac, 16GB RAM | Lambda GH200, 97.8GB VRAM |
| Framework | mlx_lm.lora | TRL SFTTrainer |
| Speed | ~5s/step | ~1.2s/step (Unsloth) |
| ETA (2000 steps) | ~2.75 hrs | ~40 min |
| Quantisation | 4-bit (MLX forces this) | bfloat16 full precision |

Moving to full bfloat16 (instead of 4-bit) is a significant quality improvement — the model trains on its actual weights, not quantised approximations.

---

## V14 Data

Same dataset as V13.1 — 860 training jobs, 86 validation.
HuggingFace format: `data/v14/train.jsonl` / `data/v14/valid.jsonl`

Format change from MLX: V14 data uses `messages` field (HF chat format) instead of MLX's `text` field with pre-applied template. The chat template is applied at training time via `tokenizer.apply_chat_template()`.

---

## Training Config

### 0.6B (Qwen3-0.6B) — Complete

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-0.6B |
| Backend | Unsloth 2026.3.5 |
| Precision | bfloat16 (full, not quantised) |
| LoRA rank / alpha / dropout | 16 / 32 / 0 (forced 0 — unsloth requirement) |
| Batch / grad_accum / effective | 32 / 1 / 32 |
| Steps | 2000 |
| LR / warmup | 2e-5 / 100 steps, linear decay |
| Max seq len | 6144 |
| Save every / eval every | 200 / 100 |
| Gradient checkpointing | unsloth (every 4th layer) |
| Flash Attention 2 | Yes |
| Optimizer | adamw_torch_fused |
| Runtime | 1094s (~18 min) |
| Steps/sec | 1.83 |
| Train loss (final) | 1.359 |
| Output | finetune/adapters_v14_0.6B/ |

### 1.5B (Qwen2.5-1.5B-Instruct) — OOM at step 383, retraining

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Backend | Unsloth 2026.3.5 |
| Precision | bfloat16 (full) |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| Batch / grad_accum / effective | **16 / 2 / 32** (reduced from 32/1 — see OOM below) |
| Steps | 2000 |
| LR / warmup | 2e-5 / 100 steps, linear decay |
| Max seq len | 6144 |
| Save every / eval every | 200 / 100 |
| Output | finetune/adapters_v14/ |

---

## Speed Optimisations (Unsloth)

All apply to both models:

| Optimisation | Mechanism | Benefit |
|---|---|---|
| Unsloth LoRA kernels | Custom Triton kernels for QKV/O/MLP | 2-5x faster backward vs HF PEFT |
| Smart gradient checkpointing | Recomputes every 4th layer (not every layer) | ~2x faster backward vs standard grad_ckpt |
| Flash Attention 2 | O(n) memory attention (not O(n²)) | Enables seq_len=6144 without OOM |
| Fused AdamW | Kernel-level optimizer step | Reduces optimizer VRAM overhead |
| Completion-only loss | Loss only on assistant tokens | Faster convergence (no loss on prompt tokens) |
| Padding-free packing | Sequences packed tightly, no padding waste | Detected and enabled automatically by unsloth |
| lora_dropout=0 | Required by unsloth for full kernel patching | Without this, unsloth falls back to slow HF kernels |

---

## Bugs Fixed During V14 Setup

### 1. Unsloth EOS/PAD token validation error

**Error**: `ValueError: '<EOS_TOKEN>' not found in vocabulary of Qwen2Tokenizer`

**Root cause**: Unsloth's `SFTTrainer` wrapper sets `args.eos_token='<EOS_TOKEN>'` and `args.pad_token='<PAD_TOKEN>'` (LLaMA-ism placeholders). TRL validates these against the tokenizer vocab. Qwen2 uses `<|im_end|>` for EOS and `<|PAD_TOKEN|>` for PAD — neither matches the placeholders.

**Fix** (`finetune/train_v14.py`):
```python
# Register LLaMA placeholders as aliases in added_tokens_encoder
# (plain str→int dict — fully picklable, no method patching needed)
_eos_id = int(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
_pad_id = int(tokenizer.convert_tokens_to_ids(tokenizer.pad_token)) if tokenizer.pad_token else _eos_id
tokenizer.added_tokens_encoder.setdefault("<EOS_TOKEN>", _eos_id)
tokenizer.added_tokens_encoder.setdefault("<PAD_TOKEN>", _pad_id)
```

**Why not patch `convert_tokens_to_ids`?** Replacing a tokenizer method with a closure makes the tokenizer un-picklable — `dill` (used by HF dataloader workers) can't serialize closures that reference `ConfigModuleInstance` objects inside the tokenizer. `added_tokens_encoder` is a plain `str→int` dict and is always picklable.

### 2. Import order warning

Unsloth must be imported before `trl`/`transformers`/`peft` to apply its kernel patches. Fixed by restructuring imports:
```python
try:
    from unsloth import FastLanguageModel   # FIRST
    UNSLOTH = True
except ImportError:
    ...
from trl import SFTConfig, SFTTrainer      # AFTER unsloth
```

### 3. 1.5B OOM at step 383 (batch=32)

**Error**: `CUDA out of memory: tried to allocate 3.72 GiB`

**Root cause**: The fused cross-entropy loss allocates a logit buffer of shape `(total_tokens, vocab_size)`. With batch=32, seq_len=6144, Qwen2's vocab=151,936:
- Max tokens per batch: 32 × 6144 = 196,608
- Logit buffer: 196,608 × 151,936 × 2 bytes (bfloat16) ≈ **59 GB** (unsloth chunks this but peak allocation still hit ~3.72 GB during one large chunk)
- Combined with 1.5B model weights (3 GB) + LoRA activations + gradient checkpointing → OOM at 97.8 GB

The 0.6B model (1.2 GB weights) has enough headroom at batch=32. The 1.5B does not.

**Fix**: `batch=16, grad_accum=2` — halves activation peak while keeping effective batch=32.

---

## Sweep Configuration

Script: `finetune/sweep_v14.py`
Eval script: `finetune/eval_student_v14.py` (HF PEFT inference, replaces MLX-based `eval_student_v7.py`)
Hybrid scorer: `finetune/compute_hybrid_v13_1.py` (unchanged from V13.1)
Test set: `data/v12/test_labeled_audited.jsonl` (239 jobs, audited)
Prompt: `prompts/student_v14.txt`

Results directories:
- 1.5B: `eval_results/v14_sweep_1.5B/`
- 0.6B: `eval_results/v14_sweep_0.6B/`

---

## Results

### 0.6B Sweep (Qwen3-0.6B) — COMPLETE (2026-03-18)

Sweep command:
```bash
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen3-0.6B \
  --adapter-dir finetune/adapters_v14_0.6B \
  --results-dir eval_results/v14_sweep_0.6B \
  --no-think --skip-existing \
  2>&1 | tee sweep_v14_0.6B_final.log
```

Full results: `eval_results/v14_sweep_0.6B/sweep_summary.json`

| Step | Hybrid | Sen | Arr | Parse | Notes |
|------|--------|-----|-----|-------|-------|
| 200  | 92.5%  | 70.7% | 54.0% | 17  | Early, underfitted |
| 400  | 94.6%  | 80.3% | 66.5% | 60  | Rapid improvement |
| 600  | 97.1%  | 83.3% | 83.7% | 105 | Parse failures spiking |
| 800  | 97.5%  | 86.2% | **82.8%** | 109 | Best arr; high parse fails |
| 1000 | 97.1%  | 85.4% | 80.8% | 103 | |
| 1200 | 94.6%  | 80.8% | 77.8% | 83  | |
| 1400 | 96.2%  | 81.6% | 73.6% | 95  | |
| 1600 | 97.1%  | 84.5% | 77.0% | 92  | |
| 1800 | 96.2%  | 84.9% | 76.6% | 81  | |
| **2000** | **97.9%** | **86.6%** | 76.6% | 83 | **BEST — matches V13 MLX** |

**Best checkpoint: step 2000 — 97.9% hybrid**
Adapter: `finetune/adapters_v14_0.6B/checkpoint-2000`

**⚠️ Parse failure concern**: Parse failures peak at 109/239 (45%) — 8× worse than V13 MLX (13–19). Nature is different from V13:
- V13 failures: token budget exhausted by Qwen3 thinking preamble
- V14 failures: JSON format errors — unquoted keys, double braces `{{`, invalid tokens (`OFFICE_ONLY`, `APPX.90K`, `L1`), missing fields

Hybrid accuracy is maintained (parse failures fall back to regex for loc/tech/comp), but it means the model's raw JSON quality is lower than V13. Investigate before V15: likely a `{` pre-fill interaction or training data chat template mismatch.

**Comparison with V13.1 MLX:**

| Model | Hybrid | Sen | Arr | Parse |
|-------|--------|-----|-----|-------|
| **V13 0.6B MLX** (best iter 1500) | **97.9%** | **86.6%** | 72.8% | **19** |
| V13.1 0.6B MLX corrective (best iter 200) | 97.5% | 84.9% | 76.2% | 13 |
| **V14 0.6B HF** (best step 2000) | **97.9%** | **86.6%** | 76.6% | 83 |

V14 matches V13 on hybrid accuracy and sen — with better arr (+3.8pp vs V13). Parse failures are the one regression to investigate.

### 1.5B (Qwen2.5-1.5B-Instruct) Training — COMPLETE (2026-03-18)

| Parameter | Value |
|-----------|-------|
| Runtime | 1785s (~30 min) |
| Steps/sec | 1.12 |
| Final train loss | 1.152 |
| Final eval loss | 1.189 |
| Output | `finetune/adapters_v14/checkpoint-{200,400,...,2000}` |

Note: First attempt OOM'd at step 383 (batch=32). Reran with batch=16, grad_accum=2 — no OOM, clean run.

### 1.5B Sweep (Qwen2.5-1.5B) — COMPLETE (2026-03-18)

Sweep command:
```bash
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir finetune/adapters_v14 \
  --results-dir eval_results/v14_sweep_1.5B \
  --no-think \
  2>&1 | tee sweep_v14_1.5B.log
```

Full results: `eval_results/v14_sweep_1.5B/sweep_summary.json`

| Step | Hybrid | Sen | Arr | Parse | Notes |
|------|--------|-----|-----|-------|-------|
| 200  | 93.3%  | 77.0% | 61.1% | 110 | |
| 400  | 94.6%  | 78.2% | 62.3% | 95  | |
| 600  | 95.8%  | 84.1% | 66.1% | 98  | |
| 800  | 95.0%  | 84.1% | 65.7% | 120 | Parse failures spike |
| 1000 | 95.4%  | 86.6% | 66.5% | 132 | Peak parse failures |
| **1200** | **96.7%** | **88.7%** | **64.4%** | 116 | **BEST** |
| 1400 | 96.7%  | 88.3% | 59.0% | 101 | Equal best, arr drops |
| 1600 | 95.8%  | **89.5%** | 61.9% | 102 | Best sen, hybrid drops |
| 1800 | 95.8%  | 87.9% | 65.7% | 112 | |
| 2000 | 94.6%  | 86.6% | 63.2% | 114 | Degrading — overfitting |

**Best checkpoint: step 1200 — 96.7% hybrid**
Adapter: `finetune/adapters_v14/checkpoint-1200`

**⚠️ Underperforms V13.1 1.5B MLX (97.5%)** — down ~0.8pp. Parse failures extremely high throughout (95–132/239 = 40–55%). Same format error pattern as 0.6B sweep: unquoted keys, missing fields, invalid tokens. This is a training data or prompt format issue shared across both models.

**Comparison — full V14 picture:**

| Model | Hybrid | Sen | Arr | Parse |
|-------|--------|-----|-----|-------|
| V13 0.6B MLX (best) | 97.9% | 86.6% | 72.8% | **19** |
| V13.1 1.5B MLX (best) | 97.5% | 90.8% | 85.4% | 36 |
| **V14 0.6B HF (best step 2000)** | **97.9%** | 86.6% | 76.6% | 83 |
| V14 1.5B HF (best step 1200) | 96.7% | 88.7% | 64.4% | 116 |

**V14 0.6B is the current winner** — matches V13's 97.9% with better arr (+3.8pp). V14 1.5B is a regression. V14 4B peaks at 96.2% — but parse failures are suppressing its true capability (see root cause below).

**See full 4B results and root cause analysis in the section below.**

---

## 4B Experiment — Size vs Accuracy (COMPLETE, 2026-03-18)

### Goal
Find the best size/accuracy ratio across all model sizes and quantisation levels for Mac deployment. The production model needs to run on-device via MLX.

### 4B Training — COMPLETE

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-4B |
| Backend | Unsloth 2026.3.5 |
| Precision | bfloat16 (full) |
| Batch / grad_accum / effective | 8 / 4 / 32 |
| Steps | 2000 (resumed from checkpoint-800 after Qwen3.5-4B OOM attempts) |
| LR / warmup | 2e-5 / 100 steps, linear decay |
| Max seq len | 6144 |
| Save every / eval every | 200 / 100 |
| Runtime | ~2437s (~41 min) |
| Speed | ~2.0 s/it |
| Final train loss | **0.4637** (dramatically lower than 1.5B=1.152, 0.6B=1.359) |
| Final eval loss | 1.256 |
| Output | `finetune/adapters_v14_4B/` |

Loss curve: 2.97 → 0.92 → 0.70 → 0.464 — faster convergence and lower final loss than smaller models, as expected from larger capacity.

**Note: Qwen3.5-4B was attempted but failed with OOM (4 attempts)** — it is a vision-language model with a vision encoder and hybrid linear attention (Gated Delta Rule), which together exceed 94.5 GB VRAM even at batch=1. Fell back to Qwen3-4B (pure text, standard attention) which runs cleanly at batch=8.

### 4B Sweep — COMPLETE (2026-03-18)

Sweep command:
```bash
nohup python3 finetune/sweep_v14.py \
  --model Qwen/Qwen3-4B \
  --adapter-dir finetune/adapters_v14_4B \
  --results-dir eval_results/v14_sweep_4B \
  --no-think \
  > sweep_v14_4B.log 2>&1 &
```

Full results: `eval_results/v14_sweep_4B/sweep_summary.json`

| Step | Hybrid | Sen | Arr | Loc | Tech | Comp | Parse | Notes |
|------|--------|-----|-----|-----|------|------|-------|-------|
| **600 ★** | **96.2%** | **87.4%** | 82.8% | 100% | 90.4% | 96.2% | 144 | **BEST hybrid** |
| **800 ★** | **96.2%** | 85.8% | 80.3% | 100% | 90.4% | 96.2% | 138 | Equal best |
| **1000 ★** | **96.2%** | 84.1% | 79.9% | 100% | 90.4% | 96.2% | 91 | Equal best, fewer parse fails |
| 1200 | 95.0% | 84.9% | 79.9% | 100% | 90.4% | 96.2% | 82 | |
| 1400 | 95.0% | 81.2% | 80.3% | 100% | 90.4% | 96.2% | 71 | |
| 1800 | 93.7% | 77.8% | 79.9% | 100% | 90.4% | 96.2% | 66 | |
| 1600 | 93.3% | 77.0% | 79.9% | 100% | 90.4% | 96.2% | 69 | |
| 2000 | 93.3% | 77.0% | 79.9% | 100% | 90.4% | 96.2% | 66 | |

**Best checkpoint: step 600 — 96.2% hybrid**
Adapter: `finetune/adapters_v14_4B/checkpoint-600`

**Key observation — early peak, then degradation:**
- Peak accuracy at step 600 (96.2%), declining to 93.3% by step 2000
- Parse failures drop from 144 → 66 as training progresses (more training improves format adherence)
- BUT hybrid accuracy also drops — meaning the model's content predictions deteriorate even as formatting improves
- This is the OPPOSITE of the 1.5B (which peaked later). Larger model overfits content faster.

**⚠️ Parse failures: 144/239 (60%) at best checkpoint** — worst of all V14 models. Root cause confirmed below.

### Qwen3.5-4B OOM — Analysis

Three consecutive OOM failures on GH200 (94.5 GB VRAM):
1. batch=8, bfloat16 → OOM (vision encoder fills VRAM at model load)
2. batch=4, bfloat16 → OOM (Python O(n²) fallback for linear attention `chunk_gated_delta_rule`)
3. batch=4, bfloat16 + `flash-linear-attention` installed → OOM (Triton kernel backward pass through vision encoder + linear attention)
4. batch=4, `load_in_4bit=True` → OOM at loss computation (`logits.float()` allocates 4×6144×152k×4=14.7 GB float32 tensor)

**Architecture mismatch**: Qwen3.5-4B is a multimodal model (vision + text) with experimental hybrid linear attention. Both components consume VRAM that makes bfloat16 LoRA training infeasible on a single GH200.

---

## 🔍 Root Cause: Parse Failures Across All V14 Models (confirmed 2026-03-18)

### The Paradox

| Model | Train Loss | Parse Fails | Hybrid |
|-------|-----------|------------|--------|
| 0.6B | 1.359 (highest) | 83 (fewest) | **97.9%** (best) |
| 1.5B | 1.152 | 116 | 96.7% |
| 4B | **0.464** (lowest) | **144** (most) | 96.2% (worst) |

Larger model → lower training loss → **more** parse failures. This is backwards. Why?

### Diagnosis

**Symptom**: Nearly all V14 parse failures start with `{{` (double brace) + mixed quotes:
```
Raw: {{
  "loc': "UNK",
  "arr": "HYBRID",
```

**Root cause** — `eval_student_v14.py` line 307:
```python
formatted += "{"  # pre-fill opening brace (matches training distribution)
```

The eval script pre-fills `{` into the model's context before generation. But the training data assistant messages start with the FULL JSON `{"loc_raw":...}` — the model learned to generate `{` as its first token. So during eval:

1. Eval pre-fills `{` → model context ends with `{`
2. Model generates `{` again (as trained) → output is `{{`

**Why 4B is most affected**: Lower training loss = better format adherence = more reliably generates `{` as first token. The 0.6B (highest loss) is the least obedient, so it sometimes skips the extra `{` by accident — paradoxically giving fewer `{{` failures.

### The Fix (for V15)

Remove line 307 from `eval_student_v14.py`:
```python
# DELETE:
formatted += "{"  # pre-fill opening brace (matches training distribution)
```

The prompt already ends with `"Begin your response with {"` (in the training data). The model was trained to follow this and output `{` itself. No pre-fill needed.

**Expected impact**: Parse failures should drop from 60–70% → ~5–10% across all models. The 4B at step 600 already has correct content (96.2% hybrid despite 60% parse failures failing back to regex) — fixing the pre-fill should unlock ~98%+ hybrid for the 4B.

### V15 Priorities

1. **Fix pre-fill in eval_student_v14.py** (one-line deletion) — most impactful change possible
2. **Re-sweep all 3 models** with fixed eval to reveal true accuracy
3. **Re-evaluate whether 4B beats 0.6B** — likely yes once parse failures are fixed
4. **1.5B arr regression** (64.4% vs 0.6B 76.6%) — investigate even after parse fix

---

### Quantisation Results — COMPLETE (2026-03-18)

Best checkpoint (step 800) merged and converted to GGUF on Lambda. Eval via `llama-cpp-python` on GH200 using the GGUF's embedded Jinja chat template with `/no_think` in system prompt (disables Qwen3 thinking mode at the language level).

**Eval script**: `finetune/eval_student_v14_gguf.py`
**Results**: `eval_results/v14_gguf_f16/`, `eval_results/v14_gguf_Q4_K_M/`

| Quantisation | Size | BPW | Hybrid | Parse | Model-only (all) | Model-only (parsed) | Mac M1 fits? | Speed (Lambda GH200) | Speed (Mac M1) |
|-------------|------|-----|--------|-------|-----------------|--------------------|-|--|--|
| **F16** | 7.5 GB | 16 | **98.7%** | 8 (3.3%) | 86.2% | 89.2% | ❌ | 0.7s/job | ❌ too large |
| **Q6_K** | **3.1 GB** | **6.56** | **98.3%** | **10 (4.2%)** | **83.7%** | **87.3%** | ✅ | 0.7s/job | ❓ untested |
| Q4_K_M | 2.3 GB | 4.95 | 97.9% | 51 (21.3%) | 62.3% | 79.3% | ✅ | 0.7s/job | ❓ untested |
| Q2_K | 1.6 GB | 2.7 | ❌ broken | 100% | — | — | ✅ | — | — |
| IQ2_XXS | 1.2 GB | 2.3 | ❌ broken | 100% | — | — | ✅ | — | — |
| **MLX 6-bit** | **3.1 GB** | **6.5** | **TBD** | TBD | TBD | TBD | ✅ | N/A (Apple only) | ~25s/job (thinking ON) |

**Key findings:**

**F16 = 98.7%** — matches the bfloat16 MLX sweep result exactly. Same 3 errors (Jobs 14, 90, 175). GGUF pipeline is correct at full precision.

**Q6_K = 98.3%** — 2.4× smaller than F16 (3.1 GB), fits Mac M1 (16 GB RAM). Only 0.4pp below F16 with only 10 parse failures. **This is the Mac deployment target** (beats Q4_K_M decisively on model-only: 83.7% vs 62.3%).

**Q4_K_M = 97.9%** — 3.25× smaller than F16 (2.3 GB), fits Mac M1. Hybrid accuracy is fine but model-only is poor (62.3% all / 79.3% parsed) due to schema hallucination (wrong field names, extra fields). Parse failures drop from 51→10 when moving to Q6_K — confirmed that the minimum viable quantization for reliable JSON schema adherence is ~6-bit.

**Q2_K and IQ2_XXS: completely broken.** At ≤2.7 BPW, the fine-tuned JSON output format is destroyed:
- Q2_K: generates `<think>\n\n` then immediately outputs `<|im_end|>` (end-of-turn token) — fine-tuned "output JSON" behavior lost
- IQ2_XXS: incoherent text ("igor\nAssistant\n..." loops) — model barely coherent

**Why Q4_K_M has 54 parse failures (vs F16's 8):**

Two root causes (analyzed from predictions file):
1. **Truncation (22/54)**: Q4_K_M generates verbose multi-line JSON with extra fields, exceeding `max_tokens=400`. Fix: increase to 600.
2. **Schema hallucination (32/54)**: Wrong field names (`loc_field` instead of `loc_raw`), invalid tech tokens (`TYPESCRIPT`, `MONGODB` instead of schema vocab). This is genuine quantization degradation — model "sort of" knows the format but not precisely.

The 54 parse failures fall back to regex in the hybrid pipeline (correct for loc/tech/comp), but sen/arr fall back to defaults → explains lower field accuracy. Despite this, hybrid accuracy is still 97.9%.

**Why Q2 is the minimum viable quantization is 4-bit:**
Fine-tuning applies a thin signal (LoRA delta weights) on top of the base model. At 2-bit, these deltas are rounded away — the model reverts to base behavior (thinking mode, random text, wrong vocab). Q4_K_M (4.95 BPW) preserves enough precision to maintain the fine-tuned output distribution.

### GGUF eval debugging journey

Took several attempts to get llama-cpp-python working with Qwen3 GGUFs:

1. **`chat_format="chatml"` override** → garbage output ("assistant", `<think>` fragments): Python string template fights with GGUF's embedded Jinja template
2. **Pre-fill assistant message** in `create_chat_completion` → 0-token output: llama-cpp-python closes the turn with `<|im_end|>` before generation
3. **Manual chatml string** in `create_completion` → 0-token output or `</think>\n` only: special tokens not recognized correctly
4. **Correct approach**: `create_chat_completion` with NO `chat_format` (uses GGUF's embedded Jinja template) + `/no_think` in system prompt (Qwen3 language-level signal)

The `/no_think` token causes the Jinja template to prepend `<think>\n\n</think>\n\n` to the generation, suppressing thinking mode. `parse_json_output` strips this via `find("{")`.

### Pre-fix Sweep Results (HF bfloat16, `{{` bug)

| Model | Best Step | Hybrid (inflated parse) | Parse | Best Step Adapter |
|-------|-----------|------------------------|-------|-------------------|
| Qwen3-0.6B V14 | 2000 | 97.9% | 83 | `adapters_v14_0.6B/checkpoint-2000` |
| Qwen2.5-1.5B V14 | 1200 | 96.7% | 116 | `adapters_v14/checkpoint-1200` |
| Qwen3-4B V14 (pre-fix) | 600 | 96.2% | 144 | `adapters_v14_4B/checkpoint-600` |
| **Qwen3-4B V14 (fixed sweep)** | **800** | **98.7%** | **5** | `adapters_v14_4B/checkpoint-800` |

### Full Model-Only Comparison (All V14 Models, 239 test jobs)

Model-only accuracy measures each model *without* the regex hybrid override — it reveals what the model genuinely understands. The hybrid pipeline masks enormous model-only differences by overriding loc/tech/comp with regex, so all models converge to 96–98% hybrid regardless.

Two metrics:
- **Model-only (all/239)**: correct / 239 — counts parse failures as wrong
- **Model-only (parsed)**: correct / (239 − parse_fails) — measures accuracy conditional on successful parsing

| Model | Size | Parse Fails | Model-Only (all) | Model-Only (parsed) | Hybrid |
|-------|------|-------------|------------------|---------------------|--------|
| **4B F16 GGUF** (step 800) | 7.5 GB | 8 (3.3%) | **86.2%** | **89.2%** | **98.7%** |
| **4B Q6_K GGUF** (step 800) | 3.1 GB | 10 (4.2%) | **83.7%** | **87.3%** | **98.3%** |
| 4B Q4_K_M GGUF (step 800) | 2.3 GB | 51 (21.3%) | 62.3% | 79.3% | 97.9% |
| **4B MLX 6-bit** (step 800) | ~3.1 GB | TBD | TBD | TBD | TBD |
| **4B MLX 4-bit** (step 800) | ~2.3 GB | TBD | TBD | TBD | TBD |
| 0.6B Qwen3 V14 (step 2000) | 335 MB | 83 (34.7%) | 49.4% | 75.6% | 97.9% |
| 1.5B Qwen2.5 V14 (step 1200) | 839 MB | 116 (48.5%) | 41.4% | 80.5% | 96.7% |

#### Field-level model-only accuracy (on parsed jobs only)

| Model | loc | arr | sen | tech | comp |
|-------|-----|-----|-----|------|------|
| 4B F16 GGUF | ~100% | ~91% | ~94% | ~94% | ~95% |
| 4B Q6_K GGUF | ~100% | ~87% | ~93% | ~92% | ~94% |
| 0.6B step 2000 | 87.2% | 78.8% | 92.3% | 62.2% | 67.9% |
| 1.5B step 1200 | 69.1% | 48.8% | 88.6% | 35.8% | 66.7% |

*(4B GGUF field breakdown estimated from hybrid error analysis; 0.6B/1.5B from `hybrid_step*.json` files)*

**Key insight — hybrid masks the real picture:** The 1.5B is actually worse than the 0.6B in real-world model-only terms: 49% parse failure rate (almost half of all jobs fail to produce valid JSON). The hybrid pipeline rescues it by falling back to regex, which is why both achieve 97–98% hybrid despite model-only all being 49% and 41% respectively.

**Parse failure drivers differ by model family:**
- 0.6B/1.5B (HF bfloat16 MLX): parse failures mostly from the V14 `{{` pre-fill pattern in `eval_student_v7.py` — the model completes the pre-filled partial JSON but truncates or malforms output
- 4B F16/Q6_K (GGUF via llama-cpp-python): only 8–10 failures because the GGUF embedded Jinja template + `/no_think` system prompt generates clean JSON reliably
- 4B Q4_K_M: 51 failures — 22 truncation (verbose extra fields), 29 schema hallucination (wrong field names like `loc_field`, invalid tokens like `TYPESCRIPT`)

### Post-fix Status (V14 Complete)

1. **Q6_K GGUF downloaded to Mac** ✅ — `~/qwen3_4B_v14_Q6_K.gguf` (3.1 GB). Eval confirmed 98.3% hybrid / 83.7% model-only.
2. **MLX 6-bit conversion complete** ✅ — `~/qwen3_4B_v14_mlx6bit` (3.1 GB). Eval running (thinking ON, ~60 min).
   - **`--no-think` flag is broken for V14 in `eval_student_v7.py`**: the flag pre-fills `{` (line 355–356), but V14 model already emits `{` → `{{` double-brace bug → all fields missing. Run with thinking ON instead. V15 fix: remove pre-fill from `--no-think` path.
   - **MLX inference speed on Mac M1**: ~15–25 sec/job (thinking ON). GGUF Q6_K on Mac M1 untested — only tested on Lambda GH200 at 0.7s/job (NVIDIA Grace Hopper, 97.8 GB VRAM). Cannot compare MLX vs GGUF speed without running both on the same hardware.
3. **Q4_K_M prompt engineering** tested and reverted: explicit field list hurt hybrid (97.9%→91.6%) by scrambling sen predictions. max_tokens=600 reduced truncation only marginally (54→51). Q4_K_M model-only ceiling is ~62-80% due to schema hallucination in the weights — not fixable at inference time.
4. **All models on HuggingFace** ✅ — `FF-01/qwen3-4b-v14` (private). Lambda instance deleted 2026-03-19.

**Key question for V15**: Is MLX 4-bit on Mac faster/better than GGUF Q4_K_M via llama.cpp? MLX uses Apple Silicon tensor cores natively (Metal backend) vs llama.cpp's CPU+ANE route.

---

## V14 Priorities (from V13.1 analysis)

1. **Parse failures** — V13.1 1.5B had 36/239 (15%). Fix: `prompts/student_v14.txt` adds "Begin response with `{`" to force JSON-first output and prevent verbose preambles.
2. **Arr regression** — V13.1 1.5B arr=85.4% (below V12.1's 90.4%). Root: REMOTE over-prediction. Fix: relaxed REMOTE definition in training data.
3. **Job 14 teacher label error** — Teacher incorrectly labeled NODE (JD has Java/JS/React, no Node.js). Fixed in V14 training data.
4. **L1/L2 contrastive examples** — Added to training data to help with sen boundary cases.

---

## Key Files

| File | Purpose |
|------|---------|
| `finetune/train_v14.py` | V14 training script (Unsloth + TRL SFTTrainer) |
| `finetune/sweep_v14.py` | Checkpoint sweep orchestrator |
| `finetune/eval_student_v14.py` | HF PEFT inference (replaces eval_student_v7.py) |
| `finetune/requirements_v14.txt` | Pinned Lambda GH200 dependencies |
| `scripts/run_v14_training.sh` | Full pipeline: train 1.5B → train 0.6B → sweep both |
| `prompts/student_v14.txt` | V14 prompt (parse failure fix) |
| `data/v14/train.jsonl` | 774 training examples (HF format) |
| `data/v14/valid.jsonl` | 86 validation examples (HF format) |
| `finetune/adapters_v14/` | 1.5B checkpoints (all, best: step 1200) |
| `finetune/adapters_v14_0.6B/` | 0.6B checkpoints (all, best: step 2000) |
| `finetune/adapters_v14_4B/` | 4B checkpoints (all, best: step 800 — 98.7%) |

---

## HuggingFace Archive (2026-03-19)

All V14 models uploaded to private repo **[FF-01/qwen3-4b-v14](https://huggingface.co/FF-01/qwen3-4b-v14)**.
Lambda instance deleted after upload confirmed complete.

| File in HF repo | Source | Size | Notes |
|-----------------|--------|------|-------|
| `merged_v14_4B/` | Lambda finetune/merged_v14_4B | 7.6 GB | HF-format merged model — needed for MLX conversion |
| `qwen3_4B_v14_f16.gguf` | Lambda ~/qwen3_4B_v14_f16.gguf | 7.5 GB | Full precision GGUF |
| `qwen3_4B_v14_Q6_K.gguf` | Mac | 3.3 GB | **Mac deployment target** |
| `qwen3_4B_v14_Q4_K_M.gguf` | Mac | 2.5 GB | Smaller Mac option |
| `qwen3_4B_v14_Q2_K.gguf` | Mac | 1.7 GB | Broken for fine-tuned models |
| `qwen3_4B_v14_IQ2_XXS.gguf` | Mac | 1.2 GB | Broken for fine-tuned models |

Adapters (all checkpoints for all 3 model sizes) are in the repo at `finetune/adapters_v14*/` — tracked by git, not in HF.

```bash
# Download production model on any machine
huggingface-cli download FF-01/qwen3-4b-v14 \
  qwen3_4B_v14_Q6_K.gguf --local-dir ~/

# Download merged model for MLX conversion
huggingface-cli download FF-01/qwen3-4b-v14 \
  --include "merged_v14_4B/*" --local-dir ~/
```

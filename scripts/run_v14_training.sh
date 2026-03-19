#!/bin/bash
# V14 Training Queue — GH200 sequential (1.5B then 0.6B)
# Uses Unsloth for 4x faster training (~40 min per model vs 2.75 hrs)
# Run from: ~/ai_eval_harness on Lambda

set -e
cd ~/ai_eval_harness

echo "============================================================"
echo "V14 TRAINING QUEUE — GH200 sequential (1.5B then 0.6B)"
echo "Backend: Unsloth 2026.3.5 — 4x faster LoRA kernels"
echo "============================================================"

# ── Run 1: Qwen2.5-1.5B ─────────────────────────────────────────
echo ""
echo "[1/4] Starting Qwen2.5-1.5B at $(date)"
echo "      Unsloth batch=32, grad_ckpt=unsloth, 2000 steps (~40 min)"
echo ""

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

echo ""
echo "[1/4] 1.5B done at $(date)"

# ── Run 2: Qwen3-0.6B ───────────────────────────────────────────
echo ""
echo "[2/4] Starting Qwen3-0.6B at $(date)"
echo "      Unsloth batch=32, grad_ckpt=unsloth, 2000 steps (~20 min)"
echo ""

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

echo ""
echo "[2/4] 0.6B done at $(date)"

# ── Run 3: Sweep 1.5B checkpoints ────────────────────────────────
echo ""
echo "[3/4] Sweeping 1.5B checkpoints at $(date)"
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-dir finetune/adapters_v14 \
  --results-dir eval_results/v14_sweep_1.5B \
  2>&1 | tee sweep_v14_1.5B.log
echo "[3/4] 1.5B sweep done at $(date)"

# ── Run 4: Sweep 0.6B checkpoints ────────────────────────────────
echo ""
echo "[4/4] Sweeping 0.6B checkpoints at $(date)"
python3 finetune/sweep_v14.py \
  --model Qwen/Qwen3-0.6B \
  --adapter-dir finetune/adapters_v14_0.6B \
  --results-dir eval_results/v14_sweep_0.6B \
  2>&1 | tee sweep_v14_0.6B.log
echo "[4/4] 0.6B sweep done at $(date)"

echo ""
echo "============================"
echo " ALL DONE — results in:"
echo "  eval_results/v14_sweep_1.5B/sweep_summary.json"
echo "  eval_results/v14_sweep_0.6B/sweep_summary.json"
echo "============================"

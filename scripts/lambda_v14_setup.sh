#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# V14 Lambda H100 Setup Script
#
# Run steps 1-3 LOCALLY first, then steps 4+ ON LAMBDA.
#
# Machine specs: H100 80GB GPU, 525GB system RAM
# ─────────────────────────────────────────────────────────────────────────────

set -e

LAMBDA_IP="YOUR_LAMBDA_IP"          # e.g. 192.168.1.100
LAMBDA_USER="ubuntu"                # Lambda default user
REMOTE_DIR="~/ai_eval_harness"      # destination on Lambda

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 (LOCAL): Generate V14 training data with new prompt
# The V14 prompt has "Begin your response with {" — must bake into training data
# ─────────────────────────────────────────────────────────────────────────────
echo "=== STEP 1 (local): Format V14 training data ==="
echo "Run this on your local machine:"
echo ""
echo "  npx tsx src/cli/format-for-mlx-v7.ts \\"
echo "    --input data/v13_1/train_merged.jsonl \\"
echo "    --output-dir data/v14 \\"
echo "    --prompt prompts/student_v14.txt"
echo ""
echo "  # Verify output:"
echo "  wc -l data/v14/train.jsonl data/v14/valid.jsonl"
echo "  # Expected: ~774 train / ~86 valid"
echo ""
read -p "Press Enter once data/v14/ is ready..."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 (LOCAL): Transfer files to Lambda
# Uses rsync for efficiency — only sends changed files on re-runs
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 2 (local): Transfer files to Lambda ==="

# Transfer training scripts
rsync -avz --progress \
  finetune/train_v14.py \
  finetune/eval_student_v14.py \
  finetune/sweep_v14.py \
  finetune/compute_hybrid_v13_1.py \
  finetune/deterministic_baseline_v13_1.py \
  finetune/semantic_tokens_v7.py \
  finetune/requirements_v14.txt \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/finetune/"

# Transfer prompts
rsync -avz --progress \
  prompts/student_v14.txt \
  prompts/student_v13_1.txt \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/prompts/"

# Transfer V14 training data
rsync -avz --progress \
  data/v14/ \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/data/v14/"

# Transfer test set (for eval)
rsync -avz --progress \
  data/v12/test_labeled_audited.jsonl \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/data/v12/"

echo ""
echo "Transfer complete."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 (LOCAL): Quick verification of transfer
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 3 (local): Verify transfer ==="
ssh "${LAMBDA_USER}@${LAMBDA_IP}" "
  echo 'Remote files:'
  wc -l ${REMOTE_DIR}/data/v14/train.jsonl ${REMOTE_DIR}/data/v14/valid.jsonl ${REMOTE_DIR}/data/v12/test_labeled_audited.jsonl
  ls ${REMOTE_DIR}/finetune/*.py
"

echo ""
echo "=== STEPS 4+ run ON LAMBDA: ssh ${LAMBDA_USER}@${LAMBDA_IP} ==="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 (LAMBDA): Python environment setup
# ─────────────────────────────────────────────────────────────────────────────
cat << 'LAMBDA_SETUP'
# ── Run these commands ON LAMBDA ─────────────────────────────────────────────

cd ~/ai_eval_harness

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.x for H100)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention 2 (builds for H100 SM90 — takes ~10 min)
# H100 uses SM90 architecture, flash-attn needs to compile CUDA kernels for it
pip install flash-attn --no-build-isolation

# Install remaining dependencies
pip install -r finetune/requirements_v14.txt

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
# Expected: CUDA: True, GPU: H100, VRAM: 80.0GB

LAMBDA_SETUP

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 (LAMBDA): Download base model from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
cat << 'LAMBDA_MODEL'

# Download Qwen2.5-1.5B-Instruct (full bfloat16, ~3GB)
# HF Hub will cache to ~/.cache/huggingface/hub/
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen2.5-1.5B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
print('Download complete.')
"

LAMBDA_MODEL

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 (LAMBDA): Run training
# ─────────────────────────────────────────────────────────────────────────────
cat << 'LAMBDA_TRAIN'

# Verify data before training (safety check per critical rule)
wc -l data/v14/train.jsonl data/v14/valid.jsonl
# Expected: 774 train / 86 valid

# Start training (log to file so you can disconnect and reconnect)
# Use nohup or tmux so training continues if SSH disconnects
tmux new-session -d -s train "
  source .venv/bin/activate
  python3 finetune/train_v14.py 2>&1 | tee training_v14.log
"

# Monitor progress
tmux attach -t train
# or: tail -f training_v14.log

# Training will save checkpoints to:
#   finetune/adapters_v14/checkpoint-200/
#   finetune/adapters_v14/checkpoint-400/
#   ...
#   finetune/adapters_v14/checkpoint-2000/

# Expected training time on H100:
#   ~774 samples / batch=16 = 48 steps/epoch × 41 epochs = 2000 steps
#   At ~0.5s/step on H100 = ~17 min total (vs ~40h on M1!)

LAMBDA_TRAIN

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 (LAMBDA): Run checkpoint sweep
# ─────────────────────────────────────────────────────────────────────────────
cat << 'LAMBDA_SWEEP'

# After training completes, score all checkpoints
source .venv/bin/activate
python3 finetune/sweep_v14.py

# Or score specific steps:
# python3 finetune/sweep_v14.py --steps 1400 1600 1800 2000

# Results: eval_results/v14_sweep/sweep_summary.json

LAMBDA_SWEEP

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 (LOCAL): Pull results back
# ─────────────────────────────────────────────────────────────────────────────
cat << 'PULL_RESULTS'
# Run on LOCAL machine after sweep completes:

rsync -avz --progress \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/eval_results/v14_sweep/" \
  eval_results/v14_sweep/

rsync -avz --progress \
  "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/training_v14.log" \
  ./

# Compare V14 vs V13.1:
.venv/bin/python3 finetune/compare_evals.py \
  eval_results/v13_1_1.5B_sweep/sweep_summary.json \
  eval_results/v14_sweep/sweep_summary.json

PULL_RESULTS

echo ""
echo "Setup script complete. Follow the steps above."

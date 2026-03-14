#!/bin/bash
# V12 Phase 2E: Checkpoint evaluation sweep
#
# Evaluates checkpoints from iter 400 to max_iter in steps of 200,
# runs the V12 hybrid on each, and saves results to CSV.
#
# Usage:
#   bash finetune/eval_v12_checkpoints.sh [adapter_dir] [max_iter] [step]
#
# Example:
#   bash finetune/eval_v12_checkpoints.sh finetune/adapters_v12 2500 200

ADAPTER_DIR="${1:-finetune/adapters_v12}"
MAX_ITER="${2:-2500}"
STEP="${3:-200}"
TEST_FILE="data/v12/test_labeled_audited.jsonl"
PROMPT="prompts/student_v7.txt"
MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
OUTPUT_DIR="eval_results/v12"
CSV_FILE="$OUTPUT_DIR/checkpoint_selection.csv"

mkdir -p "$OUTPUT_DIR"

echo "V12 Checkpoint Evaluation Sweep"
echo "  Adapter dir: $ADAPTER_DIR"
echo "  Test file: $TEST_FILE"
echo "  Model: $MODEL"
echo "  Range: 400 to $MAX_ITER step $STEP"
echo ""

# CSV header
echo "iter,val_loss,hybrid_acc,sen_acc,inv_tok,parse_fail" > "$CSV_FILE"

for ((iter=400; iter<=MAX_ITER; iter+=STEP)); do
    # Format checkpoint name (zero-padded to 7 digits)
    CKPT=$(printf "%07d" $iter)
    CKPT_FILE="$ADAPTER_DIR/${CKPT}_adapters.safetensors"

    if [ ! -f "$CKPT_FILE" ]; then
        echo "  Skipping iter $iter (no checkpoint at $CKPT_FILE)"
        continue
    fi

    echo "=== Evaluating iter $iter ==="

    # Run standalone eval (--preprocess required: V12 training used preprocessed JDs)
    python3 finetune/eval_student_v7.py \
        --model "$MODEL" \
        --adapter "$CKPT_FILE" \
        --test-file "$TEST_FILE" \
        --prompt "$PROMPT" \
        --output-dir "$OUTPUT_DIR" \
        --preprocess \
        2>&1 | tail -5

    # Find the predictions file just created (may be in a subdirectory)
    PRED_FILE=$(find "$OUTPUT_DIR" -name "*${CKPT}*.predictions.jsonl" -newer "$CSV_FILE" 2>/dev/null | head -1)

    if [ -z "$PRED_FILE" ]; then
        echo "  WARNING: No predictions file found for iter $iter"
        continue
    fi

    echo "  Predictions: $PRED_FILE"

    # Run V12 hybrid
    HYBRID_OUTPUT="$OUTPUT_DIR/hybrid_iter_${CKPT}.json"
    python3 finetune/compute_hybrid.py \
        --test-file "$TEST_FILE" \
        --predictions "$PRED_FILE" \
        --v12 \
        --output "$HYBRID_OUTPUT" \
        2>&1 | grep -E "v12_hybrid:|V12 Hybrid"

    # Extract metrics from hybrid output
    if [ -f "$HYBRID_OUTPUT" ]; then
        HYBRID_ACC=$(python3 -c "import json; d=json.load(open('$HYBRID_OUTPUT')); print(d.get('v12_hybrid',{}).get('accuracy_pct',0))")
        # Get sen accuracy from the summary
        SEN_ACC=$(python3 -c "import json; d=json.load(open('$HYBRID_OUTPUT')); print(d.get('v12_hybrid',{}).get('field_accuracy',{}).get('sen',0))")
        PARSE_FAIL=$(python3 -c "import json; d=json.load(open('$HYBRID_OUTPUT')); print(d.get('v12_hybrid',{}).get('parse_fail',0))")
    else
        HYBRID_ACC=0
        SEN_ACC=0
        PARSE_FAIL=0
    fi

    # Get invalid tokens from the eval summary (may be in a subdirectory)
    SUMMARY_FILE=$(find "$OUTPUT_DIR" -name "*${CKPT}*.summary.json" -newer "$CSV_FILE" 2>/dev/null | head -1)
    if [ -n "$SUMMARY_FILE" ]; then
        INV_TOK=$(python3 -c "import json; d=json.load(open('$SUMMARY_FILE')); print(d.get('invalid_tokens',0))")
    else
        INV_TOK=0
    fi

    # Append to CSV
    echo "$iter,,$HYBRID_ACC,$SEN_ACC,$INV_TOK,$PARSE_FAIL" >> "$CSV_FILE"
    echo "  iter=$iter hybrid=$HYBRID_ACC% sen=$SEN_ACC% inv_tok=$INV_TOK parse_fail=$PARSE_FAIL"
    echo ""
done

echo "=== Done ==="
echo "Results saved to $CSV_FILE"
echo ""
echo "Best checkpoint:"
# Sort by hybrid_acc descending, skip header
tail -n +2 "$CSV_FILE" | sort -t, -k3 -rn | head -3

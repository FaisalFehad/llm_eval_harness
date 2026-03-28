#!/bin/bash
set -e

START=${1:-33}
END=${2:-64}
NUM_CHUNKS=${3:-64}

mkdir -p eval_results/v12_1_reeval/logs

echo "Running chunks $START-$END of $NUM_CHUNKS in parallel..."

for i in $(seq $START $END); do
  .venv/bin/python3 finetune/eval_student_v7.py \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --adapter finetune/adapters_v12 \
    --test-file data/v12/test_labeled_audited.jsonl \
    --prompt prompts/student_v7.txt \
    --output-dir eval_results/v12_1_reeval \
    --save-predictions \
    --chunk $i \
    --num-chunks $NUM_CHUNKS \
    > eval_results/v12_1_reeval/logs/chunk_${i}.log 2>&1 &
done

wait
echo "All chunks $START-$END done!"

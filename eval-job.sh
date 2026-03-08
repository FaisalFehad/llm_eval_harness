#!/usr/bin/env zsh
# Usage: ./eval-job.sh --prompt 9.8                (eval with v9.8 prompt)
#        ./eval-job.sh --job 16 --prompt 9.8       (single-job debug)
#        ./eval-job.sh --file data/new_uk_jobs_golden.jsonl  (different test set)
#
#   --prompt   Prompt version e.g. 9, 9.2, 9.8 (default: 9.8)
#   --job      Job number (single-job debug mode)
#   --adapter  Adapter path (default: finetune/adapters_v2b)
#   --file     Test file (default: data/linkedin_teacher_v2_eval_human_corrected.jsonl)

JOB=""
PROMPT_VER="9.8"
ADAPTER="finetune/adapters_v2b"
TESTFILE="data/linkedin_teacher_v2_eval_human_corrected.jsonl"

while [[ $# -gt 0 ]]; do
  case $1 in
    --job)     JOB="$2";        shift 2 ;;
    --prompt)  PROMPT_VER="$2"; shift 2 ;;
    --adapter) ADAPTER="$2";    shift 2 ;;
    --file)    TESTFILE="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

PROMPT_FILE="prompts/scorer_v${PROMPT_VER}.txt"

source .venv/bin/activate

if [[ -n "$JOB" ]]; then
  python3 finetune/eval_finetuned.py \
    --adapter "$ADAPTER" \
    --test-file "$TESTFILE" \
    --job "$JOB" \
    --prompt "$PROMPT_FILE"
else
  python3 finetune/eval_finetuned.py \
    --adapter "$ADAPTER" \
    --test-file "$TESTFILE" \
    --prompt "$PROMPT_FILE"
fi

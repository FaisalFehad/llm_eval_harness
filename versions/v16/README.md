# V16 — Qwen3-4B Student Model (fix6 prompt — SEN fallback restored)

**Status**: IN PROGRESS (2026-05-03). Started from V15 base, fix6 prompt.
**Change from V15**: sen rule restored to teacher behavior — "Decide using the job TITLE primarily. If the title is ambiguous, use the Description as fallback."

## Rationale

V15 fix4 says: "Decide using the job TITLE only. Ignore experience language in the description."
Teacher prompt says: "SENIORITY (Title primarily, Description as fallback)"

The fix4/input/label mismatch causes ~80 training examples where the label used JD fallback but the prompt forbade it. Fix6 restores alignment between prompt instructions and training labels.

## Champion Checkpoint

**TBD — needs retraining with fix6 prompt baked into training data.**

## Key files

| Path | Purpose |
|------|---------|
| `manifest.json` | Machine-readable config |
| `prompts/student.txt` (symlink to fix6 prompt) | Champion prompt |
| `prompts/teacher.txt` | gpt-4.1-mini labeling prompt |
| `configs/lora.yaml` | LoRA training config |
| `configs/promptfoo.yaml` | Eval config — uses fix6 variant |
| `data/` | V15 training data (copied, reformat to fix6 on train) |
| `scripts/` | Pipeline scripts (copied from V15) |
| `adapters/` | LoRA checkpoints (to be generated) |
| `eval_results/` | Eval outputs |

## Next steps

1. Reformat training data with fix6 prompt text:
   ```bash
   npx tsx src/cli/format-for-mlx-v7.ts \
     --input versions/v15/data/v14_reformatted_v2/train.jsonl \
     --output versions/v16/data/mlx/train.jsonl \
     --prompt versions/v16/prompts/student.txt
   ```
2. Configure `configs/lora.yaml`:
   - `adapter_path: "versions/v16/adapters/v16_4B"`
3. Run training from V15 iter 700 checkpoint:
   ```bash
   # Resume from V15 champion with corrective LR
   harness train lora --version v16
   ```
4. Sweep eval to pick best checkpoint.

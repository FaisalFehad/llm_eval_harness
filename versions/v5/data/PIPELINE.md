# V5 Student Training Pipeline — Semantic Tokens

Step-by-step execution guide. Each step depends on the previous one.

---

## Step 1A: Build combined input pool

Combines all data sources, deduplicates, strips old labels.

```bash
npx tsx src/cli/build-input-pool.ts
```

**Output**: `data/v5/all_input_pool.jsonl` (~1524 jobs)
**Verify**: Check total count matches expectations.

---

## Step 1B: Label everything with GPT-4o-mini (semantic tokens)

Sends all jobs through the V5 teacher prompt. ~$0.20, ~5 minutes.

```bash
npx tsx src/cli/label-jobs.ts \
  --input data/v5/all_input_pool.jsonl \
  --output data/v5/all_labeled_pool.jsonl \
  --prompt prompts/teacher_v5.txt \
  --model gpt-4o-mini \
  --concurrency 10
```

**Output**: `data/v5/all_labeled_pool.jsonl`
**Also creates**: `.failures.jsonl` and `.inconsistencies.jsonl` if any
**Verify**:
- Parse failure rate < 2%
- Inconsistencies < 5%
- Check semantic token distributions in output

---

## Step 1C: Check distributions against minimums

```bash
npx tsx src/cli/check-distribution.ts \
  --input data/v5/all_labeled_pool.jsonl
```

**Output**: `data/v5/all_labeled_pool.distribution.json`
**Review**: Note which categories need synthetic data.

---

## Step 1D: Generate synthetic jobs for gaps (if needed)

```bash
npx tsx src/cli/generate-synthetic.ts \
  --distribution data/v5/all_labeled_pool.distribution.json \
  --output data/v5/synthetic_jobs.jsonl \
  --model gpt-4o-mini
```

**Output**: `data/v5/synthetic_jobs.jsonl`

---

## Step 1E: Label synthetic jobs

```bash
npx tsx src/cli/label-jobs.ts \
  --input data/v5/synthetic_jobs.jsonl \
  --output data/v5/synthetic_labeled.jsonl \
  --prompt prompts/teacher_v5.txt \
  --model gpt-4o-mini \
  --concurrency 10
```

---

## Step 1F: Combine labeled pool + synthetic

```bash
cat data/v5/all_labeled_pool.jsonl data/v5/synthetic_labeled.jsonl > data/v5/full_pool.jsonl
```

**Verify**: Re-run distribution check on full_pool.jsonl:
```bash
npx tsx src/cli/check-distribution.ts --input data/v5/full_pool.jsonl
```

---

## Step 2: Build eval and training sets

```bash
npx tsx src/cli/build-datasets.ts \
  --input data/v5/full_pool.jsonl \
  --eval-output data/v5/eval_150_golden.jsonl \
  --train-output data/v5/train_800.jsonl \
  --seed 42 \
  --target-train 800
```

**Output**:
- `data/v5/eval_150_golden.jsonl` (150 jobs: 50/50/50)
- `data/v5/train_800.jsonl` (~800 jobs)
- `data/v5/split_manifest.json`

**MANUAL STEP**: Verify eval labels!
- Review ALL 50 maybe labels (read JD, check each token)
- Spot-check 20 good_fit + 20 bad_fit labels
- Fix any incorrect labels in eval_150_golden.jsonl

---

## Step 3: Format for MLX training

```bash
npx tsx src/cli/format-for-mlx.ts \
  --input data/v5/train_800.jsonl \
  --output-dir data/v5/mlx \
  --prompt prompts/student_v5.txt \
  --valid-pct 10 \
  --max-tokens 7500
```

**Output**:
- `data/v5/mlx/train.jsonl` (~720 examples)
- `data/v5/mlx/valid.jsonl` (~80 examples)

**Verify**: Spot-check samples — student prompt format, semantic token responses.

---

## Step 4: Train

```bash
source .venv/bin/activate
python3 -m mlx_lm.lora --config finetune/lora_config_v5.yaml
```

**Time**: ~1-2 hours on M1 (600 iters, 0.5B model)

**Monitor**: val loss at each checkpoint. At iters 150, 300, 450, 600 run eval:

```bash
python3 finetune/eval_student.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter finetune/adapters_v5/<checkpoint> \
  --test-file data/v5/eval_150_golden.jsonl \
  --verbose --save-predictions
```

Select checkpoint with highest label accuracy.

---

## Step 5: Final eval

```bash
python3 finetune/eval_student.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter finetune/adapters_v5 \
  --test-file data/v5/eval_150_golden.jsonl \
  --verbose --save-predictions
```

**Acceptance criteria**:
- Label accuracy ≥ 90%
- No field accuracy < 85%
- Invalid token rate < 5%

**If < 90%**:
1. Analyse field × token error patterns
2. Generate 50-100 targeted synthetic jobs
3. Retrain at 400 iters
4. Re-evaluate

---

## Files Created

| File | Description |
|------|-------------|
| `prompts/teacher_v5.txt` | GPT-4o-mini teacher prompt (semantic tokens) |
| `prompts/student_v5.txt` | Minimal student prompt (3 lines) |
| `src/lib/semantic-tokens.ts` | Token definitions + score conversion (TypeScript) |
| `finetune/semantic_tokens.py` | Token definitions + score conversion (Python) |
| `src/cli/build-input-pool.ts` | Combines all data sources |
| `src/cli/label-jobs.ts` | Labels with OpenAI + validates tokens |
| `src/cli/check-distribution.ts` | Distribution checker vs minimums |
| `src/cli/generate-synthetic.ts` | Synthetic job generator for gaps |
| `src/cli/build-datasets.ts` | Eval/train set assembly |
| `src/cli/format-for-mlx.ts` | MLX formatter with smart truncation |
| `finetune/lora_config_v5.yaml` | LoRA training config |
| `finetune/eval_student.py` | V5 eval script (semantic tokens) |

# Student Distillation — Pipeline

Step-by-step guide to prepare training data for the student model (Qwen2.5-0.5B).

---

## Step 1: Preprocess raw jobs

Normalizes LinkedIn scraper fields for teacher labeling.

```bash
npx tsx src/cli/preprocess-raw-jobs.ts \
  --input "data/Student Training Data/real_linkedin_500.jsonl" \
  --output "data/Student Training Data/preprocessed_500.jsonl" \
  --min-length 50
```

**Verify**: Output has ~490+ records, all with `title` + `jd_text` >= 50 chars.

---

## Step 2: Teacher-label the preprocessed jobs

```bash
source .venv/bin/activate
python3 finetune/generate_teacher_labels.py \
  --adapter finetune/adapters_v2b \
  --input "data/Student Training Data/preprocessed_500.jsonl" \
  --output "data/Student Training Data/teacher_labeled_500.jsonl" \
  --prompt prompts/scorer_v9.8.txt \
  --verbose
```

**Time**: ~2-3 hours on M1. Close other apps.

**Verify**: Parse failure rate < 5%. Check label distribution:
```bash
python3 -c "
import json
from collections import Counter
jobs = [json.loads(l) for l in open('data/Student Training Data/teacher_labeled_500.jsonl')]
print(f'Total: {len(jobs)}')
print('Labels:', dict(Counter(j['label'] for j in jobs)))
print('Comp:', dict(Counter(j['comp'] for j in jobs)))
"
```

---

## Step 3: Generate augmented data

Creates 4 augmented files from teacher-labeled jobs (unlabeled, ready for Step 4).

```bash
npx tsx src/cli/augment-training-data.ts \
  --input "data/Student Training Data/teacher_labeled_500.jsonl" \
  --output-dir "data/Student Training Data" \
  --seed 42
```

**Output files**:
- `salary_augmented.jsonl` (~200 records)
- `contrastive_pairs.jsonl` (~90 records)
- `location_variants.jsonl` (~50 records)
- `truncated_jds.jsonl` (~100 records)

**Verify**: Check counts printed by script match expectations.

---

## Step 4: Teacher-label augmented data

Run teacher on each augmented file. Can be done sequentially (one at a time — GPU memory).

```bash
source .venv/bin/activate

python3 finetune/generate_teacher_labels.py \
  --adapter finetune/adapters_v2b \
  --input "data/Student Training Data/salary_augmented.jsonl" \
  --output "data/Student Training Data/salary_augmented_labeled.jsonl" \
  --prompt prompts/scorer_v9.8.txt

python3 finetune/generate_teacher_labels.py \
  --adapter finetune/adapters_v2b \
  --input "data/Student Training Data/contrastive_pairs.jsonl" \
  --output "data/Student Training Data/contrastive_pairs_labeled.jsonl" \
  --prompt prompts/scorer_v9.8.txt

python3 finetune/generate_teacher_labels.py \
  --adapter finetune/adapters_v2b \
  --input "data/Student Training Data/location_variants.jsonl" \
  --output "data/Student Training Data/location_variants_labeled.jsonl" \
  --prompt prompts/scorer_v9.8.txt

python3 finetune/generate_teacher_labels.py \
  --adapter finetune/adapters_v2b \
  --input "data/Student Training Data/truncated_jds.jsonl" \
  --output "data/Student Training Data/truncated_jds_labeled.jsonl" \
  --prompt prompts/scorer_v9.8.txt
```

**Time**: ~2-3 hours total for ~440 jobs.

**Verify**: Spot-check salary_augmented_labeled — do comp scores match the injected salary tier?

---

## Step 5: Assemble curated training set

Merges all labeled sources, deduplicates, excludes eval set, stratified samples.

```bash
npx tsx src/cli/assemble-student-training.ts \
  --eval-set "data/Student Training Data/clean_eval.jsonl" \
  --output "data/Student Training Data/curated_training_set.jsonl" \
  --target-size 450 \
  --augmented-cap 30 \
  --seed 42
```

**Verify**:
- Total ~450 records
- good_fit 20-30%, maybe 33-42%, bad_fit 33-42%
- comp≠0 > 35%
- Augmented ≤ 30%
- Zero overlap with clean_eval.jsonl

---

## Step 6: Format for MLX

Uses the existing format script with student-specific settings.

```bash
npx tsx src/cli/format-finetune-training-data-for-mlx.ts \
  --input "data/Student Training Data/curated_training_set.jsonl" \
  --output-dir "data/Student Training Data/mlx" \
  --prompt prompts/scorer_v9.8.txt \
  --valid-pct 10 \
  --model qwen2.5
```

**Verify**:
- System message = "Respond with JSON only." (NOT "/no_think" — student is Qwen2.5)
- ~405 train / ~45 valid records
- Spot-check sample: user message has correct template, assistant response has all fields

---

## Step 7: Train student

```bash
source .venv/bin/activate
python3 -m mlx_lm.lora --config finetune/lora_config_student.yaml
```

**Time**: ~1-2 hours on M1 (600 iters, batch_size=2, 0.5B model).

**Verify**: Loss curve decreases, validation loss doesn't diverge.

---

## Step 8: Eval student

```bash
python3 finetune/eval_finetuned.py \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --adapter finetune/adapters_student \
  --test-file "data/Student Training Data/clean_eval.jsonl" \
  --prompt prompts/scorer_v9.8.txt \
  --verbose
```

**Acceptance criteria**:

| Metric | Target | Stretch |
|---|---|---|
| Overall label accuracy | >88% | >92% |
| good_fit accuracy | >70% | >80% |
| maybe accuracy | >80% | >88% |
| bad_fit accuracy | >92% | >95% |
| comp accuracy | >80% | >88% |
| loc accuracy | >90% | >95% |
| Parse failure rate | <3% | <1% |

---

## Total time estimate

| Step | Time |
|---|---|
| 1. Preprocess | < 1 min |
| 2. Teacher-label 500 | 2-3 hours |
| 3. Augment | < 1 min |
| 4. Teacher-label augmented | 2-3 hours |
| 5. Assemble | < 1 min |
| 6. Format MLX | < 1 min |
| 7. Train | 1-2 hours |
| 8. Eval | ~30 min |

**Total**: ~6-9 hours (mostly GPU time). Steps 2 and 4 can run overnight.

# Student Training Data Audit

Generated: 2026-03-05

## Purpose

Audit of all available data for training the student model (Qwen2.5-0.5B) via knowledge distillation from the teacher (Qwen3-4B fine-tuned, v2b adapter, v9.8 prompt).

---

## Trainable Pool (255 unique labeled jobs)

Excludes 145 `clean_eval.jsonl` jobs reserved for student evaluation.

### By source

| Source | Count | Notes |
|---|---|---|
| linkedin_teacher_v2 (scored + human-corrected) | 144 | 217 scraped, 101 eval (excluded from training pool via clean_eval), rest usable |
| golden_jobs_scored | 92 | 103 total minus 11 in clean_eval |
| new_uk_jobs_golden | 11 | 72 total minus 61 in clean_eval |
| location_diversity_supplement | 8 | Non-London UK locations |

### Label distribution

```
good_fit:  ███                          15  (5.9%)   ← CRITICAL SHORTAGE
maybe:     ███████████████              77  (30.2%)
bad_fit:   ████████████████████████████████ 163 (63.9%)
```

### Label × Comp cross-tabulation

```
           comp=-30   comp=0   comp=5  comp=15  comp=25    TOTAL
good_fit        0        3        2        5        5       15
maybe           0       56        2       11        8       77
bad_fit         5      143        8        3        4      163
TOTAL           5      202       12       19       17      255
```

**WARNING**: comp=0 is 79.2% of all data. Teacher v2 had 89% comp=0 and it caused systematic comp under-scoring.

### Label × Loc cross-tabulation

```
            loc=-50    loc=0   loc=10   loc=25    TOTAL
good_fit        0        0        2       13       15
maybe           0        0       21       56       77
bad_fit        26        2      106       29      163
TOTAL          26        2      129       98      255
```

---

## Unlabeled Data (ready for teacher labeling)

| File | Records | Status |
|---|---|---|
| `Student Training Data/real_linkedin_500.jsonl` | 500 | **HIGH PRIORITY** — diverse locations (256 UK, 244 non-UK), needs `description→jd_text` field rename |
| `new_uk_jobs.jsonl` | 205 | 72 already labeled (new_uk_jobs_golden), 133 have empty/short JD |
| `custom_linkedin_data_to_train_teacher_v2_raw.jsonl` | 217 | Already labeled as linkedin_teacher_v2_scored |

---

## File Relationships & Deduplication

### Golden set (103 jobs)
- `golden_jobs_scored.jsonl` — MASTER (deterministic scores)
- `finetune/train.jsonl` — 70 of these (teacher training split)
- `finetune/test.jsonl` — 33 of these (teacher test split)
- `distillation/train_relabeled.jsonl` — same 70, teacher-labeled (26 label disagreements vs deterministic)

### UK LinkedIn (72 jobs)
- `new_uk_jobs_golden.jsonl` — MASTER (deterministic scores)
- `distillation/uk_jobs_relabeled.jsonl` — same 72, teacher-labeled
- `test_location_failures.jsonl` — 9 of these (subset)

### LinkedIn v2 scraped (217 jobs)
- `linkedin_teacher_v2_scored.jsonl` — all 217 (deterministic scores)
- `linkedin_teacher_v2_train.jsonl` — 100 (training split)
- `linkedin_teacher_v2_eval.jsonl` — 101 (eval split)
- `linkedin_teacher_v2_eval_human_corrected.jsonl` — same 101, human-corrected (3 label + 10 sub-score changes)

### Composite teacher training files (subsets + oversampling)
- `finetune/train_v2.jsonl` — 126 records, 122 unique
- `finetune/train_v2b.jsonl` — 194 records, 130 unique
- `finetune/train_v2c.jsonl` — 282 records, 130 unique

### Student eval (DO NOT TRAIN ON)
- `clean_eval.jsonl` — 145 jobs (73 linkedin_v2_eval_hc + 61 new_uk_jobs_golden + 11 finetune/test)

---

## Shopping List

### Must do

| # | What | Why | How to get |
|---|---|---|---|
| 1 | Label the 500 raw jobs | Triples pool from 255→~750 | `generate_teacher_labels.py` with v9.8 prompt + v2b adapter |
| 2 | ~50 good_fit-likely jobs | Only 15 unique good_fit — model can't learn the class | Scrape: "Senior Node.js Engineer London £100k+", "Staff Engineer TypeScript UK remote" |
| 3 | ~30 visible-salary jobs (£80k+) | comp=25 has only 17 examples | Filter LinkedIn for jobs showing salary ranges |
| 4 | ~10 low-salary UK jobs (£30-44k) | comp=-30 has only 5 examples | Junior dev roles in UK with visible salary |

### Nice to have

| # | What | Why |
|---|---|---|
| 5 | ~20 "Up to £X" salary jobs | Teaches comp=0 for ceiling-only salary pattern |
| 6 | ~15 non-London UK tech jobs with salary | Strengthens loc=10 + comp≠0 combinations |

### Don't need more of

- bad_fit (163 unique + ~440 incoming from 500 raw)
- US/non-UK jobs (26 unique + ~244 incoming)
- comp=0 jobs (already 79% of pool)

---

## Recommended Training Distribution (~350 examples)

After labeling 500 new + merging existing + deduplication + stratified sampling:

```
TARGET:
  good_fit:  ~90  (26%)  ← oversample 3x from ~30-50 unique
  maybe:     ~130 (37%)  ← oversample 1.3x from ~100 unique
  bad_fit:   ~130 (37%)  ← undersample from ~500+
    non-UK:   ~50         (cap — diminishing returns after diverse locations)
    UK:       ~80         (interesting bad_fits: wrong role/tech/comp)

COMP TARGET:
  comp=0:   ~60%  (vs current 79%)
  comp≠0:   ~40%  (need oversampling of salary examples)

LOC TARGET:
  loc=-50:  ~14%  (~50 non-UK)
  loc=10:   ~37%  (~130 non-London UK)
  loc=25:   ~29%  (~100 London)
  other:    ~20%  (~70 remote/unclear)
```

Config: `finetune/lora_config_student.yaml` expects ~350 examples, 600 iters = ~3.4 epochs.

# Student Model — Data Design

The plan for assembling high-quality training data to distil the 4B teacher
(Qwen3-4B, v2b adapter, v9.8 prompt) into the 0.5B student (Qwen2.5-0.5B-Instruct-4bit).

---

## Goal

~400-450 curated training examples with balanced label and comp distributions.
All examples labeled by the teacher model — the student learns the teacher's judgement,
not the deterministic scorer.

---

## Target Distribution

### Labels

```
good_fit:  ~110 (25%)   ← needs augmentation (only ~15 unique exist today)
maybe:     ~170 (38%)
bad_fit:   ~170 (38%)   ← undersample (500+ available)
  non-UK:   ~50          cap at ~50 diverse locations
  UK:       ~120         interesting failures: wrong role/tech/comp
```

### Comp scores

```
comp=0:    ~55%    (down from current 79%)
comp=5:    ~10%
comp=15:   ~12%
comp=25:   ~15%    (up from current 7%)
comp=-30:  ~8%     (up from current 2%)
```

### Location

```
loc=-50:   ~12%   (~50 non-UK, diverse countries)
loc=10:    ~35%   (~150 non-London UK)
loc=25:    ~35%   (~150 London + remote UK)
loc=0:     ~18%   (~80 unclear/other)
```

---

## Data Sources

### 1. Raw jobs — teacher labeling (Step 1)

| Source | Records | Status |
|---|---|---|
| `real_linkedin_500.jsonl` (this folder) | 500 | Unlabeled, needs field rename `description→jd_text` |
| Existing labeled pool (see AUDIT.md) | 255 | Already labeled, ready to merge |

**Output**: `teacher_labeled_500.jsonl` in this folder.

### 2. Salary injection augmentation (Step 2)

Take real UK software jobs with comp=0 (no salary in JD). Inject a salary line in a
randomly chosen format and position. Run through teacher to get labels.

#### Salary range formats

```
£60,000 - £80,000                  full, hyphen, spaces
£60k - £80k                        k shorthand, hyphen, spaces
£60K–£80K                          uppercase K, en-dash
£60k to £80k                       k, "to"
£60,000 to £80,000                 full, "to"
£60,000–£80,000 per year           full, en-dash, "per year" suffix
£60k–80k                           k, en-dash, second £ omitted
60k–80k                            no currency symbol
£60,000-£80,000 per annum          full, hyphen, "per annum"
£60,000 - £80,000 p.a.             full, "p.a."
```

#### Single value formats

```
£80,000
£80k
from £60,000
```

#### Ceiling-only formats (comp=0 — no lower bound)

```
Up to £80,000
up to £80k
Up to £80K
```

#### Wrong currency formats (comp=0 — not GBP)

```
$80,000 - $120,000                 USD
€70,000 - €90,000                  EUR
```

#### Vague formats (comp=0)

```
Competitive salary
Competitive package
Market rate
```

#### Salary tiers

| Tier | Salary range | comp score | Example |
|---|---|---|---|
| High | £100,000+ | 25 | £105,000 - £130,000 |
| Mid-high | £75,000–£99,999 | 15 | £78,000 - £92,000 |
| Mid | £55,000–£74,999 | 5 | £58,000 - £70,000 |
| Low | £45,000–£54,999 | 0 | £48,000 - £52,000 |
| Very low | <£45,000 | -30 | £30,000 - £38,000 |
| Ceiling only | "Up to £X" | 0 | Up to £90,000 |
| USD | Any | 0 | $120,000 - $150,000 |
| EUR | Any | 0 | €70,000 - €90,000 |
| Vague | N/A | 0 | Competitive salary |

#### Injection position (randomize)

- Top of JD (before first section)
- Inline (after responsibilities/requirements section)
- Bottom of JD (in benefits/package section)

**Target**: ~60 real UK jobs × 3-4 salary variants = ~200 augmented examples.

**Output**: `salary_augmented.jsonl` in this folder.

### 3. Contrastive minimal pairs (Step 3)

Same real job, change ONE dimension to flip the label. Teaches decision boundaries.

```
Job A: Senior Node.js | London  | £110k  →  good_fit
Job B: Senior Node.js | London  | (none) →  maybe       ← only comp changed
Job C: Senior Node.js | Paris   | £110k  →  bad_fit     ← only loc changed
Job D: Junior Python  | London  | £110k  →  bad_fit     ← only role+tech changed
```

**Target**: ~30 real jobs × 3 variants = ~90 examples.

**Output**: `contrastive_pairs.jsonl` in this folder.

### 4. Location format variation (Step 4)

Same job, rewrite location string in different formats LinkedIn uses.

```
"London, England, United Kingdom"  →  loc=25
"London, UK"                       →  loc=25
"London"                           →  loc=25
"Greater London Area"              →  loc=25
"London (Hybrid)"                  →  loc=25
```

**Target**: ~50 examples.

**Output**: `location_variants.jsonl` in this folder.

### 5. JD truncation (Step 5)

Same job, cut JD to 50% and 25% length. Teaches graceful degradation when
LinkedIn scrapes return truncated descriptions.

**Target**: ~50 jobs × 2 truncation levels = ~100 examples.

**Output**: `truncated_jds.jsonl` in this folder.

---

## Assembly Pipeline

```
Step 1:  Label 500 raw jobs with teacher         → teacher_labeled_500.jsonl
Step 2:  Salary injection on ~60 UK jobs          → salary_augmented.jsonl
Step 3:  Contrastive pairs from ~30 jobs          → contrastive_pairs.jsonl
Step 4:  Location format variation on ~50 jobs    → location_variants.jsonl
Step 5:  JD truncation on ~50 jobs                → truncated_jds.jsonl
                                                    ─────
Step 6:  Merge all + existing 255 labeled         → combined_pool.jsonl (~1,190)
Step 7:  Deduplicate by title+location            → deduplicated_pool.jsonl
Step 8:  Stratified sample + oversample           → curated_training_set.jsonl (~400-450)
Step 9:  Shuffle (seed=42) + split 90/10          → train.jsonl + valid.jsonl
Step 10: Format to MLX chat                       → mlx/train.jsonl + mlx/valid.jsonl
Step 11: Train student                            → adapters_student/
Step 12: Eval against clean_eval.jsonl (145 jobs)
```

---

## Augmentation Rules

1. **All augmented jobs go through the teacher** — student learns teacher labels only
2. **Cap augmented data at ~30% of final training set** — majority is real unmodified jobs
3. **Randomize formats** — don't cluster similar salary formats together
4. **Shuffle everything** — format script handles this (seed=42)
5. **Never include clean_eval.jsonl jobs** — reserved for student evaluation

---

## Acceptance Criteria

After training, eval against `clean_eval.jsonl` (145 jobs):

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

## Files in this folder

```
Student Training Data/
├── DATA_DESIGN.md                  ← this file
├── clean_eval.jsonl                ← 145 jobs, student eval set (DO NOT TRAIN ON)
├── real_linkedin_500.jsonl         ← 500 raw unlabeled jobs (input to Step 1)
├── teacher_labeled_500.jsonl       ← (Step 1 output, pending)
├── salary_augmented.jsonl          ← (Step 2 output, pending)
├── contrastive_pairs.jsonl         ← (Step 3 output, pending)
├── location_variants.jsonl         ← (Step 4 output, pending)
├── truncated_jds.jsonl             ← (Step 5 output, pending)
├── combined_pool.jsonl             ← (Step 6 output, pending)
├── curated_training_set.jsonl      ← (Step 8 output, pending)
└── mlx/                            ← (Step 10 output, pending)
    ├── train.jsonl
    └── valid.jsonl
```

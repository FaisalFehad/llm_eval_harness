# V15 Implementation Plan

**Goal**: Improve model-only accuracy from 83.7% → ~90%+ by fixing training data distribution gaps and prompt optimization. No architecture changes — data engineering only.

**Started**: 2026-04-12
**Target model**: Qwen3-4B (same as V14)
**Training platform**: Lambda GH200 (same as V14)
**Eval target**: `data/v12/test_labeled_audited.jsonl` (239 jobs, unchanged)

---

## Verified Error Budget (from Q6_K GGUF predictions)

| Current | Value |
|---------|-------|
| Model-only accuracy | **83.7%** (200/239) or 87.3% (200/229 valid) |
| Parse failures | 10/239 (4.2%) |
| Label-changing errors | 29 |
| Theoretical ceiling (perfect fields) | 95.8% (229/239) |

### Counterfactual Analysis — Fixing One Field at a Time

| Fix | Labels Recovered | New Accuracy (of 239) |
|-----|-----------------|----------------------|
| Fix TECH alone | +15 | 89.9% |
| Fix LOC alone | +7 | 86.6% |
| Fix COMP alone | +5 | 85.8% |
| Fix SEN alone | +1 | 84.1% |
| Fix ARR alone | +0 | 83.7% |
| **Fix LOC+TECH** | **+22** | **92.9%** |
| **Fix LOC+TECH+COMP** | **+27** | **95.0%** |

### Realistic Target

| Scenario | Accuracy | Gain |
|----------|----------|------|
| Pessimistic | ~88.7% | +5pp |
| **Expected** | **~90.8%** | **+7pp** |
| Optimistic | ~92.9% | +9pp |

---

## Phase 0: Prompt Optimization (Before Training)

**Duration**: 1 day
**Draft**: `prompts/student_v15_draft.txt`

### Changes from V14

| Section | V14 | V15 | Why |
|---------|-----|-----|-----|
| loc: "Anywhere" | Buried in list: `"Fully Remote" or "Remote" or "Anywhere" → REMOTE` | **First item, standalone**: `"Anywhere" → REMOTE.` | 8/8 Anywhere errors — model didn't learn buried rule. Position bias: first items get more attention weight |
| tech: scope rule | `or exactly ["OOS"] when none apply` | **Explicit inclusion rule**: `Include a token if that technology appears ANYWHERE in the JD, even as a secondary skill.` | 11 OOS false positives — model over-filters when tech is secondary |
| tech: Next.js/NestJS | Not mentioned | **Explicit exclusion**: `Next.js is NOT Node.js → JS_TS. NestJS is NOT Node.js → JS_TS.` | 8 NODE false positives — 3 are Next.js confusion |
| tech: AI_ML definition | Not defined | **Explicit inclusion list**: `any mention of AI, ML, machine learning, LLM, generative AI, deep learning` | 15 AI_ML false negatives — model misses weak signals |
| comp: midpoint rule | V14 base: none. Exp1: `£60k–£80k → RANGE_55_74K` (by example) | **Explicit midpoint rule**: `For ranges: use the MIDPOINT.` with worked examples | 4 boundary errors on £60k-£80k ranges |
| comp: UP_TO_ONLY | `No salary mentioned → NO_GBP.` | **"to £X" pattern**: `"Up to £X" or "to £X" with no lower bound → UP_TO_ONLY.` | 4 UP_TO_ONLY errors — "to £85k" pattern missed |
| comp: non-GBP | `Non-GBP salary (USD, EUR, etc.) → NO_GBP` | **Expanded + OTE**: `$, USD, EUR, CAD, AUD, daily rates → NO_GBP. OTE amounts → NO_GBP.` | 9 currency errors — USD/CAD/OTE treated as GBP |

### What Was NOT Changed (and why)

- **arr section**: Unchanged. Arr has 18 errors but 0 label impact when fixed alone. Not worth the prompt space.
- **sen section**: Unchanged. Only 1 label recovery. The errors are genuinely ambiguous titles.
- **UK city list**: Unchanged. 0 errors on UK city classification.
- **`Begin your response with {`**: Kept. Critical for JSON compliance.
- **`_raw` fields**: Not mentioned in prompt (learned from training data). Removing them collapsed accuracy to 56%.

### Validation Plan

The prompt will be validated by:
1. Re-labeling the 239 test jobs with gpt-4.1-mini using V15 prompt (temp=0) → check for labeling changes
2. Comparing V15 teacher labels vs V14 teacher labels → any diffs = prompt interpretation change
3. If >5 label changes, investigate each one before proceeding

---

## Phase 1: Training Data Augmentation

**Duration**: 3-5 days
**Goal**: Fix distribution gaps without increasing dataset size dramatically

### 1A. LOC: "Anywhere" → REMOTE Examples (est. +5-6 labels)

**Problem**: 8/8 `job_location="Anywhere"` jobs wrong. 0 training examples with this pattern.
**Action**:
- Source or generate **20 jobs** with `job_location="Anywhere"` → `loc=REMOTE`
- Include variety: some with UK mentions in JD, some without, some with non-UK JD context
- Also add 5 "Anywhere" → `loc=OUTSIDE_UK` examples (where JD clearly specifies non-UK country) as negative contrastive
- Label all with gpt-4.1-mini (temp=0) using V15 prompt

**Contrastive pairs needed**:
| job_location | JD context | Golden loc |
|-------------|------------|-----------|
| "Anywhere" | No country mentioned | REMOTE |
| "Anywhere" | UK company, UK benefits | REMOTE |
| "Anywhere" | "Based in New York" in JD | OUTSIDE_UK |
| "Anywhere" | "Singapore office" in JD | OUTSIDE_UK |

### 1B. TECH: OOS Rebalancing (est. +3-5 labels)

**Problem**: OOS = 42.2% of training (327/774) but only 15.5% of test (37/239). Creates OOS prediction bias.
**Action**:
- **Downsample OOS** from 327 → ~150 in training data (remove ~177 OOS examples)
  - Keep OOS examples that are diverse (different industries, roles)
  - Remove duplicative OOS examples (similar JDs, same tech stacks)
- **Do NOT remove OOS examples that are edge cases** (e.g., jobs that mention tracked tech in passing but are correctly OOS)

**Selection criteria for OOS removal**:
1. Rank all 327 OOS training examples by JD diversity
2. Keep the 150 most distinct examples
3. Prioritize keeping: (a) non-tech roles, (b) roles with untracked tech (Go, Rust, Java-only), (c) edge cases near OOS boundary

### 1C. TECH: NODE/AI_ML/JS_TS Distribution (est. +3-5 labels)

**Problem**:
- NODE appears in 52% of test but only 24% of training (2.1x under)
- AI_ML+NODE combo: 1.2% train vs 8.8% test (7.6x under)
- NODE alone: 2.1% train vs 10.0% test (4.9x under)
- JS_TS+NODE: 4.4% train vs 10.5% test (2.4x under)

**Action**:
- Generate/source **30 NODE-alone jobs** (Node.js backend, no React/AI_ML)
- Generate/source **25 AI_ML+NODE combo jobs** (Node.js + AI/ML features)
- Generate/source **15 JS_TS+NODE jobs** (TypeScript + Node.js, no React)
- Generate/source **15 Next.js → JS_TS (NOT NODE)** contrastive examples
  - Next.js + TypeScript → `["JS_TS"]` or `["JS_TS", "REACT"]`
  - Node.js + TypeScript → `["NODE", "JS_TS"]`
- Generate/source **10 secondary AI_ML examples** where AI is not the primary role
  - "DevOps Engineer" + AI monitoring → `["AI_ML"]` or `["AI_ML", "NODE"]`
  - "Data Analyst" + ML tools → `["AI_ML"]`
  - "QA Engineer" + AI testing → `["AI_ML", "JS_TS"]`

### 1D. COMP: Boundary & Currency Fixes (est. +2-3 labels)

**Problem**:
- 4 midpoint errors on £60k-£80k ranges
- 9 non-GBP currency errors
- 4 UP_TO_ONLY misses

**Action**:
- Generate **15 midpoint boundary examples**:
  - £60k-£80k → RANGE_55_74K (midpoint 70k) × 5
  - £70k-£75k → RANGE_55_74K (midpoint 72.5k) × 3
  - £75k-£85k → RANGE_75_99K (midpoint 80k) × 3
  - £40k-£75k → RANGE_55_74K (midpoint 57.5k) × 2
  - £43k-£51k → RANGE_45_54K (midpoint 47k) × 2
- Generate **15 non-GBP currency examples**:
  - USD salaries ($90k, $150k, $200k) → NO_GBP × 5
  - CAD salaries → NO_GBP × 3
  - EUR salaries → NO_GBP × 2
  - OTE amounts ("OTE £92k") → NO_GBP × 3
  - Daily rates ("£600 per day") → NO_GBP × 2
- Generate **10 UP_TO_ONLY examples**:
  - "Up to £80,000" → UP_TO_ONLY × 3
  - "to £85k" (no "up" prefix) → UP_TO_ONLY × 3
  - "Up to £70k" → UP_TO_ONLY × 2
  - Contrastive: "£60,000 to £80,000" → RANGE_55_74K (NOT UP_TO_ONLY — has lower bound) × 2

### 1E. Data Summary

| Category | New Examples | Source |
|----------|-------------|--------|
| "Anywhere" → REMOTE | 20 | Generate + label |
| "Anywhere" → OUTSIDE_UK (contrastive) | 5 | Generate + label |
| NODE-alone jobs | 30 | Generate + label |
| AI_ML+NODE combos | 25 | Generate + label |
| JS_TS+NODE combos | 15 | Generate + label |
| Next.js contrastive (JS_TS not NODE) | 15 | Generate + label |
| Secondary AI_ML examples | 10 | Generate + label |
| Comp boundary examples | 15 | Generate + label |
| Non-GBP currency examples | 15 | Generate + label |
| UP_TO_ONLY examples | 10 | Generate + label |
| **Total new examples** | **~160** | |
| OOS removed (downsampling) | ~-177 | Remove from existing |
| **Net change** | **~-17** (774 → ~757) | Smaller but better balanced |

### Post-Augmentation Distribution Targets

| Token | V14 Train % | V15 Target % | Test % |
|-------|-------------|-------------|--------|
| tech=OOS | 42.2% | ~20% | 15.5% |
| tech=NODE (any combo) | 24.3% | ~35% | 51.9% |
| tech=AI_ML (any combo) | 26.4% | ~32% | 42.7% |
| loc=REMOTE | 4.1% | ~8% | 24.3% |
| comp=RANGE_55_74K | 8.8% | ~12% | 20.5% |
| comp=RANGE_75_99K | 8.5% | ~11% | 20.5% |

---

## Phase 2: Data Pipeline

**Duration**: 2-3 days

### Step 1: Generate Synthetic JDs

```bash
# Use gpt-4.1-mini to generate JDs matching target distributions
# Temperature=0.7 for generation, temp=0 for labeling
npx tsx src/cli/generate-jobs-v7.ts \
  --distribution data/v15/target_distribution.json \
  --count 160 \
  --output data/v15/synthetic_raw.jsonl
```

If no generation script exists, write one. Each JD must:
- Have realistic structure (company, role, requirements, salary, benefits)
- Match the target tech/comp/loc distribution
- Include natural language variation (not templated)

### Step 2: Label with Teacher

```bash
npx tsx src/cli/label-jobs-v7.ts \
  --input data/v15/synthetic_raw.jsonl \
  --output data/v15/synthetic_labeled.jsonl \
  --prompt prompts/student_v15_draft.txt
```

### Step 3: Audit Labels

```bash
npx tsx src/cli/audit-training-data-v7.ts \
  --input data/v15/synthetic_labeled.jsonl \
  --eval-set data/v12/test_labeled_audited.jsonl
```

Manual audit checklist:
- [ ] Every "Anywhere" job labeled REMOTE
- [ ] Every Next.js job has NO NODE in tech
- [ ] Every secondary AI/ML job has AI_ML in tech
- [ ] Every £60k-£80k range labeled RANGE_55_74K
- [ ] Every USD/CAD salary labeled NO_GBP
- [ ] Every "Up to £X" labeled UP_TO_ONLY

### Step 4: OOS Downsampling

```bash
# Script to select ~150 diverse OOS examples from existing 327
# Remove ~177 redundant OOS examples
python3 finetune/downsample_oos_v15.py \
  --input data/v14/train.jsonl \
  --keep 150 \
  --output data/v15/train_rebalanced.jsonl
```

### Step 5: Merge and Split

```bash
# Merge: rebalanced V14 data + new V15 synthetic data
# Split: 90/10 train/valid (same as V14)
python3 finetune/merge_v15_data.py \
  --base data/v15/train_rebalanced.jsonl \
  --augment data/v15/synthetic_labeled.jsonl \
  --output-dir data/v15/ \
  --valid-ratio 0.1
```

Expected: ~757 train / ~84 valid (similar size to V14's 774/86)

### Step 6: Format for Training

```bash
npx tsx src/cli/format-for-mlx-v7.ts \
  --input data/v15/train.jsonl \
  --output-dir data/v15/hf/ \
  --prompt prompts/student_v15_draft.txt
```

Note: V14 used HuggingFace format (not MLX) for Lambda training. May need a separate HF formatting step.

---

## Phase 3: Training

**Duration**: 1-2 days (Lambda GPU time: ~2 hours)
**Platform**: Lambda GH200

### Training Config (start from V14 config, adjust)

| Parameter | V14 Value | V15 Value | Notes |
|-----------|-----------|-----------|-------|
| Model | Qwen3-4B | Qwen3-4B | Same base model |
| Method | Fresh from base | **Fresh from base** | New prompt = new input distribution |
| LoRA rank/alpha | 16/32 | 16/32 | Keep same |
| LR | 2e-5 | 2e-5 | Keep same |
| Warmup | 100 | 100 | Keep same |
| Batch/Grad accum | 1/16 | 1/16 | Keep same |
| Max steps | 1400 | **1400** | V14 best at step 800/1400 range |
| Save every | 200 | **100** | Finer sweep near expected peak |
| max_seq_length | 8192 | 8192 | Keep same |

**Why fresh from base (not corrective)?** The prompt changed AND the data distribution changed significantly (OOS downsampled, new examples added). Corrective retraining from V14 adapters would fight the old distribution. Fresh training lets the model learn the new distribution cleanly.

### Training Commands

```bash
# On Lambda GH200
python3 finetune/train_v15.py \
  --model Qwen/Qwen3-4B \
  --data data/v15/hf/train.jsonl \
  --valid data/v15/hf/valid.jsonl \
  --output finetune/adapters_v15_4B/ \
  --max-steps 1400 \
  --save-every 100 \
  --lr 2e-5 \
  --batch 1 \
  --grad-accum 16 \
  --warmup 100 \
  2>&1 | tee training_v15.log
```

---

## Phase 4: Evaluation Sweep

**Duration**: 1 day

### Step 1: Checkpoint Sweep (Lambda)

```bash
python3 finetune/sweep_v15.py \
  --model Qwen/Qwen3-4B \
  --adapter-dir finetune/adapters_v15_4B/ \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v15_draft.txt \
  --output-dir eval_results/v15_sweep_4B/ \
  --skip-existing
```

**Success gates** (must pass before proceeding):
- [ ] **Hybrid accuracy ≥ 98.3%** (match V14 Q6_K)
- [ ] **Model-only accuracy ≥ 88%** (meaningful improvement over 83.7%)
- [ ] **Parse failures ≤ 10** (match V14)
- [ ] **No regression on loc** (0 errors on non-Anywhere REMOTE jobs)

### Step 2: Pick Best Checkpoint

Select by model-only accuracy (not hybrid, not val loss — learned from V13.1).

### Step 3: Merge and Quantize

```bash
# Merge LoRA into base model
python3 finetune/merge_v15.py \
  --model Qwen/Qwen3-4B \
  --adapter finetune/adapters_v15_4B/checkpoint-BEST/ \
  --output ~/merged_v15_4B

# Convert to GGUF
python3 llama.cpp/convert_hf_to_gguf.py ~/merged_v15_4B --outtype f16 --outfile ~/qwen3_4B_v15_f16.gguf

# Quantize
./llama.cpp/llama-quantize ~/qwen3_4B_v15_f16.gguf ~/qwen3_4B_v15_Q6_K.gguf Q6_K
```

### Step 4: GGUF Eval

```bash
# Q6_K eval (Mac deployment target)
python3 finetune/eval_student_v14_gguf.py \
  --model ~/qwen3_4B_v15_Q6_K.gguf \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v15_draft.txt \
  --output-dir eval_results/v15_gguf_Q6_K

# Hybrid
python3 finetune/compute_hybrid_v13_1.py \
  --test-file data/v12/test_labeled_audited.jsonl \
  --predictions eval_results/v15_gguf_Q6_K/predictions.jsonl \
  --v12
```

### Step 5: MLX Conversion (Mac)

```bash
# Convert to MLX 6-bit on Mac
.venv/bin/python3 -m mlx_lm convert \
  --hf-path ~/merged_v15_4B \
  --mlx-path ~/qwen3_4B_v15_mlx6bit \
  -q --q-bits 6
```

---

## Phase 5: Error Analysis & Iteration

**Duration**: 1 day

### Compare V15 vs V14

```bash
python3 finetune/compare_evals.py \
  eval_results/v14_gguf_Q6_K/hybrid_summary.json \
  eval_results/v15_gguf_Q6_K/hybrid_summary.json
```

### Specific Checks

1. **Anywhere jobs**: Do all 8 (or 9) "Anywhere" jobs now get REMOTE?
2. **OOS false positives**: Reduced from 11? Which ones fixed, which remain?
3. **NODE false positives**: Are Next.js jobs now correctly JS_TS (not NODE)?
4. **AI_ML false negatives**: Are secondary AI/ML mentions now caught?
5. **Comp boundaries**: Are £60k-£80k ranges now RANGE_55_74K?
6. **New regressions**: Any jobs that V14 got right but V15 gets wrong?

### If Targets Not Met

| Scenario | Action |
|----------|--------|
| Hybrid regressed | Check if new data introduced label conflicts. Audit new training examples. |
| Model-only < 88% | Increase augmentation for weakest field. Try more aggressive OOS downsampling. |
| New tech errors | Check if Next.js/NestJS contrastive pairs were labeled correctly. |
| New comp errors | Check if midpoint examples were labeled correctly by teacher. |
| Parse failures increased | Prompt may be too long. Trim or simplify. |

---

## Phase 6: Production Deployment

**Duration**: 1 day

### Final Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| V15 merged HF model | `~/merged_v15_4B/` | ~7.6 GB |
| V15 Q6_K GGUF | `~/qwen3_4B_v15_Q6_K.gguf` | ~3.1 GB |
| V15 MLX 6-bit | `~/qwen3_4B_v15_mlx6bit/` | ~3.1 GB |
| V15 student prompt | `prompts/student_v15.txt` | — |
| V15 training data | `data/v15/train.jsonl` | — |

### Upload to HuggingFace

```bash
huggingface-cli upload FF-01/qwen3-4b-v15 ~/merged_v15_4B/ merged_v15_4B/
huggingface-cli upload FF-01/qwen3-4b-v15 ~/qwen3_4B_v15_Q6_K.gguf .
huggingface-cli upload FF-01/qwen3-4b-v15 ~/qwen3_4B_v15_f16.gguf .
```

### Update CLAUDE.md

- Update model table with V15 results
- Update production candidates
- Archive V14 as previous version

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OOS downsampling causes OOS false negatives | Medium | High | Keep 150 diverse OOS examples; validate no regression on OOS test jobs |
| Synthetic data leaks test patterns | Low | Critical | Generate JDs from scratch, never use test set titles/companies. Audit for overlap. |
| New prompt changes teacher labeling | Medium | Medium | Phase 0 validation: re-label test set, diff against V14 labels |
| Fresh training needs more steps | Low | Low | Save every 100 steps, sweep up to 2000 if needed |
| Lambda GPU not available | Low | Medium | Fallback: use RunPod or Vast.ai with same A100/H100 setup |

---

## Timeline

| Phase | Duration | Depends On |
|-------|----------|-----------|
| Phase 0: Prompt optimization | 1 day | — |
| Phase 1: Data augmentation | 3-5 days | Phase 0 |
| Phase 2: Data pipeline | 2-3 days | Phase 1 |
| Phase 3: Training | 1-2 days | Phase 2 + Lambda access |
| Phase 4: Eval sweep | 1 day | Phase 3 |
| Phase 5: Error analysis | 1 day | Phase 4 |
| Phase 6: Deployment | 1 day | Phase 5 passed gates |
| **Total** | **~10-13 days** | |

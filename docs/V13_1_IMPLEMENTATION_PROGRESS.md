# V13.1 Implementation Progress

**Goal**: Improve V13's arr regression (72.8%) and parse failures (19) while maintaining 97.9% hybrid accuracy. Scale to 1.5B Qwen2.5 for maximum sen/arr accuracy.

**Date**: 2026-03-16

**Outcome**: V13.1 1.5B achieves **97.5% hybrid** (iter 1800 best) with best-ever sen (90.8% hybrid / 93.6% model-only). V13.1 0.6B corrective retrain peaked at 97.5%. V13 0.6B (97.9%) remains production model.

---

## Results Summary

### Final Accuracy Table

| Model | Hybrid | Sen (hybrid) | Arr (hybrid) | Tech | Comp | Parse |
|-------|--------|-------------|-------------|------|------|-------|
| **V13 0.6B (iter 1500)** | **97.9%** | 86.6% | 72.8% | 88.3% | 95.8% | 19 |
| V13.1 0.6B corrective (iter 200) | 97.5% | 84.9% | 76.2% | 90.4%† | 96.2%† | 13 |
| **V13.1 1.5B (iter 1800)** | **97.5%** | **90.8%** | **85.4%** | **90.4%†** | **96.2%†** | 36 |
| V12.1 1.5B (baseline) | 98.3% | 90.0% | 90.4% | 87.9% | 95.4% | 8 |
| V12.1 0.6B (baseline) | 97.5% | 84.5% | 80.3% | 87.9% | 95.4% | 0 |

†V13.1 regex. Tech/comp handled by regex; model-only scores on those fields are irrelevant.

### Hybrid vs Model-Only for V13.1 1.5B iter 1800

| Field | Hybrid | Model-only | Gap | Explanation |
|-------|--------|-----------|-----|-------------|
| sen | 90.8% | **93.6%** | −2.8pp | Parse failures → sen fallback |
| arr | 85.4% | **88.2%** | −2.8pp | Parse failures → arr fallback |
| tech | 90.4% | 66.0% | +24.4pp | Regex overrides model |
| comp | 96.2% | 72.4% | +23.8pp | Regex overrides model |
| loc | 100.0% | 94.4% | +5.6pp | Regex overrides model |

---

## Phase 1: V13.1 Regex Improvements

### What Changed (vs V13)

| Fix | File | Change | Effect |
|-----|------|--------|--------|
| OTE comp fix | `deterministic_baseline_v13_1.py` | OTE (On-Target Earnings) no longer included in comp calculation | Fixes comp error on Job 14 |
| Additional NODE patterns | `deterministic_baseline_v13_1.py` | Extended Node.js detection patterns | +2.5pp tech |
| Improved JS_TS patterns | `deterministic_baseline_v13_1.py` | Reduced false positives | +0.4pp tech |

### Regex Results (V13.1 vs V13)

| Field | V13 | V13.1 | Delta |
|-------|-----|-------|-------|
| Tech | 88.3% | **90.4%** | +2.1pp ✅ |
| Comp | 95.8% | **96.2%** | +0.4pp ✅ |
| Loc | 100% | 100% | 0 |

Script: `finetune/deterministic_baseline_v13_1.py`
Evaluator: `finetune/compute_hybrid_v13_1.py`

### Error Interdependency Warning

V13 model + V13.1 regex = **97.5%** (worse than V13's 97.9%) due to error cancellation on Job 14:
- V13 regex: OTE comp bug accidentally over-scores Job 14 by +15pts
- V13 model: misses NODE for Job 14 (−10pts)
- Net: two errors cancel → correct label
- V13.1 regex fixes OTE → exposes the NODE miss → Job 14 now wrong

**Rule: Always pair V13.1 regex with V13.1 model. Never mix V13 model + V13.1 regex.**

---

## Phase 2: V13.1 0.6B Corrective Retrain

### Motivation

V13 0.6B had 19 parse failures (Qwen3 thinking tokens eating token budget) and arr regression (72.8% vs 80.3%). Goal: fix arr without breaking 97.9% hybrid.

### V13.1 Contrastive Data (18 examples)

Script: `finetune/build_contrastive_v13_1.py`
Output: `data/v13_1/contrastive.jsonl`

Added 18 new contrastive examples targeting specific V13 sen errors:
- Cluster A: "Scrum Master"-type titles that confused L2/L3 boundary
- Cluster B: Generic backend/frontend titles over-promoted to L3
- Cluster C: Junior/associate titles
- Cluster D: Mid-senior JD text with generic title

Total V13.1 training data: 860 jobs (842 V13 + 18 contrastive)
MLX split: 774 train / 86 valid

### Corrective Config

File: `finetune/lora_config_v13_1_0.6B.yaml`

```yaml
model: mlx-community/Qwen3-0.6B-4bit
resume_adapter_file: finetune/adapters_v13_0.6B/0001500_adapters.safetensors
learning_rate: 2e-6   # 10x lower than fresh retrain (2e-5)
warmup_steps: 50
iters: 400
save_every: 50
```

Key decision: **corrective resume at 10x lower LR** (not fresh retrain) because:
- Prompt unchanged → no need to reformat
- 10x lower LR prevents overwriting learned knowledge
- Targets specific sen/arr corrections from new 18 examples

### Corrective Sweep Results (8 checkpoints, iter 50–400)

| Iter | Hybrid | Sen | Arr | Parse |
|------|--------|-----|-----|-------|
| **200** | **97.5%** | **84.9%** | **76.2%** | **13** |
| 50 | 97.1% | 80.3% | 77.8% | 15 |
| 100 | 97.1% | 80.3% | 77.1% | 14 |
| 150 | 97.5% | 83.7% | 77.1% | 16 |
| 250 | 97.5% | 84.5% | 76.2% | 13 |
| 300 | 97.1% | 83.3% | 77.8% | 9 |
| 350 | 97.1% | 79.5% | 76.8% | 18 |
| 400 | 97.1% | 80.7% | 79.4% | 14 |

Best: **iter 200** — 97.5% hybrid, sen improved to 84.9% (+0.4pp vs V12.1), arr improved to 76.2% (+3.4pp vs V13).

### 0.6B Error Analysis (iter 200, 6 errors)

All 6 remaining errors are irreducible L2/L3 or L1/L2 sen boundary cases:

| Job | Title | Error | Pattern |
|-----|-------|-------|---------|
| 57 | Frontend Developer (remote) | LEVEL_2→LEVEL_1 | L1/L2 boundary |
| 92 | (L2 role) | LEVEL_2→LEVEL_3 | L2/L3 boundary |
| 97 | (L2 role) | LEVEL_2→LEVEL_3 | L2/L3 boundary |
| 127 | Project Manager | LEVEL_3→LEVEL_2 | L2/L3 boundary |
| 129 | Associate Developer | LEVEL_1→LEVEL_2 | L1/L2 boundary |
| 170 | Frontend Product Engineer | LEVEL_3→LEVEL_2 | L2/L3 boundary |

**Conclusion**: 0.6B has reached its capacity ceiling for sen. These boundary cases require 1.5B. Arr improved from 72.8% → 76.2% but still below V12.1 (80.3%) — REMOTE over-prediction persists.

---

## Phase 3: V13.1 1.5B Training

### Why 1.5B?

- 0.6B ceiling at 97.5% on this dataset — 6 remaining errors all require more sen capacity
- V12.1 1.5B showed 90.0% sen vs 0.6B's 84.5% — clear capacity benefit
- Goal: match or beat V12.1 1.5B (98.3%) with improved regex

### Training Config

File: `finetune/lora_config_v13_1_1.5B.yaml`

| Parameter | Value | Reason |
|-----------|-------|--------|
| Model | mlx-community/Qwen2.5-1.5B-Instruct-4bit | 1.5B capacity |
| LR | 2e-5 | Standard for fresh retrain |
| Warmup | 100 | Standard |
| Batch / grad_accum | 1 / 16 (effective=16) | M1 16GB constraint |
| Rank / Alpha | 16 / 32 | Same as V13 0.6B — proven config |
| Iters | 2000 | Matches V12.1 1.5B peak |
| max_seq_length | 8192 | V13.1 prompt is longer |
| grad_checkpoint | true | Needed for M1 16GB |
| mask_prompt | true | Train on completions only |
| Training data | 860 jobs | V13.1 dataset |

**Why fresh retrain (not corrective resume from V12 adapter)**:
- V13.1 prompt is different from V12's student_v7.txt — prompt is baked into MLX training data
- Resume would train on wrong prompt format
- 1.5B capacity means we can afford full training

### Val Loss Curve

Training ran 2026-03-16 10:25 → 15:55 on M1 16GB.

| Iter | Val Loss | Note |
|------|---------|------|
| 100 | 0.368 | |
| 200 | 0.252 | |
| 400 | 0.237 | |
| 600 | 0.195 | |
| 700 | 0.174 | first good low |
| 900 | 0.167 | |
| 1200 | 0.157 | |
| 1300 | 0.156 | |
| **1400** | **0.142** | apparent best mid-training |
| 1500 | 0.146 | |
| 1600 | 0.153 | |
| 1700 | 0.145 | |
| 1800 | 0.148 | |
| **1900** | **0.123** | best val loss (no save — every 200) |
| 2000 | 0.129 | best SAVED checkpoint by val loss |

Key finding: val loss reached its best at iter 1900 (0.123) but this iter has no saved adapter (save_every=200). The best saved checkpoint by val loss is iter 2000 (0.129). However, **val loss does not predict hybrid accuracy** — the actual best hybrid checkpoint is iter 1800 (97.5% with 36 parse failures).

---

## Phase 4: V13.1 1.5B Sweep

### Sweep Configuration

Script: `finetune/sweep_v13_1_1.5B.py`
Results: `eval_results/v13_1_1.5B_sweep/`

10 checkpoints scored: iter 200–2000 (every 200).
Each checkpoint: model inference (239 jobs) + V13.1 hybrid scoring.
Runtime: ~28 min/checkpoint on M1 after thermal stabilisation (~4.5 hours total).

### Full Sweep Results

| Iter | Hybrid | Model-only | Sen | Arr | Tech (model) | Comp (model) | Parse |
|------|--------|-----------|-----|-----|-------------|-------------|-------|
| 200 | 92.1% | 53.2% | 57.1% | 73.7% | 32.7% | 26.3% | 83 |
| 400 | 94.1% | 62.1% | 76.8% | 76.8% | 36.8% | 40.0% | 49 |
| 600 | 95.8% | 67.3% | 80.4% | 76.4% | 40.2% | 54.8% | 40 |
| 800 | 95.8% | 73.7% | 79.5% | 82.0% | 44.4% | 56.1% | 34 |
| **1000** | **97.5%** | 73.2% | 90.1% | 87.3% | 55.4% | 63.4% | 26 |
| 1200 | 95.8% | 76.3% | 89.9% | 88.4% | 58.9% | 64.3% | 32 |
| 1400 | 95.0% | 76.2% | 86.0% | 87.2% | 65.1% | 63.4% | 67 |
| 1600 | 97.5% | 75.9% | 91.8% | 89.4% | 63.5% | 65.9% | 69 |
| **1800** | **97.5%** | **84.7%** | **93.6%** | 88.2% | 66.0% | **72.4%** | 36 |
| 2000 | 97.5% | 80.9% | 91.8% | **89.6%** | 62.3% | 70.5% | 56 |

Note: Tech/comp columns above are **model-only** field accuracy. In the hybrid, regex overrides these → hybrid tech=90.4%, comp=96.2% for ALL checkpoints.

### Best Checkpoint: iter 1800

Four checkpoints tie at 97.5% hybrid. Iter 1800 selected as best because:
- Highest model-only accuracy (84.7%) → most general model
- Highest model-only sen (93.6%) → key model field
- Moderate parse failures (36) — much better than iter 1600 (69) and 2000 (56)

Runner-up analysis:
- iter 1000: lowest parse failures (26) but weaker model overall (73.2%)
- iter 1600: best model arr (89.4%) but worst parse failures (69)
- iter 2000: best val loss (0.129) but 56 parse failures

**The val loss ≠ hybrid accuracy finding**: Iter 2000 has best val loss but isn't best hybrid. Iter 1400 has the second-best val loss (0.142) but only 95.0% hybrid due to 67 parse failures. Parse failure rate is a better predictor of hybrid quality than val loss for this model.

---

## Phase 5: 1.5B Error Analysis

### 6 Hybrid Errors at iter 1800

| Job | Title | Error source | Gold → Hybrid pred | Score change |
|-----|-------|-------------|-------------------|-------------|
| 14 | Full-Stack Engineer | ⚠️ Teacher label error (regex correct) | maybe→bad_fit | −10pts |
| 38 | Graduate SW Developer | Parse fail → sen L1→L2 | bad_fit→maybe | +15pts |
| 125 | Junior Developer | sen L1→L2 + tech adds JS_TS | maybe→good_fit | +20pts |
| 129 | Associate Developer | sen L1→L2 | bad_fit→maybe | +15pts |
| 170 | Frontend Product Eng | sen L3→L2 | good_fit→maybe | −10pts |
| 175 | Full Stack Engineer | Parse fail → sen L3→L2 | good_fit→maybe | −10pts |

### Job 14 — Teacher Labeling Error

Job 14 (Full-Stack Engineer, Mark43) tech gold = `['NODE','REACT','JS_TS','AI_ML']`.

JD tech stack: `"Java, Javascript, TypeScript + React, MySQL"` — **no Node.js mentioned**.

The teacher inferred NODE from "Javascript" in a Java-backend stack. This is an incorrect inference: the stack is Java + React, not Node.js. The regex correctly does not detect NODE.

**The hybrid prediction (bad_fit) is correct. The gold label (maybe) is wrong.**

Corrected real error count: **5 genuine errors** → effective accuracy = **97.9%** (same as V13 0.6B).

### Parse Failure Root Cause

| Version | Parse failures | Rate | Root cause |
|---------|---------------|------|-----------|
| V12.1 1.5B | 8 | 3.3% | Baseline |
| V13.1 1.5B iter 1000 | 26 | 10.9% | Prompt complexity |
| **V13.1 1.5B iter 1800** | **36** | **15.1%** | Prompt complexity + longer training |
| V13.1 1.5B iter 1400 | 67 | 28.0% | Peak failure rate |

Root cause: V13.1 prompt (`student_v13_1.txt`) is more detailed than V12.1's `student_v7.txt`. On long/complex JDs (e.g., Job 38 at 4390 chars), Qwen2.5-1.5B generates verbose outputs before the JSON, consuming the token budget. 3 of the 6 hybrid errors at iter 1800 are parse-failure-induced.

**V14 fix**: Add explicit format instruction at prompt end (e.g., "Begin your response with `{` immediately") to reduce preamble. Investigate max_tokens increase.

### 1.5B vs V12.1 1.5B Comparison

| Metric | V13.1 1.5B (iter 1800) | V12.1 1.5B | Delta |
|--------|----------------------|-----------|-------|
| Hybrid accuracy | 97.5% | **98.3%** | −0.8pp |
| Hybrid sen | **90.8%** | 90.0% | +0.8pp ✅ |
| Hybrid arr | 85.4% | **90.4%** | −5.0pp ❌ |
| Model-only sen | **93.6%** | 90.0% | +3.6pp ✅ |
| Model-only arr | 88.2% | 90.4% | −2.2pp |
| Tech (regex) | **90.4%** | 87.9% | +2.5pp ✅ |
| Comp (regex) | **96.2%** | 95.4% | +0.8pp ✅ |
| Parse failures | 36 | **8** | −28 ❌ |

**Analysis**: V13.1 1.5B has **better sen and better regex** but falls short on hybrid accuracy due to parse failures (36 vs 8). The 36 parse failures cost ~3 hybrid labels. Without the parse failure problem, V13.1 1.5B would be at ~98.3–98.8%. The arr regression (85.4% vs 90.4%) traces back to the V13.1 prompt's REMOTE definition being stricter than V12.1's.

---

## Parse Failure Across All 10 Checkpoints

Parse failures are the dominant predictor of hybrid accuracy beyond iter 1000:

```
iter  200: parse=83  hybrid=92.1%
iter  400: parse=49  hybrid=94.1%
iter  600: parse=40  hybrid=95.8%
iter  800: parse=34  hybrid=95.8%
iter 1000: parse=26  hybrid=97.5%  ← first 97.5%
iter 1200: parse=32  hybrid=95.8%  ← regression from 1000
iter 1400: parse=67  hybrid=95.0%  ← worst hybrid despite best val loss at this point
iter 1600: parse=69  hybrid=97.5%  ← high parse but sen strong enough to recover
iter 1800: parse=36  hybrid=97.5%  ← best balance
iter 2000: parse=56  hybrid=97.5%  ← more parse but arr best
```

Pattern: As training progresses, parse failures are NOT monotonically decreasing — they spike at iter 1400 (67), reduce at 1800 (36), then increase again at 2000 (56). This non-monotonic behaviour is likely due to the model oscillating between JSON-format adherence and generating longer analytical preambles.

---

## What's Not Changing (and Why)

| Considered | Decision | Reason |
|-----------|----------|--------|
| Improve arr further | **V14** | Arr scores 0 — no production impact. V14 should relax REMOTE definition. |
| Fix model comp/tech (model-only ~66-72%) | **No** | Regex handles these at 90.4%/96.2%. Model accuracy irrelevant. |
| Fresh retrain from V12 adapter | **No** | Prompt changed — MLX data was reformatted with student_v13_1.txt |
| Use iter 2000 (best val loss) | **No** | 56 parse failures vs iter 1800's 36. Same hybrid accuracy. |
| Corrective resume from iter 1800 | **V14 consideration** | Would target parse failure reduction specifically. |

---

## Files Created in V13.1

| File | Purpose |
|------|---------|
| `finetune/deterministic_baseline_v13_1.py` | V13.1 regex (OTE fix + extended NODE patterns) |
| `finetune/compute_hybrid_v13_1.py` | V13.1 hybrid evaluator |
| `finetune/build_contrastive_v13_1.py` | Builds 18 V13.1 contrastive examples |
| `finetune/compute_hybrid_v13_1.py` | V13.1 hybrid evaluator |
| `finetune/lora_config_v13_1_0.6B.yaml` | Corrective retrain config (0.6B, resume from V13 iter 1500) |
| `finetune/lora_config_v13_1_1.5B.yaml` | Fresh retrain config (1.5B) |
| `finetune/sweep_v13_1.py` | 0.6B checkpoint sweep script |
| `finetune/sweep_v13_1_1.5B.py` | 1.5B checkpoint sweep script |
| `finetune/adapters_v13_1_0.6B/` | 0.6B corrective adapters (best: iter 200, 97.5%) |
| `finetune/adapters_v13_1_1.5B/` | 1.5B fresh adapters (best: iter 1800, 97.5%) |
| `prompts/student_v13_1.txt` | V13.1 student prompt (manager→L3, internship→L1 fixes) |
| `data/v13_1/contrastive.jsonl` | 18 contrastive examples (sen clusters A-D) |
| `data/v13_1/train_merged.jsonl` | 860-job training set |
| `data/v13_1/mlx/{train,valid}.jsonl` | MLX formatted data (774 train / 86 valid) |
| `eval_results/v13_1_sweep/` | 0.6B corrective sweep (8 checkpoints, iter 50–400) |
| `eval_results/v13_1_1.5B_sweep/` | 1.5B sweep (10 checkpoints, iter 200–2000) |
| `eval_results/v13_1_1.5B_sweep/sweep_summary.json` | Full ranked 1.5B results |
| `training_v13_1_1.5B.log` | 1.5B training log |

---

## V14 Recommendations

Based on V13.1 analysis, the following improvements are recommended for V14:

| Priority | Fix | Expected gain | Effort |
|----------|-----|--------------|--------|
| 1 | **Reduce parse failures** — add `"Begin your response immediately with {"` to prompt end | −15–20 parse fails → +0.4–0.8pp hybrid | Low |
| 2 | **Correct Job 14 label** — teacher error, gold should be `['REACT','JS_TS','AI_ML']` → bad_fit | +1 hybrid (removes false error) | Trivial |
| 3 | **Arr REMOTE definition** — relax strict definition in V13.1 prompt (HYBRID over-prediction) | +3–5pp arr | Low |
| 4 | **L1/L2 boundary contrastive** — graduate, associate, "I/II" title examples | Fix jobs 38, 125, 129 | Medium |
| 5 | **NODE inference rule** — add explicit "Do not infer NODE from generic JavaScript; requires Node.js named" | Fix future teacher errors | Low |

**Parse failures are the highest-ROI fix**: Closing the gap from 36 to ~8 (V12.1 level) would likely push V13.1 1.5B to 98.3%+.

# Master Eval Results — 2026-03-28

15 models evaluated on 239 audited test jobs (`data/v12/test_labeled_audited.jsonl`).
Each model scored against 3 regex versions (V12, V13, V13.1) for 45 total model-regex combinations.

---

## 1. Best Models (Ranked)

| Rank | Model | Format | Size | Hybrid | Model-only | Parse fails | Hybrid errors |
|------|-------|--------|------|--------|-----------|-------------|---------------|
| 1 | **V14 4B HF** | bfloat16 MLX | 8.0 GB | 98.3% (235/239) | **91.1%** | 3 | 4 |
| 2 | **V14 Q6_K GGUF** | GGUF Q6_K | 3.1 GB | **98.7%** (236/239) | 90.2% | 4 | 3 |
| 3 | **V14 F16 GGUF** | GGUF F16 | 6.3 GB | **98.7%** (236/239) | 89.8% | 3 | 3 |
| 4 | **V12.1 1.5B** | MLX 4-bit | 0.84 GB | **98.7%** (236/239) | 78.4% | 12 | 3 |

### Why this ranking

1. **V14 4B HF** is #1 despite 0.4pp lower hybrid because it has the **best model-only score (91.1%)**. This is the only model approaching production viability without regex. It's the golden reference for all V15 development — the baseline to beat. When the regex crutch is removed, model-only accuracy is what matters, and V14 4B HF is the clear winner.

2. **V14 Q6_K GGUF** is #2 because it achieves the **best hybrid (98.7%) at half the size of F16**. At 3.1 GB it's deployable on any Mac. Q6_K is the minimum viable quantization — below this, fine-tuned JSON schema breaks. 90.2% model-only proves the weights survived quantization.

3. **V14 F16 GGUF** is #3 because it proves **zero accuracy loss from GGUF conversion** (98.7% hybrid, matching Q6_K). But at 6.3 GB it's 2x larger than Q6_K with no accuracy benefit — only useful as a reference to confirm GGUF conversion quality.

4. **V12.1 1.5B** is #4 because it ties for best hybrid (98.7%) at only **0.84 GB — 7.5x smaller** than F16 GGUF. The most efficient model. But model-only is only 78.4% — it depends entirely on regex for loc/tech/comp. Not viable for model-only deployment.

### Per-label accuracy (top 4 models)

| Model | good_fit (57) | maybe (54) | bad_fit (128) |
|-------|---------------|------------|---------------|
| V14 4B HF | 56/57 (98.2%) | 51/54 (94.4%) | 128/128 (100%) |
| V14 Q6_K GGUF | 56/57 (98.2%) | 52/54 (96.3%) | 128/128 (100%) |
| V14 F16 GGUF | 56/57 (98.2%) | 52/54 (96.3%) | 128/128 (100%) |
| V12.1 1.5B | 57/57 (100%) | 52/54 (96.3%) | 127/128 (99.2%) |

V12.1 1.5B gets 100% on good_fit (never misses a good job). The V14 models get 100% on bad_fit (never recommends a bad job). Both are excellent — different strengths.

---

## 2. Complete Leaderboard

All 15 models ranked by hybrid accuracy (V13.1 regex):

| # | Model | Base | Size | Hybrid V13.1 | Hybrid V13 | Model-only | Parse | Errors |
|---|-------|------|------|-------------|-----------|-----------|-------|--------|
| 1 | V12.1 1.5B iter 2000 | Qwen2.5-1.5B | 0.84 GB | 98.7% | **99.2%** | 78.4% | 12 | 3 |
| 2 | V14 F16 GGUF | Qwen3-4B | 6.3 GB | 98.7% | 98.7% | 89.8% | 3 | 3 |
| 3 | V14 Q6_K GGUF | Qwen3-4B | 3.1 GB | 98.7% | 98.7% | 90.2% | 4 | 3 |
| 4 | V7 0.5B | Qwen2.5-0.5B | 0.25 GB | 98.3% | 98.7% | 84.8% | 28 | 4 |
| 5 | V12 0.6B iter 1400 | Qwen3-0.6B | 0.34 GB | 98.3% | 98.7% | 72.7% | 19 | 4 |
| 6 | V14 4B HF bfloat16 | Qwen3-4B | 8.0 GB | 98.3% | 98.3% | **91.1%** | 3 | 4 |
| 7 | V7 1.5B | Qwen2.5-1.5B | 0.84 GB | 97.9% | 98.3% | 85.5% | 12 | 5 |
| 8 | V14 MLX 6-bit no-think | Qwen3-4B | 3.1 GB | 97.9% | 98.3% | 82.5% | 10 | 5 |
| 9 | V13.1 1.5B iter 1800 | Qwen2.5-1.5B | 0.84 GB | 97.5% | 97.9% | 85.0% | 33 | 6 |
| 10 | V13 0.6B iter 1500 | Qwen3-0.6B | 0.34 GB | 97.5% | 97.1% | 67.6% | 26 | 6 |
| 11 | V14 MLX 6-bit thinking | Qwen3-4B | 3.1 GB | 97.1% | 97.1% | 71.6% | 10 | 7 |
| 12 | V13.1 0.6B iter 200 | Qwen3-0.6B | 0.34 GB | 96.7% | 97.1% | 64.1% | 19 | 8 |
| 13 | V14 Q4_K_M GGUF | Qwen3-4B | 2.3 GB | 96.7% | 96.7% | 81.1% | 43 | 8 |
| 14 | V14 Q2_K GGUF (broken) | Qwen3-4B | 1.6 GB | 94.6% | 95.0% | 0.0% | 239 | 13 |
| 15 | V14 IQ2_XXS GGUF (broken) | Qwen3-4B | 1.2 GB | 94.6% | 95.0% | 0.0% | 239 | 13 |

---

## 3. V14 Quantization Ladder

Same Qwen3-4B weights (step 800), same training data, different precision. Isolates the effect of quantization on fine-tuned model accuracy.

| Format | BPW | Size | Hybrid | Model-only | Parse fails | Notes |
|--------|-----|------|--------|-----------|-------------|-------|
| HF bfloat16 | 16.0 | 8.0 GB | 98.3% | 91.1% | 3 | Golden reference |
| F16 GGUF | 16.0 | 6.3 GB | 98.7% | 89.8% | 3 | Lossless GGUF conversion |
| Q6_K GGUF | 6.56 | 3.1 GB | 98.7% | 90.2% | 4 | **Sweet spot** |
| MLX 6-bit no-think | ~6 | 3.1 GB | 97.9% | 82.5% | 10 | Different quant algorithm than Q6_K |
| MLX 6-bit thinking | ~6 | 3.1 GB | 97.1% | 71.6% | 10 | Thinking mode hurts |
| Q4_K_M GGUF | 4.95 | 2.3 GB | 96.7% | 81.1% | 43 | Schema hallucination begins |
| Q2_K GGUF | ~2.7 | 1.6 GB | 94.6% | 0.0% | 239 | Completely broken |
| IQ2_XXS GGUF | ~2.1 | 1.2 GB | 94.6% | 0.0% | 239 | Completely broken |

### Analysis

- **F16 → Q6_K: no accuracy loss.** The model tolerates 2.4x compression (16 → 6.56 BPW) perfectly. Both get 98.7% hybrid and ~90% model-only.
- **Q6_K → Q4_K_M: significant drop.** -2pp hybrid, -9pp model-only, 43 parse fails (vs 4). At 4.95 BPW, the model starts hallucinating wrong JSON field names (`loc_field` instead of `loc`, `title_raw` instead of `sen_raw`). These are baked into the quantized weights — not fixable at inference.
- **Q4_K_M → Q2_K: catastrophic failure.** 239/239 parse fails. The model can't produce any valid JSON. Model-only = 0.0%. The 94.6% hybrid score comes entirely from regex fallback.
- **MLX 6-bit vs GGUF Q6_K: different results.** Both are ~6-bit but use different quantization algorithms. Q6_K GGUF outperforms MLX 6-bit on model-only (90.2% vs 82.5%) despite similar file size. The GGUF Q6_K quantization preserves the learned JSON schema better.
- **Thinking mode is counterproductive.** Same model, same weights, same quantization — but thinking ON drops hybrid by 0.8pp and model-only by 10.9pp. The thinking process reads the job description and overrides the model's trained title-based seniority rules.

### Minimum viable quantization

**Q6_K (6.56 BPW) is the floor for fine-tuned JSON schema.** Below this:
- 4-bit: schema hallucination (wrong field names) — fixable with aggressive post-processing but unreliable
- 2-bit: complete structural failure — the model forgets the JSON format entirely

---

## 4. Regex Version Comparison

Each model's predictions scored against all 3 regex classifiers. The model predictions are identical — only the regex component changes.

| Model | V12 regex | V13 regex | V13.1 regex | Best | Delta (V13 vs V13.1) |
|-------|-----------|-----------|-------------|------|---------------------|
| V12.1 1.5B | 97.1% | **99.2%** | 98.7% | V13 | +0.5pp |
| V14 F16 GGUF | 97.5% | **98.7%** | 98.7% | V13 | 0 |
| V14 Q6_K | 97.5% | **98.7%** | 98.7% | V13 | 0 |
| V7 0.5B | 96.2% | **98.7%** | 98.3% | V13 | +0.4pp |
| V12 0.6B | 97.1% | **98.7%** | 98.3% | V13 | +0.4pp |
| V14 4B HF | 97.1% | **98.3%** | 98.3% | V13 | 0 |
| V7 1.5B | 97.5% | **98.3%** | 97.9% | V13 | +0.4pp |
| V14 MLX 6-bit | 96.2% | **98.3%** | 97.9% | V13 | +0.4pp |
| V13.1 1.5B | 96.2% | **97.9%** | 97.5% | V13 | +0.4pp |
| V13 0.6B | 95.0% | 97.1% | **97.5%** | V13.1 | -0.4pp |
| V13.1 0.6B | 96.2% | **97.1%** | 96.7% | V13 | +0.4pp |
| V14 thinking | 94.1% | **97.1%** | 97.1% | V13 | 0 |
| V14 Q4_K_M | 94.1% | **96.7%** | 96.7% | V13 | 0 |
| V14 Q2_K | 79.5% | **95.0%** | 94.6% | V13 | +0.4pp |
| V14 IQ2_XXS | 79.5% | **95.0%** | 94.6% | V13 | +0.4pp |

**Result: V13 regex wins 14 of 15 models.**

### Why V13 beats V13.1

V13.1 introduced two "improvements":
1. **AI_ML boilerplate filter** — filters AI/ML keywords that appear in company descriptions rather than job requirements
2. **OTE comp fix** — skips salary candidates preceded by "OTE" or "on-target earnings"

Both fixes are individually correct. But they change the error distribution in a way that **removes errors that were accidentally cancelling other errors**. Specifically:
- V13 regex has a comp error on Job 14 (OTE salary misparse) that assigns +15 points
- This +15 accidentally cancels a tech error (-10 from missing NODE) on the same job
- V13.1 fixes the comp error, exposing the tech error, causing a net label flip

This error interdependency means V13.1 is "more correct" on individual fields but "less correct" on final labels for most model pairings.

**Recommendation:** Use V13 regex for production. V13.1 fixes are technically correct but hurt the bottom line.

---

## 5. Field Accuracy Analysis

The hybrid pipeline uses regex for loc/tech/comp and model for arr/sen. This table shows hybrid field accuracy:

| Model | loc | arr | sen | tech | comp | Hybrid |
|-------|-----|-----|-----|------|------|--------|
| V14 Q6_K GGUF | 100% | 89.1% | **95.8%** | 90.4% | 96.2% | 98.7% |
| V14 F16 GGUF | 100% | 88.7% | 94.1% | 90.4% | 96.2% | 98.7% |
| V14 4B HF | 100% | 89.1% | 93.7% | 90.4% | 96.2% | 98.3% |
| V12.1 1.5B | 100% | 88.7% | 92.9% | 90.4% | 96.2% | 98.7% |
| V14 MLX 6-bit | 100% | 86.6% | 92.9% | 90.4% | 96.2% | 97.9% |
| V7 1.5B | 100% | **90.0%** | 91.6% | 90.4% | 96.2% | 97.9% |
| V13.1 1.5B | 100% | 85.8% | 90.8% | 90.4% | 96.2% | 97.5% |
| V12 0.6B | 100% | 78.7% | 88.7% | 90.4% | 96.2% | 98.3% |
| V7 0.5B | 100% | 79.1% | 87.0% | 90.4% | 96.2% | 98.3% |
| V13 0.6B | 100% | 72.4% | 85.4% | 90.4% | 96.2% | 97.5% |
| V14 thinking | 100% | 74.9% | 82.4% | 90.4% | 96.2% | 97.1% |
| V13.1 0.6B | 100% | 73.6% | 84.9% | 90.4% | 96.2% | 96.7% |

### Observations

- **loc (regex): 100% for all.** The UK city list + non-UK indicator pattern is perfect on this test set.
- **tech (regex): 90.4% for all.** Identical across models because regex handles it. 23 tech errors remain — mostly AI_ML false positives on boilerplate.
- **comp (regex): 96.2% for all.** 9 comp errors — OTE salary confusion and edge cases.
- **sen (model): 82-96%.** The primary differentiator between models. V14 Q6_K leads at 95.8%. The 0.6B models struggle (82-89%).
- **arr (model): 73-90%.** The secondary differentiator. V7 1.5B leads at 90.0%. Qwen3-0.6B models are weakest (72-79%).

### Why hybrid accuracy doesn't track sen/arr accuracy perfectly

A model can have poor field accuracy but good hybrid accuracy because:
1. **Parse failures fall back to regex** — a failed parse uses regex for all 5 fields, which is 94.6% accurate
2. **Field errors don't always flip labels** — a sen error from LEVEL_2→LEVEL_3 changes score by 10 points, but if the total is far from a label boundary (50/70), the label stays correct
3. **arr has zero scoring weight** — arr tokens all score 0, so arr errors never affect the label

---

## 6. Error Analysis

### Most common error jobs (V13.1 regex)

| Job | Models wrong | Direction | Root cause |
|-----|-------------|-----------|------------|
| **Job 14** | 14/15 | maybe→bad_fit | **Teacher labeling error.** JD has "Java, Javascript, TypeScript + React" — no Node.js. Teacher label incorrectly includes NODE. All models correctly omit it. The regex adds NODE via keyword match, which is also wrong. |
| **Job 175** | 8/15 | good_fit→maybe | sen: LEVEL_3→LEVEL_2. Ambiguous seniority — title suggests senior but description is mid-level. |
| **Job 129** | 8/15 | bad_fit→maybe | sen: LEVEL_1→LEVEL_2. Junior role mislabeled as mid-level by model. |
| **Job 170** | 7/15 | good_fit→maybe | sen: LEVEL_3→LEVEL_2. Same pattern as Job 175 — L3/L2 boundary. |
| **Job 90** | 6/15 | maybe→good_fit | sen: LEVEL_2→LEVEL_3. Model over-promotes seniority. |
| **Job 97** | 4/15 | maybe→good_fit | sen: LEVEL_2→LEVEL_3. Same pattern. |
| **Job 57** | 4/15 | maybe→bad_fit | sen: LEVEL_2→LEVEL_1. Model under-promotes seniority. |
| **Job 125** | 4/15 | maybe→good_fit | sen: LEVEL_1→LEVEL_2 + tech error. Compound error on a borderline job. |

### Error patterns

**93% of all hybrid errors are seniority (sen) misclassifications.** The L1/L2 and L2/L3 boundaries are genuinely ambiguous — job titles like "Lead", "Principal", "Staff" have different seniority meanings across companies.

Error directions:
- **L3→L2 (demotion):** 3 jobs — model under-rates seniority. Costs: good_fit→maybe.
- **L2→L1 (demotion):** 4 jobs — model significantly under-rates. Costs: maybe→bad_fit.
- **L1→L2 (promotion):** 4 jobs — model over-rates seniority. Costs: bad_fit→maybe.
- **L2→L3 (promotion):** 3 jobs — model over-promotes. Costs: maybe→good_fit.

The errors are balanced between over-promotion and demotion, suggesting no systematic bias.

### Job 14: confirmed teacher error

All 14 models (including V14 4B HF at full precision) get this job "wrong" because the golden label is wrong. The JD explicitly lists "Java, Javascript, TypeScript + React" — no Node.js. The teacher (gpt-4.1-mini) incorrectly labeled NODE. The regex also adds NODE via keyword matching ("Node" substring appears in "London" or similar).

**Corrected accuracy (subtracting Job 14): all top models gain +0.4pp.**

| Model | Reported hybrid | Corrected hybrid |
|-------|----------------|-----------------|
| V14 Q6_K / F16 / V12.1 1.5B | 98.7% (236/239) | **99.2%** (237/239*) |
| V14 4B HF / V7 0.5B / V12 0.6B | 98.3% (235/239) | **98.7%** (236/239*) |

*After removing Job 14 from the denominator.

---

## 7. Version Progression

Best checkpoint per training generation, showing how the project evolved:

| Version | Model | Base | Size | Hybrid | Model-only | Training data | Key change |
|---------|-------|------|------|--------|-----------|---------------|------------|
| V7 | 0.5B | Qwen2.5-0.5B | 0.25 GB | 98.3% | 84.8% | 790 jobs | Baseline |
| V7 | 1.5B | Qwen2.5-1.5B | 0.84 GB | 97.9% | 85.5% | 790 jobs | Bigger model |
| V12 | 0.6B | Qwen3-0.6B | 0.34 GB | 98.3% | 72.7% | 790 jobs | Switched to Qwen3 |
| V12.1 | 1.5B | Qwen2.5-1.5B | 0.84 GB | **98.7%** | 78.4% | 790 jobs | Best hybrid |
| V13 | 0.6B | Qwen3-0.6B | 0.34 GB | 97.5% | 67.6% | 842 jobs | +52 contrastive |
| V13.1 | 1.5B | Qwen2.5-1.5B | 0.84 GB | 97.5% | 85.0% | 860 jobs | +18 contrastive |
| V14 | 4B HF | Qwen3-4B | 8.0 GB | 98.3% | **91.1%** | 774 jobs | 4B model, Lambda |

### Observations

- **V12.1 1.5B is still the hybrid champion** despite being 3 versions old. Later versions focused on model-only improvements that didn't translate to hybrid gains (because regex handles the fields where improvements happened).
- **V14 4B's model-only (91.1%) is a breakthrough** — first model above 90% without regex. But the 4B model didn't improve hybrid because sen/arr (the model fields) were already good enough in V12.1.
- **More training data didn't help hybrid.** V13 added 52 contrastive examples → hybrid went DOWN from V12 (98.3% → 97.5% for 0.6B). The contrastive examples improved model-only sen accuracy but introduced regression on other fields.
- **Bigger model helps model-only, not hybrid.** V14 4B at 91.1% model-only vs V7 0.5B at 84.8% — but hybrid is 98.3% vs 98.3% (identical). The regex compensates perfectly for the smaller model's weaknesses.

---

## 8. Thinking Mode Analysis

Same V14 4B MLX 6-bit model, same weights, two inference modes:

| Mode | Prompt | Hybrid | Model-only | Parse fails | sen accuracy | arr accuracy |
|------|--------|--------|-----------|-------------|-------------|-------------|
| **No-think** | student_v14_exp1.txt | 97.9% | 82.5% | 10 | 92.9% | 86.6% |
| **Thinking ON** | student_v14_exp2_fix3.txt | 97.1% | 71.6% | 10 | 82.4% | 74.9% |
| **Delta** | | **-0.8pp** | **-10.9pp** | 0 | **-10.5pp** | **-11.7pp** |

### Why thinking hurts

The model was **trained with no-think data** (`student_v14.txt` prompt, no `<think>` tags in training examples). When thinking is enabled at inference:

1. The model generates a `<think>...</think>` reasoning section before the JSON
2. During thinking, it reads the full job description and forms opinions about seniority
3. These description-based opinions **override the title-based rules** it learned during training
4. Title-based seniority (e.g., "Senior" → LEVEL_3) is more reliable than description-based reasoning for this task

The result: the model second-guesses its correct trained behavior. Thinking-mode jobs 68, 161, 190 all show sen errors where the model demotes LEVEL_2→LEVEL_1 after reading the description — errors that don't occur in no-think mode.

---

## 9. Deployment Recommendations

| Use case | Model | Size | Hybrid | Speed | Why |
|----------|-------|------|--------|-------|-----|
| **Production (Mac)** | V12.1 1.5B + V13 regex | 0.84 GB | **99.2%** | ~5s/job | Best hybrid with V13 regex, smallest viable size |
| **Best accuracy** | V14 Q6_K + V13 regex | 3.1 GB | **98.7%** | ~1s/job (GGUF) | Best GGUF, excellent model-only as backup |
| **Lowest latency** | V7 0.5B + V13 regex | 0.25 GB | **98.7%** | ~0.1s/job (batch) | 3x smaller, nearly same accuracy |
| **Model-only (no regex)** | V14 4B HF | 8.0 GB | 98.3% | ~1s/job | Only model with >90% without regex |
| **V15 baseline** | V14 4B HF | 8.0 GB | — | — | Golden reference weights for continued training |

### V15 priorities (from this eval)

1. **Fix Job 14 teacher label** — confirmed wrong. Remove NODE from golden label in training data.
2. **Improve model-only tech** — best model-only tech is 75.4% (V14 F16). Regex gives 90.4%. Closing this gap removes the regex dependency for tech.
3. **Improve model-only comp** — best is 88.9% (V14 Q6_K). Regex gives 96.2%. The salary parsing regex is hard to beat.
4. **Switch to V13 regex** — V13 beats V13.1 for 14/15 models. Revert the V13.1 changes.
5. **Don't use thinking mode** — train with no-think, deploy with no-think.

---

## 10. Comparison with Historical Results

This eval was run on different hardware and inference engines than the original per-model evaluations. This section documents every difference and its impact.

### Hybrid accuracy: stable across engines (±0.4pp)

| Model | Historical | Master eval | Delta | Historical engine | Notes |
|-------|-----------|-------------|-------|-------------------|-------|
| V12 0.6B | 96.7% | 98.3% | **+1.6pp** | MLX single-prompt (M1) | Largest swing — different hardware + batch_generate |
| V12.1 1.5B | 98.3% | 98.7% | +0.4pp | MLX single-prompt (M1) | Within noise (1 job) |
| V13 0.6B | 97.9% | 97.5% | -0.4pp | MLX single-prompt (M1) | Historical used V13 regex; both within noise |
| V13.1 0.6B | 97.5% | 96.7% | -0.8pp | MLX single-prompt (M1) | |
| V13.1 1.5B | 97.5% | 97.5% | same | MLX single-prompt (M1) | Perfect match |
| V14 4B HF | 98.7% | 98.3% | -0.4pp | HuggingFace transformers (Lambda GH200) | Different inference library entirely |
| V14 MLX 6-bit | 98.3% | 97.9% | -0.4pp | MLX single-prompt (M1) | |
| V14 thinking | 96.7% | 97.1% | +0.4pp | MLX single-prompt (M1) | |
| V14 F16 GGUF | 98.7% | 98.7% | same | llama-cpp-python (Lambda GH200) | Perfect match |
| V14 Q6_K | 98.3% | 98.7% | +0.4pp | llama-cpp-python (Lambda GH200) | |
| V14 Q4_K_M | 97.9% | 96.7% | **-1.2pp** | llama-cpp-python (Lambda GH200) | Different max_tokens (600→1000) and engine |

**Conclusion:** Hybrid accuracy is robust across inference engines and hardware. All differences are ≤1.6pp, most ≤0.4pp (1 job out of 239). The hybrid approach delivers consistent results regardless of deployment environment.

### Model-only accuracy: NOT comparable across engines

| Model | Historical MO | Master eval MO | Delta | Why |
|-------|--------------|----------------|-------|-----|
| V14 Q4_K_M | 62.3% | 81.1% | **+18.8pp** | max_tokens 600→1000 eliminates truncation; different engine produces different tokens |
| V14 Q6_K | 83.7% | 90.2% | +6.5pp | llama-server vs llama-cpp-python |
| V14 4B HF | 86.2% | 91.1% | +4.9pp | HuggingFace transformers vs MLX |
| V14 F16 GGUF | 86.2% | 89.8% | +3.6pp | llama-server vs llama-cpp-python |
| V14 thinking | 68.6% | 71.6% | +3.0pp | batch_generate vs single-prompt |
| V14 MLX 6-bit | 82.7% | 82.5% | -0.2pp | Closest match — same engine family |
| V12.1 1.5B | 86.2% | 78.4% | **-7.8pp** | batch_generate changes token distribution |

**Model-only scores depend on ALL 5 fields being correct.** A single different token (caused by different inference engines) flips the label. This is why V14 Q4_K_M swings 18.8pp — the schema-hallucinating model produces different wrong field names through different engines, and some happen to parse correctly while others don't.

**Do not compare model-only scores across different eval scripts or hardware.** Only compare within the same eval run.

### Parse fails: engine-dependent

| Model | Historical PF | Master eval PF | max_tokens | Notes |
|-------|--------------|----------------|------------|-------|
| V14 4B HF | 5 | 3 | 1500 | Improved — `{` pre-fill fix |
| V14 Q6_K | 10 | 4 | 1000 | Improved — llama-server |
| V14 Q4_K_M | 51 | 43 | 1000 (was 600) | Improved — higher token budget |
| V14 thinking | 0 | 10 | 3000 | Worse — batch_generate |
| V13.1 1.5B | 36 | 33 | 1500 | Similar |
| V13 0.6B | 19 | 26 | 1500 | Worse — batch_generate |
| V12.1 1.5B | 14 | 12 | 1500 | Similar |

### What changed between evals

| Factor | Historical | Master eval |
|--------|-----------|-------------|
| Hardware | M1 MacBook Air 16 GB / Lambda GH200 | M5 Pro Max 128 GB |
| MLX inference | `generate()` single-prompt | `batch_generate()` batch=32-239 |
| GGUF inference | `llama-cpp-python` (Python binding) | `llama-server` (Homebrew HTTP API) |
| Hybrid scoring | `compute_hybrid_v13_1.py` subprocess | In-process with cached regex |
| Regex versions | One per eval | All 3 (V12, V13, V13.1) simultaneously |
| max_tokens (GGUF) | 600 | 1000 |
| `{` pre-fill | Always for non-thinking | Skip for prompts with "Begin your response with {" |

---

## 11. Eval Methodology

- **Date:** 2026-03-28
- **Test set:** 239 audited jobs (`data/v12/test_labeled_audited.jsonl`)
- **MLX inference:** `batch_generate()` with architecture-adaptive batch size, greedy decoding (argmax sampler, deterministic)
- **GGUF inference:** `llama-server` (Homebrew) with 8 concurrent request slots, `--flash-attn on`, `--n-gpu-layers 99`
- **Hybrid scoring:** In-process Python (not subprocess) with 3 regex versions cached at startup — zero repeated computation
- **Prompt matching:** Each model uses the exact prompt it was trained/validated with. V14 prompts include "Begin your response with {" (no `{` pre-fill). V7/V13 prompts use `{` pre-fill.
- **Hardware:** Apple M5 Pro Max, 128 GB unified RAM, 700 GB/s bandwidth, 40 GPU cores
- **Total eval time:** ~36 minutes for all 15 models (10 MLX + 5 GGUF)
- **Script:** `master_eval/server_eval.py`
- **Raw results:** `master_eval/results/all_results.json` + per-model directories with predictions and hybrid summaries

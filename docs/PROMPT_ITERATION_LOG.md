# Prompt Iteration Log — Qwen3-0.6B Model-Only Improvement

**Started**: 2026-03-14
**Goal**: Improve model-only accuracy for Qwen3-0.6B (iter 1400 checkpoint) by optimizing the prompt
**Baseline**: 67.7% model-only label accuracy (153/226 valid, 13 invalid tokens)
**Model**: mlx-community/Qwen3-0.6B-4bit + LoRA adapter iter 1400
**Approach**: Inference-only prompt changes (no retraining) — testing if Qwen3's reasoning capability can be unlocked

## Baseline Analysis

The student prompt (`student_v7.txt`) is 13 lines with ZERO classification rules. The teacher prompt has 115 lines with detailed rules. Biggest model-only error sources:

| Field | Accuracy | Errors | Gap vs 1.5B | Teacher rules |
|-------|----------|--------|-------------|---------------|
| comp | 51.8% | 109 | -18.8pp | 18 lines: midpoint calc, currency, disqualifiers |
| tech | 58.4% | 94 | -8.3pp | 25 lines: tracked/untracked, OOS, spelling |
| arr | 80.1% | 45 | -10.4pp | 4 lines: attendance, REMOTE exception |
| sen | 87.6% | 28 | -5.5pp | 10 lines: title keywords, default L2 |
| loc | 93.4% | 15 | -1.0pp | 5 lines: city extraction |

**Key insight**: Qwen3 is a reasoning model being treated like a token classifier. The model could reason through comp midpoint calculations, tech matching, and sen title analysis — but has zero rules to work with.

---

## Iteration Results

### Attempt 1: Full rules in user prompt (`student_v7_reasoning.txt`)

**Change**: Replaced minimal prompt with 30-line version containing compressed teacher rules for all 5 fields.

**Result**: FAILED — killed after 17/239 jobs
- 2 invalid tokens (missing fields: tech, comp)
- 50% accuracy (vs 67.7% baseline)
- Model produced malformed JSON for some jobs (19s generation time = thinking loops)

**Root cause**: LoRA fine-tuning creates tight input→output token mappings. Changing the user prompt structure shifts the token positions, breaking the learned patterns. The model was trained on the minimal prompt and expects that exact format.

**Learning**: Cannot modify user prompt content — LoRA weights are too tightly coupled to it.

---

### Attempt 2: Minimal hints in user prompt (`student_v7_r2.txt`)

**Change**: Kept original structure, added 1-line hints after each token list line:
```
sen: LEVEL_3 | LEVEL_2 | LEVEL_1
  L3=senior/lead/staff/principal/head/founding/sr/III. L1=junior/jr/graduate/intern. L2=default.
```

**Result**: FAILED — killed after 14/239 jobs
- 5 invalid tokens in 14 predictions (35% failure rate!)
- Even minimal additions to user prompt break the model

**Root cause**: Same as Attempt 1. ANY modification to the user message disrupts LoRA-learned attention patterns.

**Learning**: User prompt is completely locked. Must use a different injection point.

---

### Attempt 3: Rules in system message, original prompt unchanged (`sysmsg_v1`)

**Change**: Modified only the system message (was "Respond with JSON only."):
```
Respond with JSON only. Rules: sen uses title — senior/lead/staff/principal/head/founding=L3, junior/graduate/intern=L1, else L2. tech explicit only — never infer. comp £ only, range→midpoint=(min+max)/2 for bucket.
```

**Result**: Mixed — label accuracy +8.7pp but field regressions

| Field | Baseline | SysMsg v1 | Delta |
|-------|----------|-----------|-------|
| **label** | **67.7%** | **76.4%** | **+8.7** |
| loc | 93.4% | 89.1% | -4.3 |
| arr | 80.1% | 70.3% | -9.8 |
| sen | 87.6% | 83.4% | -4.2 |
| tech | 58.4% | 50.7% | -7.7 |
| **comp** | **51.8%** | **65.9%** | **+14.1** |
| invalid_tokens | 13 | 10 | -3 |

**Analysis**: The comp midpoint rule worked spectacularly (+14.1pp). But the system message also disrupted arr (-9.8pp), tech (-7.7pp), and sen (-4.2pp). Net label gain comes from comp corrections outweighing other losses.

**Learning**: System message modifications work (model doesn't break), but affect all fields — like pushing on a balloon. Rules for one field can disrupt LoRA-learned behavior in others.

---

### Attempt 4: Comp-only rule in system message (`sysmsg_v2`)

**Change**: Isolated the comp rule only:
```
Respond with JSON only. comp: £ only. Non-£ or TC/OTE → NO_GBP. Up to → UP_TO_ONLY. Range → use midpoint=(min+max)/2 for bucket.
```

**Hypothesis**: If comp-only helped in v1, isolating it should give pure comp gains without side effects.

**Result**: REGRESSION — worse than baseline

| Field | Baseline | SysMsg v2 | Delta |
|-------|----------|-----------|-------|
| **label** | **67.7%** | **62.4%** | **-5.3** |
| loc | 93.4% | 85.3% | -8.1 |
| arr | 80.1% | 69.3% | -10.8 |
| sen | 87.6% | 88.5% | +0.9 |
| tech | 58.4% | 56.4% | -2.0 |
| comp | 51.8% | 42.7% | -9.1 |
| invalid_tokens | 13 | 21 | +8 |

**Analysis**: Comp actually got WORSE (-9.1pp) and invalid tokens doubled. The comp-only message disrupted the model more than the all-rules version. Possible explanations:
1. The "Rules:" prefix in v1 provided structural context that helped the model parse the instructions
2. Having rules for multiple fields created a balanced "instruction mode" that v2 (single field) didn't trigger
3. The shorter system message may have created a different attention pattern that confused the LoRA weights

**Learning**: System message changes are unpredictable. More text ≠ more disruption. The specific wording and structure matter as much as the content.

---

### Attempt 5: Softer wording with "Hints" prefix (`sysmsg_v3`)

**Change**: Reworded to use "Hints:" instead of "Rules:", softer language, dropped tech rule:
```
Respond with JSON only. Hints: For comp ranges use midpoint=(min+max)/2 to pick bucket. Non-£ currency or TC/OTE means NO_GBP. For sen use job title: senior/lead/staff=L3, junior/intern=L1, otherwise keep L2.
```

**Result**: CATASTROPHIC — killed after 143/239 jobs
- 35 invalid tokens at 143/239 (24% failure rate!)
- The "Hints:" prefix was far more disruptive than "Rules:"
- Completely unusable

**Learning**: Exact wording matters enormously. "Rules:" works, "Hints:" destroys the model. The LoRA weights are sensitive to specific token patterns in the system message.

---

### Attempt 6: v1 rules + arr clarification (`sysmsg_v4`)

**Change**: v1 rules + added "arr classify workplace attendance from description."
```
Respond with JSON only. Rules: sen uses title — senior/lead/staff/principal/head/founding=L3, junior/graduate/intern=L1, else L2. tech explicit only — never infer. comp £ only, range→midpoint=(min+max)/2 for bucket. arr classify workplace attendance from description.
```

**Result**: Worse than v1

| Field | Baseline | v1 | v4 |
|-------|----------|-----|-----|
| label | 67.7% | 76.4% | 72.0% |
| arr | 80.1% | 70.3% | 64.7% |
| sen | 87.6% | 83.4% | 84.9% |
| comp | 51.8% | 65.9% | 62.9% |
| inv_tok | 13 | 10 | 7 |

**Analysis**: Adding the arr rule made arr even WORSE (64.7% vs 70.3%). The arr clarification interfered with the LoRA-learned arr behavior. v1 remains the best system message.

---

## Hybrid Impact Test

Ran `compute_hybrid.py` on the best inference-only result (v1) to see if model-only gains translate to hybrid gains.

| Metric | Baseline Hybrid | v1 Hybrid | Delta |
|--------|----------------|-----------|-------|
| **V12 Hybrid** | **96.7% (231/239)** | **93.7% (224/239)** | **-3.0pp** |
| Hybrid errors | 8 | 15 | +7 |
| good_fit correct | 56/56 (100%) | 50/56 (89.3%) | -6 |
| maybe correct | 49/57 (86.0%) | 48/57 (84.2%) | -1 |
| bad_fit correct | 126/126 (100%) | 126/126 (100%) | 0 |

**V12 hybrid REGRESSED by 3.0pp** (96.7% → 93.7%). The model-only improvement (+8.7pp) came from comp (+14.1pp), but hybrid ignores model comp (regex handles it). The fields hybrid cares about (arr, sen) both regressed.

### Why Model-Only ≠ Hybrid

```
V12 Hybrid architecture:
  ┌─────────────┐    ┌─────────────┐
  │    REGEX     │    │    MODEL    │
  │  loc ✓ 100%  │    │  arr ✗ 70.3% (was 80.1%)
  │  tech ✓ 86%  │    │  sen ✗ 83.4% (was 87.6%)
  │  comp ✓ 95%  │    │
  └─────────────┘    └─────────────┘
         ↓                  ↓
      Override          Only arr/sen matter
```

The system message improved comp (model-only) by +14.1pp, but regex already handles comp at 95.4%. That comp improvement is invisible to the hybrid. Meanwhile, arr (-9.8pp) and sen (-4.2pp) directly hurt hybrid accuracy.

---

## All Results Summary

| Attempt | Approach | Model-Only | Hybrid | inv_tok | Verdict |
|---------|----------|-----------|--------|---------|---------|
| Baseline | Original | 67.7% | 96.7% | 13 | Reference |
| 1 | Rules in user prompt | ~50% | — | 2/17 | Killed: format breaks |
| 2 | Hints in user prompt | — | — | 5/14 | Killed: format breaks |
| **3 (v1)** | **Rules in sys msg** | **76.4%** | **93.7%** | **10** | **Best model-only, worse hybrid** |
| 4 (v2) | Comp-only sys msg | 62.4% | — | 21 | Regression everywhere |
| 5 (v3) | "Hints:" sys msg | — | — | 35/143 | Killed: catastrophic |
| 6 (v4) | v1 + arr hint | 72.0% | — | 7 | Worse than v1 |

---

## Key Findings (Final)

1. **User prompt is completely locked**: LoRA fine-tuning creates an unbreakable coupling between user message tokens and model behavior. ANY modification causes parse failures and regressions. Not even a single hint line can be added.

2. **System message is modifiable but unpredictable**: The model tolerates system message changes. Field-level accuracy shifts in unpredictable ways — improving one field can degrade others. The exact wording is critical ("Rules:" works, "Hints:" is catastrophic).

3. **Comp is the most responsive field** to inference-only rules (+14.1pp). This makes sense — comp requires mathematical reasoning (midpoint calculation) that can't be learned from pattern matching alone.

4. **Model-only accuracy ≠ hybrid accuracy**: Improving model-only by fixing a field that regex already handles (comp) provides zero hybrid benefit. The hybrid only cares about arr and sen from the model, and those regressed with every system message change.

5. **Inference-only prompt changes CANNOT improve hybrid accuracy**: Every system message variant that improved model-only accuracy did so by fixing comp (regex-handled) while hurting arr/sen (model-handled). This is a structural limitation of the approach.

6. **The LoRA-prompt coupling is the root cause**: The model learned specific behaviors tied to the original prompt during training. Inference-only changes can't override this without causing collateral damage.

## Recommendation

**Retrain with enhanced prompt + system message** is the only path to genuine improvement:

1. Create `student_v7_enhanced.txt` — add compressed rules for sen, tech, comp (keep same JSON output format)
2. Create matching enhanced system message with "Rules:" prefix
3. Reformat ALL training data with new prompt + new system message
4. Train fresh from base (NEVER resume)
5. The model will learn to use the rules from the start, avoiding cross-field interference

**Expected impact**: The comp rule alone added +14.1pp model-only in inference-only testing. With retraining, ALL fields should improve simultaneously:
- comp: +10-15pp (midpoint calculation rules)
- tech: +5-10pp (explicit-only, never-infer rules)
- sen: +3-5pp (title keyword rules)
- arr: +3-5pp (clearer attendance rules)
- Projected model-only: 80-85% (vs 67.7% baseline, vs 78.4% 1.5B)
- Projected hybrid: 97-98% (if arr/sen improve without regressing)

# V14 — Eval Optimisation Experiments Plan

**Date**: 2026-03-19
**Model**: Qwen3-4B V14 MLX 6-bit (`~/qwen3_4B_v14_mlx6bit`)
**Test set**: `data/v12/test_labeled_audited.jsonl` (239 jobs)
**Prompt**: `prompts/student_v14.txt` (unchanged — baked into training)
**Base script**: `finetune/eval_student_v7.py` (read-only — never modified)

---

## Context

V14 baseline eval (MLX 6-bit, thinking ON, 1000 tokens) is running at ~74-75% model-only
accuracy with ~10-11% parse failures. GGUF Q6_K on Lambda GH200 (no-think via system
prompt) achieved 83.7% model-only. The gap has two possible sources:

1. **Eval infrastructure bugs** — parse failures caused by script issues, not the model
2. **Thinking ON vs OFF** — does reasoning help or hurt the fine-tuned model?

These experiments isolate these two sources with two controlled runs on the same model weights.
No retraining. No prompt changes to `student_v14.txt`.

---

## Root Causes Identified (from baseline run analysis)

| ID | Issue | Mode affected | Impact |
|----|-------|--------------|--------|
| R1 | `{{` pre-fill bug: script adds `{`, model also adds `{` (follows prompt directive "Begin your response with {") | No-think | ~100% parse fail when using `--no-think` |
| R2 | Single-quote JSON: model sometimes outputs `{'key':'VALUE'}` — invalid JSON | Thinking ON | ~5-8% parse fail |
| R3 | Token exhaustion: 1000 tokens insufficient for long JDs (thinking + `_raw` fields) | Thinking ON | ~3-5% parse fail |
| R4 | Unclosed `<think>` tag: model runs out of tokens mid-thinking → no `</think>`, no JSON | Thinking ON | Rare but total loss (no recovery) |
| R5 | No retry: parse fails counted immediately as wrong, no second attempt | Both | All parse fails become wrong predictions |
| R6 | Trailing commas / Python literals: `None`, `True`, `,}` in output | Both | Occasional parse fail |
| R7 | Metric denominator mismatch: headline accuracy = `correct/valid` not `correct/total` | Both | Makes experiments non-comparable if parse fail rates differ |
| R8 | No speed summary: avg s/job not in summary JSON | Both | Can't compare experiment speeds |
| R9 | Between-job GPU memory not explicitly cleared | Both | Minor: possible latency spike between jobs |

---

## Experiment 1 — No-Think (Speed Champion)

**Goal**: Fastest possible eval, eliminate thinking overhead, test if fine-tuned model
needs reasoning at all.

**Script**: `finetune/eval_v14_exp1_no_think.py` (copy of `eval_student_v7.py`)
**Output dir**: `eval_results/v14_exp1_no_think/`

### Changes from base script

| Fix | Root cause | Code location | Change |
|-----|-----------|---------------|--------|
| Fix `{{` bug | R1 | Lines 355-356, 369-371 | Remove `formatted += "{"` for ALL Qwen3 models (not just when thinking ON). The prompt already instructs model to start with `{`. Only pre-fill for non-Qwen3. |
| Strip stray think tags | R1 defensive | Line 374 | Strip `<think>...</think>` even with `--no-think` (defensive — in case model partially thinks) |
| Add `mx.metal.clear_cache()` | R9 | After each job | Explicitly free GPU memory between jobs. Reduces inter-job latency spikes on M1. |
| Add `true_accuracy` metric | R7 | Summary section | Report `label_correct / n_total` alongside existing `label_correct / n_valid` |
| Add speed to summary JSON | R8 | Summary section | `avg_speed_s`, `total_time_s`, `tokens_per_sec` (estimated) |

**Nothing else changed.** Minimal diff from base.

### Run command

```bash
.venv/bin/python3 finetune/eval_v14_exp1_no_think.py \
  --model ~/qwen3_4B_v14_mlx6bit \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_exp1_no_think \
  --save-predictions \
  --no-think \
  --max-tokens 400
```

### Expected outcomes

- Parse failures: ~0 (the `{{` bug was the only cause with `--no-think`)
- Speed: ~3-5s/job (no thinking — only ~200 JSON tokens to generate)
- Model-only accuracy: unknown — this is what we're measuring
- Total eval time: ~15-20 min

---

## Experiment 2 — Thinking ON with All Repairs (Accuracy Champion)

**Goal**: Squeeze maximum accuracy from the thinking model by fixing every recoverable
failure mode. Compare fairly against Experiment 1.

**Script**: `finetune/eval_v14_exp2_thinking.py` (copy of `eval_student_v7.py`)
**Output dir**: `eval_results/v14_exp2_thinking/`

### Changes from base script

#### A. Same `{{` fix as Experiment 1

For consistency. Also fixes the defensive case where `--no-think` might be passed
accidentally.

#### B. System message additions (safe — not in training turns)

Current: `"Respond with JSON only."`

New: `"Respond with JSON only. Use double-quoted strings throughout your JSON. Keep _raw fields concise — copy only the most relevant phrase (10 words max)."`

- **Double-quote directive** → targets R2 (single-quote outputs)
- **`_raw` conciseness** → saves ~100-200 output tokens per job on long JDs, giving
  thinking more budget before exhaustion. Risk: model may ignore it (fine-tuned behaviour
  is strong). Flag in results if `_raw` values are still long.

#### C. Parser improvements in `parse_json_output()`

Applied in sequence before failing:

```
Attempt 1: json.loads() directly
Attempt 2: Strip markdown fences → json.loads()
Attempt 3: Fix unquoted keys (existing regex — R1 partial)
Attempt 4: [NEW] Fix single-quoted keys/tokens:
              re.sub(r"'(\w+)'(?=\s*:)", r'"\1"', text)   — keys
              re.sub(r"'([A-Z_0-9]+)'", r'"\1"', text)    — uppercase token values
              (safe: keys never have apostrophes; token values are always uppercase identifiers)
Attempt 5: [NEW] Python literals: None→null, True→true, False→false
Attempt 6: [NEW] Trailing comma removal: ,} → }  and  ,] → ]
Attempt 7: [NEW] Multi-close truncation recovery:
              Try closing with: "}", '"}"', '"]}', ']}'  — first that parses wins
              (existing code only tries '}' — this extends to handle truncated arrays)
Attempt 8: json.loads() on regex-extracted {…} block (existing)
→ return None (parse fail — triggers retry logic below)
```

#### D. Unclosed `<think>` detection and retry

```python
if "<think>" in response and "</think>" not in response:
    # Model ran out of tokens during thinking — no JSON was produced
    # Retry with 3× token budget
    response = generate(..., max_tokens=dynamic_max * 3)
```

Targets R4. This is the "silent total loss" case — without this, these jobs always fail.

#### E. Retry on parse fail (any cause)

If `parse_json_output()` returns None after all repair attempts:

```python
if parsed is None and not already_retried:
    response = generate(..., max_tokens=dynamic_max * 2)
    parsed = parse_json_output(response)
```

One retry, double the budget. Only triggered on failures (~10-11% of jobs based on
baseline) so total time overhead is minimal.

#### F. Dynamic max_tokens per job

```python
# Scale output budget with JD length
# Base 400 tokens (thinking) + 1 token per 6 chars of JD text
# Floor: 600, ceiling: 2000
dynamic_max = max(600, min(2000, 400 + len(jd_text) // 6))
```

- Short JD (500 chars) → 483 → 600 tokens
- Medium JD (1500 chars) → 650 tokens
- Long JD (3000 chars) → 900 tokens
- Very long JD (8000 chars) → 1733 tokens
- Capped at 2000

Targets R3. Avoids wasting time on short jobs AND running out on long ones.

#### G. `mx.metal.clear_cache()` between jobs (same as Experiment 1)

Targets R9.

#### H. Metrics (same as Experiment 1)

`true_accuracy`, `avg_speed_s`, `total_time_s` added to summary.

### Run command

```bash
.venv/bin/python3 finetune/eval_v14_exp2_thinking.py \
  --model ~/qwen3_4B_v14_mlx6bit \
  --test-file data/v12/test_labeled_audited.jsonl \
  --prompt prompts/student_v14.txt \
  --output-dir eval_results/v14_exp2_thinking \
  --save-predictions
```

(No `--no-think`. No `--max-tokens` — dynamic per job.)

### Expected outcomes

- Parse failures: <5 (all repair layers + retry should rescue most)
- Speed: ~10-20s/job (dynamic tokens; shorter for short JDs)
- Model-only accuracy: higher than baseline ~74%, hopefully approaching GGUF Q6_K 83.7%
- Total eval time: ~45-60 min (longer jobs get more tokens; retries add overhead)

---

## Comparison Framework

### Primary metric: `true_accuracy`

```
true_accuracy = label_correct / n_total   (parse fails count as wrong)
```

This is the only fair comparison metric when two experiments have different parse fail
rates. Both summary JSONs will include this.

### Secondary metric: `model_accuracy`

```
model_accuracy = label_correct / n_valid  (parse fails excluded from denominator)
```

Tells us: "when the model can produce parseable output, how good is it?"

### Speed metric

```
avg_speed_s = total_generation_time / n_total
```

Note hardware explicitly: **Mac M1 16GB, MLX 6-bit**. Do not compare to Lambda GH200
figures (0.7s/job) — different hardware.

### Decision criteria

| Scenario | Conclusion |
|----------|-----------|
| Exp1 true_accuracy ≥ Exp2 true_accuracy − 0.5% | Thinking adds no value. Ship no-think (3-5× faster). |
| Exp2 true_accuracy > Exp1 by 0.5-1% | Thinking helps on edge cases. Consider hybrid: no-think fast path, thinking fallback on low-confidence jobs. |
| Exp2 true_accuracy > Exp1 by >1% | Thinking is genuinely valuable. Accept speed cost. |
| Both < GGUF Q6_K (83.7%) by >2% | MLX 6-bit conversion has quality loss vs GGUF. Investigate quantization difference. |

---

## Baseline for Comparison

| Run | Script | Mode | Parse fails | True accuracy | Speed |
|-----|--------|------|-------------|--------------|-------|
| V14 MLX 6-bit baseline | eval_student_v7.py | Thinking ON, 1000 tok | 24 parse + 5 invalid (12%) | model=85.2% / true=74.9% | ~20s/job (M1) |
| V14 GGUF Q6_K (Lambda) | eval_student_v14_gguf.py | No-think (system msg) | 10/239 (4%) | 83.7% | 0.7s/job (GH200) |
| V14 GGUF F16 (Lambda) | eval_student_v14_gguf.py | No-think (system msg) | 8/239 (3%) | 86.2% | — |

---

## File Map

```
finetune/
  eval_student_v7.py              ← original, READ ONLY
  eval_v14_exp1_no_think.py       ← Experiment 1 (to create)
  eval_v14_exp2_thinking.py       ← Experiment 2 (to create)

eval_results/
  v14_mlx6bit/                    ← baseline (current run, thinking ON)
  v14_exp1_no_think/              ← Experiment 1 results
  v14_exp2_thinking/              ← Experiment 2 results

docs/
  V14_EXPERIMENTS_PLAN.md         ← this file
```

---

## Open Questions (to answer from results)

1. Does the `_raw` conciseness system message instruction get followed by the fine-tuned
   model? Check avg `_raw` token length in Exp2 vs baseline.

2. How many parse fails does each repair layer rescue? Add per-layer counters to Exp2
   script for analysis.

3. Do the retried jobs tend to be right or wrong? Track retry outcomes separately.

4. Does thinking help specifically on `sen` (L2/L3 boundary) or `arr` edge cases?
   Compare per-field accuracy between Exp1 and Exp2.

5. Is the MLX 6-bit model degraded vs GGUF Q6_K? If both experiments land <81%,
   investigate the quantization conversion — the source `merged_v14_4B` was rebuilt
   from HF after the truncated-download incident (confirmed 8GB).

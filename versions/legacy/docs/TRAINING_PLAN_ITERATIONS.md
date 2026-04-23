# Training Plan Evolution: V1 → V5

## V1 → V2: From Generic Advice to Targeted Plan

- **Fresh LoRA, not continuing old adapter.** Old adapter learned terse reasoning like `"tech stack (15)"`. Can't mix old and new reasoning formats in a 0.5B model.
- **Re-label all 450 jobs with rich reasoning.** Student needs to see WHY a score is 15, not just that it is 15.
- **Custom scripts for reasoning + validation.** Deterministic retrofit of reasoning onto existing scores. Zero API cost.
- **800-job target with specific distribution quotas.** Problem was distribution, not volume. Needed loc=-50, tech edge cases, comp traps.
- **150-job balanced eval set.** Original eval had 7 good_fit samples — useless. New: 50/50/50.
- **Rejected:** "Use GPT-4o over GPT-4o-mini" (minimal quality gap for mechanical rules), "95-99% is realistic" (honest ceiling is 90-96%).

## V2 → V3: GPT-4o-mini as Single Source of Truth

- **Killed the custom scripts.** GPT-4o-mini generates scores AND reasoning in one pass. Handles context better than regex (e.g., "Node" as network node vs Node.js).
- **Two separate prompts.** Lean teacher prompt (fast/cheap for GPT-4o-mini). Compact student prompt (rules only, no examples — student learns from training data).
- **GPT-4o-mini for everything:** labeling, reasoning, eval, data augmentation. Single source of truth eliminates inconsistency.
- **Retrofit existing 450 via API instead of scripts.** Cost: ~$0.10. Simpler, more accurate, handles edge cases.
- **5 steps instead of 7.** Speed matters — combined labeling and set-building steps.

## V3 → V4: Self-Critique + User Insights

### Issues found in self-critique:
- **Reasoning format asked model to reproduce exact JD quotes** → 0.5B models hallucinate when copying. Relaxed to cite keyword without exact quote.
- **Changed data AND hyperparameters simultaneously** → Can't diagnose failures. Kept V1's learning rate (5e-5). One variable at a time.
- **Eval set judged against GPT-4o-mini which has ~3-5% error rate** → Manual verification of all 50 `maybe` labels + spot-check 40 others.
- **GPT-4o-mini reasoning drifts at scale** → Added deterministic normaliser for consistent formatting.
- **Blind truncation could remove salary info** → Smart truncation: scan for £/$ symbols, protect surrounding context.

### User insights that changed the design:
- **Detailed student prompt teaches wrong patterns.** Capitalised keywords like "Node.js" teach case-sensitive matching. Real JDs use every variation. → **Student prompt reduced to ~30 tokens. No rules, no keywords.**
- **Student can't memorise all UK cities.** → Required 30+ non-UK countries and 15+ UK cities in training. Dublin vs Belfast as explicit test.
- **Eval should ignore reasoning.** Reasoning is a training scaffold, not an output. → Score only on field values and label.
- **Synthetic data too clean.** → Post-processing: randomise capitalisation, remove formatting, inject contradictions.
- **Daily rates (£500/day) unhandled.** → Added to teacher prompt and training data requirements.

## V4 → V5: Semantic Tokens (The Fundamental Redesign)

User proposed replacing numeric scores with semantic labels. This was the single most impactful change.

- **`loc: 25` → `loc: "LONDON_OR_REMOTE"`.** Semantically grounded in JD text. "25" is abstract; "LONDON_OR_REMOTE" is a classification the model understands.
- **Removed `score` and `label` from model output.** Code computes both deterministically. Eliminates arithmetic errors and label misclassification entirely.
- **21-token output space** (loc: 4, role: 3, tech: 8, comp: 6). Small models excel at small classification problems.
- **`UP_TO_ONLY` gets its own token.** "Up to £90k" and "no salary" look completely different to the model. Separate token = stronger learning signal for the trap.
- **Teacher requires mentioning ignored tech and currencies.** `react ignored, python ignored → NODE_JS_TS`. Model learns what NOT to score.
- **Fuzzy matching in code layer.** Edit distance ≤ 2 catches minor misspellings. Insurance that costs nothing.

**The conversion in one line:**
```
BEFORE: Model does comprehension → extraction → scoring → arithmetic → labeling (5 tasks)
AFTER:  Model does comprehension → extraction → classification (3 tasks). Code does the rest.
```

## Rejected Suggestions (All Versions)

| Suggestion | Why Rejected |
|------------|-------------|
| Increase training to 1,200-1,500 | Problem was distribution, not volume. 720 curated > 1,200 padded. |
| Add `flags` field for confidence | 0.5B can't do metacognition. Better: code checks if score is near label boundary, routes to GPT-4o-mini tiebreaker. |
| Hard Example Mining before first training | HEM requires knowing what the model gets wrong. Train first, then patch. |
| "Senior in title but 2 years in JD" as edge case | Contradicts rules. Role = title keywords. JD experience is irrelevant. Would teach wrong rule. |
| Pretty-printed JSON in student prompt | Training data uses single-line JSON. Format mismatch confuses small models. |
| Two-pass inference | Doubles latency. Reasoning scaffold already primes internal state in single pass. |
| Detailed rules in student prompt | Capitalised keywords teach case-sensitive matching. Model should learn from diverse data. |
| Keep old 400 examples with terse reasoning | Mixed reasoning formats confuse 0.5B. Re-label everything. |

## Version Summary

| Version | Key Change | Student Prompt |
|---------|-----------|---------------|
| V1 | Generic advice collected | N/A |
| V2 | Targeted gaps, custom scripts | ~500 tokens (full rules + examples) |
| V3 | GPT-4o-mini does everything | ~250 tokens (rules, no examples) |
| V4 | Minimal prompt, learn from data | ~30 tokens (bare structure) |
| V5 | Semantic tokens, classification not regression | ~30 tokens (bare structure) |

**The journey:** "Teach the model rules and hope it does math" → "Show it 800 classified examples and let code do the math."

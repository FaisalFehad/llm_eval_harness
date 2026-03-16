# Architecture Details

> For the high-level overview, see the main [README.md](../README.md).

---

## Semantic Token Vocabulary (V7)

The model outputs **10 JSON fields** (5 raw + 5 token) representing job fit across 5 dimensions. The `_raw` fields hold verbatim JD text as chain-of-thought scaffolding; the token fields hold the classification.

### Location (`loc`)

| Token | Score | Rule |
|-------|-------|------|
| IN_LONDON | +25 | Job location contains "London" |
| REMOTE | +25 | Location says "Remote" (UK context) |
| UK_OTHER | +10 | UK city outside London (40+ cities in list) |
| OUTSIDE_UK | -50 | Non-UK location |
| UNK | 0 | Empty, unclear, or "Various locations" |

### Work Arrangement (`arr`)

| Token | Score | Rule |
|-------|-------|------|
| REMOTE | 0 | Fully remote, no office requirement |
| HYBRID | 0 | Mix of remote and office |
| IN_OFFICE | 0 | Full-time office |
| UNK | 0 | Not stated |

All arr tokens score 0 — this field is informational only. Errors in arr prediction have zero impact on the final label.

### Seniority (`sen`)

| Token | Score | Rule |
|-------|-------|------|
| LEVEL_3 | +25 | Senior, Lead, Staff, Principal, Director, Head, VP, CTO |
| LEVEL_2 | +15 | Mid-level: "Software Engineer", "Developer", "Data Engineer" |
| LEVEL_1 | 0 | Junior, Intern, Graduate, Associate, Entry-level |

Title-first detection: check the job title for keywords. If unclear, check the first hiring sentence of the JD.

### Tech Stack (`tech`) — Array field

| Token | Score | Rule |
|-------|-------|------|
| NODE | +10 | Node.js, Express, NestJS (backend JS) |
| REACT | +5 | React, React Native (frontend framework) |
| JS_TS | +5 | JavaScript, TypeScript (when explicitly named) |
| AI_ML | +10 | AI/ML, machine learning, deep learning (in requirements, not company description) |
| OOS | 0 | Out of scope — no tracked tech found |

**Tech is an array**: `["NODE", "JS_TS"]` not a combo string. Scores are additive: NODE + JS_TS = 15.

**OOS scope gate**: If `"OOS"` is in the tech array (or tech is empty), `role_score` is forced to 0. This prevents non-engineering jobs from scoring seniority points.

**Rules**: Never mix OOS with real tokens. Never leave tech empty (use `["OOS"]`). Each tech evaluated independently.

### Compensation (`comp`)

| Token | Score | Rule |
|-------|-------|------|
| NO_GBP | 0 | No GBP salary found (USD, EUR, daily rates, or no salary) |
| UP_TO_ONLY | 0 | "Up to £X" with no lower bound |
| BELOW_45K | -30 | GBP midpoint < £45,000 |
| RANGE_45_54K | 0 | GBP midpoint £45,000–£54,999 |
| RANGE_55_74K | +5 | GBP midpoint £55,000–£74,999 |
| RANGE_75_99K | +15 | GBP midpoint £75,000–£99,999 |
| ABOVE_100K | +25 | GBP midpoint ≥ £100,000 |

**Midpoint calculation**: For "£70k–£85k", midpoint = £77,500 → RANGE_75_99K.

**Disqualifiers**: OTE (on-target earnings), daily rates (£X/day), total compensation ("TC") → NO_GBP.

---

## Score Computation

Score and label are computed deterministically in code, never by the model:

```python
loc_score = LOCATION_MAP[pred["loc"]]
is_oos = "OOS" in pred["tech"] or len(pred["tech"]) == 0
role_score = 0 if is_oos else SENIORITY_MAP[pred["sen"]]
tech_score = 0 if is_oos else sum(TECH_INDIVIDUAL_MAP[t] for t in pred["tech"])
comp_score = COMP_MAP[pred["comp"]]

score = max(0, min(100, loc_score + role_score + tech_score + comp_score))
label = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
```

---

## Hybrid Pipeline

```
Job → Preprocess JD → Model inference (all 5 fields)
                      → Regex override (loc/tech/comp)
                      → Code-computed score → Label
```

### Why hybrid?

| Field | Model accuracy | Regex accuracy | Winner |
|-------|---------------|---------------|--------|
| loc | ~93% | **100%** | Regex |
| tech | ~61% | **88.3%** | Regex |
| comp | ~67% | **95.8%** | Regex |
| sen | **86.6%** | ~29% | Model |
| arr | **72.8%** | N/A | Model |

The model excels at comprehension tasks (reading a JD to judge seniority). Regex excels at mechanical tasks (matching city names, parsing salary strings, detecting tech keywords). The hybrid pipeline gives each system its strength.

### Regex classifiers

**Location** (`classify_loc`): 40+ UK cities, Northern Ireland vs Republic of Ireland, "Anywhere" detection, non-UK indicators, "Little London" exclusion.

**Tech** (`classify_tech`): Pattern matching for NODE/REACT/JS_TS/AI_ML with boilerplate filters. "AI-powered company" in the company description doesn't trigger AI_ML — only AI/ML in requirements sections.

**Comp** (`classify_comp`): Candidate-based salary parser. Finds all £ amounts in JD text, applies disqualifiers (OTE, daily rate, TC), calculates midpoint, maps to token.

### Parse failure fallback

When the model produces invalid output (19 of 239 test jobs in V13), regex provides fallback predictions for all 5 fields. This is why parse failures don't affect hybrid accuracy for loc/tech/comp (regex handles those anyway). For sen, the regex fallback is weak (~29%), so parse failures can cause sen errors.

---

## Data Pipeline

```
Scrape/Generate → Pre-label audit → Label (gpt-4.1-mini, temp=0)
    → Post-label audit → Split → Format for MLX → Train → Eval (hybrid)
```

### Pre-label audit
- Contamination check against locked eval set (job_id + family ID + JD SHA-256 fingerprint)
- Minimum JD length validation
- Structure validation

### Labeling
- Teacher: gpt-4.1-mini, temperature=0 (deterministic)
- Concurrency=10, preflight API check, non-retryable fast-fail (401/403/404)
- Auto-ID for empty job_ids, tech dedup + OOS cleanup

### Post-label audit
- Token validation against V7 vocabulary
- Fuzzy matching (edit distance ≤2) for near-miss tokens
- Suspicious pattern detection
- Quarantine for bad data

### MLX formatting
- Convert JSONL to chat format (system + user + assistant messages)
- Smart truncation: protect salary windows (100-word context around £/$), preserve first 300 and last 200 words, truncate only from middle
- Stratified train/valid split by label
- Prompt baked into training data (student learns the prompt as part of input distribution)

### Training
- MLX LoRA: rank 16, alpha 32, dropout 0.05
- `mask_prompt=true`: loss only on assistant response
- `grad_checkpoint=true`: required for M1 16GB
- Checkpoint saved every N iters, swept with hybrid evaluator

---

## Key Files

| File | Purpose |
|------|---------|
| `finetune/semantic_tokens_v7.py` | Token vocabulary, score maps, validation, fuzzy matching |
| `finetune/deterministic_baseline_v13.py` | Production regex classifier (loc/tech/comp) |
| `finetune/compute_hybrid_v13.py` | Hybrid evaluator (combines regex + model predictions) |
| `finetune/eval_student_v7.py` | Model inference + token validation |
| `src/cli/label-jobs-v7.ts` | Teacher labeling with gpt-4.1-mini |
| `src/cli/audit-training-data-v7.ts` | Pre/post-label data quality checks |
| `src/cli/format-for-mlx-v7.ts` | Convert labeled JSONL to MLX chat format |
| `prompts/student_v13.txt` | Production student prompt (35 lines) |
| `prompts/teacher_v7.txt` | Teacher labeling prompt (115 lines, 3 examples) |
| `data/v12/test_labeled_audited.jsonl` | Locked test set (239 jobs, chmod 444) |

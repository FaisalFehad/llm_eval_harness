# AI Eval Harness

A pipeline for building a fast, locally-running job-fit scoring model that matches the accuracy of large commercial LLMs — trained on your own taste.

The technique is **LLM knowledge distillation**: use a large model (Claude) to generate high-quality labeled data at scale, then fine-tune a small local model to mimic it. End result: a model that runs in Ollama, costs nothing per inference, and scores jobs the way _you_ would.

---

## The Full Pipeline

```text
Phase 1 → Ground Truth        80 hand-labeled jobs  →  golden_jobs.jsonl
Phase 2 → Naive Benchmark     ~20 Ollama models × 3 prompts  →  drop the worst
Phase 3 → Prompt Optimization Top models × better prompts × temperatures  →  keep 2-3
Phase 4 → Synthetic Data      1000+ jobs → Claude labels them  →  training_data.jsonl
Phase 5 → Fine-tuning         LoRA fine-tune best small model on Phase 4 data
Phase 6 → Final Showdown      Custom model vs Claude vs best Ollama baseline
```

---

## Phase 1 — Ground Truth Dataset

The golden dataset is the benchmark everything else is measured against. Each record needs three fields that _you_ fill in:

```jsonc
{
  "job_id": "4371838647",
  "title": "Senior Software Engineer",
  "company": "FORT",
  "jd_text": "...",

  // Your verdicts — these are the correct answers:
  "label": "good_fit", // good_fit | maybe | bad_fit
  "score": 75, // 0–100
  "reasoning": "Strong TS/Node stack at a product-focused Series A.",
}
```

**Target size:** 80–100 records. Already sampled into `data/golden_jobs.jsonl`.

**Engineering notes (Phase 1):**

- Final golden set: 103 records. Label distribution is intentionally skewed — 52 bad_fit, 40 maybe, 11 good_fit — because that reflects the real job market signal. Skewed distribution matters for sampling strategy in later phases.
- Faker.js was used to generate synthetic JD variety for edge cases in the test suite (e.g., non-UK locations, missing salary, management roles). Ensures the eval isn't overfit to a narrow slice of the dataset.
- Key lesson: hand-labeling even 103 records takes longer than expected. Having an explicit scoring rubric (the 4-category matrix) was essential — without it, scores drift as fatigue sets in.

**Commands:**

```bash
# Sample from a jobs export
npm run golden:sample -- --input data/jobs_export.jsonl --count 100 --seed 42

# Validate labels
npm run golden:validate:strict
```

**Candidate profile encoded in scoring prompts:**

- 8+ years backend/platform engineering
- Strong in TypeScript, Node.js, Python, SQL, cloud (AWS/GCP)
- Wants product-focused IC roles (senior/staff level)
- Avoids agency delivery, consulting, junior-heavy, and non-software roles

---

## Scoring Criteria

Each golden record is scored against a 100-point matrix. Models are evaluated on how closely their output matches these scores.

### 1. Role & Seniority (max 25 pts)

| Points | Criteria                                                                        |
| ------ | ------------------------------------------------------------------------------- |
| +25    | Senior Software Engineer, Tech Lead, Lead Engineer, Staff or Principal Engineer |
| +15    | Full Stack Engineer, Mid-Level Software Engineer, Software Engineer II          |
| 0      | Junior roles, management roles (EM), or unrelated positions                     |

### 2. Tech Stack & Domain (max 25 pts)

| Points | Criteria                                       |
| ------ | ---------------------------------------------- |
| +10    | Node.js explicitly required                    |
| +5     | JavaScript or TypeScript explicitly required   |
| +10    | AI, ML, or LLM experience explicitly mentioned |

Points stack (e.g. Node.js + TypeScript + AI = 25). Capped at 25 for this category.

### 3. Location & Work Arrangement (max 25 pts)

| Points  | Criteria                                              |
| ------- | ----------------------------------------------------- |
| +25     | Fully Remote (UK-based or global) OR Hybrid in London |
| +10     | Hybrid or On-site in UK outside London                |
| 0       | Location/arrangement unknown                          |
| **-50** | **Hybrid or On-site located outside the UK**          |

### 4. Compensation (max 25 pts)

| Points  | Criteria                     |
| ------- | ---------------------------- |
| +25     | Base salary £100k or above   |
| +15     | Base salary £75k–£99k        |
| +5      | Base salary £55k–£74k        |
| 0       | Salary unknown or not listed |
| **-30** | **Base salary below £45k**   |

### Label Mapping

| Score  | Label      | Fit       |
| ------ | ---------- | --------- |
| 85–100 | `good_fit` | Excellent |
| 70–84  | `good_fit` | Good      |
| 50–69  | `maybe`    | Maybe     |
| 0–49   | `bad_fit`  | Bad       |

Final score = sum of all four categories, capped at 100, floored at 0.

---

## Phase 2 — Tournament Model Selection

### Why a tournament instead of brute force?

The original plan was to run all ~20 Ollama models against all 103 golden jobs — roughly **2,100 inferences**. On hardware-constrained setups like my little Macbook Air with 16Gb RAM, can be a locker:

- Ollama can only load **one model at a time**. Every model swap means unloading/reloading into RAM.
- Many of these models are obviously bad candidates (350M param models, old LLaMA 2, coding-only models). Running them against 103 jobs each wastes hours on models that will clearly fail. So why wait for a model to finish 103 jobs when it could be dropped after just 10? - this could save MANY hours of runtime by eliminating bad models early.
- A full brute-force run would take **8–12 hours** with no early feedback.
- I could have selected only a few models but I honestly have no idea how each model could perform. I could have also kept the largest models only, as they have higher likelihood of better ratings, but given my hardware constraints, I am happy to trade a little bit of accuracy for speed and not crashing my machine.

The solution is a **3-round tournament** that drops bad models early, so only worthy candidates get the full evaluation.

### Round 1 — Smoke Test (all models, 10 jobs)

Run every model against just **10 balanced jobs** (~3 good_fit, 4 maybe, 3 bad_fit). The smoke test uses **balanced** rather than proportional sampling because the golden set is skewed (52 bad_fit, 40 maybe, only 11 good_fit). With proportional sampling, 10 jobs would include just 1 good_fit — too few to meaningfully test that class. Balanced sampling gives every label fair representation so even with 10 jobs we can spot models that fail on specific categories.

**Drop a model immediately if:**

- Parse failure > 30% (can't produce valid JSON)
- Label accuracy < 40% (worse than random guessing on 3 classes)
- Average response time > 120 seconds per job
- Model fails to load or OOM

Expected: eliminates ~8–10 models in **~30–60 minutes**.

```bash
npm run tournament:smoke
# Results → results/tournament/smoke/
```

### Smoke Test Results — Round 1

**6 of 16 models advanced. 10 eliminated.**

#### Cloud models (1 passed, 2 eliminated)

| Model | Acc | Parse Fail | MAE | Bias | Avg/job | Result |
| ----------------------------- | --- | ---------- | ---- | ----- | ------- | ------ |
| minimax-m2.5:cloud            | 70% | 0%         | 20.5 | +13.5 | 20.1s   | PASS   |
| mistral-large-3:675b-cloud    | 0%  | 100%       | —    | —     | 44.1s   | FAIL   |
| ministral-3:14b-cloud         | 0%  | 100%       | —    | —     | 1.0s    | FAIL   |

minimax-m2.5 was the top performer across all 16 models — highest accuracy, lowest MAE, zero parse failures. The two cloud failures exited immediately with 100% parse failures after 2 jobs each. These are almost certainly misconfigured Ollama cloud endpoints (wrong API format or missing auth), not a reflection of the underlying models' intelligence.

#### Larger local models — 7B–8B (5 passed, 3 eliminated)

| Model | Acc | Parse Fail | MAE | Bias | Avg/job | Result |
| ----------------------------- | --- | ---------- | ---- | ----- | ------- | ------ |
| qwen3:8b                      | 50% | 20%        | 23.1 | +23.1 | 81.2s   | PASS   |
| wizardlm2:latest (~7B)        | 40% | 0%         | 36.7 | +23.7 | 23.5s   | PASS   |
| mistral-openorca:7b           | 40% | 0%         | 31.5 | +29.5 | 14.1s   | PASS   |
| dolphin-llama3:latest (~8B)   | 40% | 0%         | 37.0 | +37.0 | 13.5s   | PASS   |
| llama2:7b                     | 40% | 0%         | 38.5 | +37.5 | 15.6s   | PASS   |
| mistral-nemo:latest (~12B)    | 20% | 0%         | 37.1 | +21.9 | 26.5s   | FAIL   |
| qwen2.5:7b-instruct           | 20% | 0%         | 35.2 | +11.4 | 14.4s   | FAIL   |
| qwen2.5:14b-instruct-q4_K_M  | 10% | 0%         | 31.2 | +16.4 | 29.4s   | FAIL   |

qwen3:8b leads local models at 50% accuracy — the only local model that showed any `bad_fit` reasoning — but carries two caveats: 81.2s/job is the slowest survivor, and 20% parse failures suggest it still occasionally chokes on JSON output. If parse failures persist into qualifying, it won't survive Round 2.

The three 40% local passes (mistral-openorca, dolphin-llama3, llama2:7b) passed by the skin of their teeth. All three predicted `good_fit` for every single job — they got 40% by correctly labelling the actual good_fits, while completely ignoring `maybe` (0/3) and `bad_fit` (0/3). These are "yes-man" models, not scorers.

qwen2.5:14b-instruct-q4_K_M is the biggest surprise: the largest local model tested scored the worst label accuracy of any non-cloud model (10%). Aggressive q4_K_M quantization likely degraded instruction-following to the point where the model size advantage was erased.

#### Small local models — sub-5B (0 passed, 5 eliminated)

| Model | Acc | Parse Fail | MAE | Bias | Avg/job | Result |
| -------------------- | --- | ---------- | ---- | ----- | ------- | ------ |
| qwen3:1.7b           | 30% | 20%        | 31.5 | +31.5 | 29.3s   | FAIL   |
| granite4:350m        | 30% | 0%         | 31.5 | -8.5  | 1.9s    | FAIL   |
| ministral-3:3b       | 20% | 0%         | 36.5 | -8.5  | 11.0s   | FAIL   |
| granite4:latest      | 20% | 0%         | 36.0 | +26.0 | 8.8s    | FAIL   |
| llama3.2:1b          | 20% | 0%         | 45.0 | +45.0 | 3.9s    | FAIL   |

No sub-5B model survived. The accuracy ceiling for this task seems to be around 30% at this scale — comparable to random guessing.

Notable: **granite4:350m** was the fastest model tested at 1.9s/job, produced valid JSON on every run, and — uniquely — was one of only two models with a negative score bias (-8.5), meaning it was slightly conservative rather than optimistic. Impressive behaviour for a 350M model, but 30% accuracy isn't enough to pass the threshold.

**llama3.2:1b** didn't fail on JSON parsing — it produced syntactically valid JSON every time — but the label fields contained literal strings like `"good_fit|maybe|bad_fit"` (copied from the prompt format instructions) and `"Role & Seniority"` (copied from the rubric). It was parroting the prompt rather than reasoning. A fundamentally different failure mode: parse success, semantic failure.

**qwen3 family scaling**: qwen3:1.7b scored 30% vs qwen3:8b's 50%, confirming a clear size effect within the same model family. The 1.7B variant was also slower (29.3s vs 81.2s) — likely because it struggles more per token, not because it does more work.

#### Key cross-cutting findings

- **Positive score bias is near-universal**: 14 of 16 models systematically over-scored. Bias ranged from +11 to +45. Only granite4:350m and ministral-3:3b had negative bias. Models that over-score are dangerous for this use case — the whole point is to filter out bad jobs, so a model that rates everything as "good" is useless regardless of parse-level metrics.
- **`maybe` is a dead class**: Only minimax-m2.5:cloud correctly classified any `maybe` jobs (3/3). Every other survivor mapped `maybe` entirely to `good_fit`. This will matter in qualifying — if a model can't distinguish nuance, it won't be useful.
- **Model size does not predict quality**: The largest local model (14B) had the worst accuracy. The 350M model had better JSON reliability than several 7B models. The best result came from a cloud model of unknown effective size.
- **Cloud ≠ automatically better**: cloud APIs can fail completely (100% parse error) while a ~8B local model scores 50%. Connectivity and output format compatibility matter.

### Round 2 — Qualifying (survivors, 30 jobs)

Surviving models face **30 stratified jobs** — enough data for statistically meaningful accuracy readings.

**Drop if:**

- Label accuracy < 55%
- Score MAE > 30
- Parse failure > 10%
- Average time > 60 seconds per job

Expected: cuts to ~5–7 models in **~1–2 hours**.

```bash
npm run tournament:qualifying
# Results → results/tournament/qualifying/
```

### Round 3 — Full Eval (finalists, all 103 jobs)

Only the ~5–7 survivors run against the full golden set. Uses the existing `eval-tagged` infrastructure for full metrics and result snapshots.

```bash
npm run tournament:full
# Results → results/runs/ (via eval-tagged)
```

### Auto mode — run all 3 rounds unattended

```bash
npm run tournament:auto
# Chains smoke → qualifying → full, passing survivors forward automatically
```

**Time comparison:**

| Approach               | Inferences | Est. Time  |
| ---------------------- | ---------- | ---------- |
| Brute force (21 × 103) | ~2,100     | 8–12 hours |
| Tournament (likely)    | ~960       | 3–5 hours  |

Thresholds for each round are tunable in `configs/tournament_thresholds.json`.

**Engineering notes (Phase 2):**

- **Prompt-to-golden mismatch (critical bug)**: All 4 scoring prompts (`promptfooconfig.yaml` + `prompts/scorer_v*.txt`) had wrong label thresholds and were missing key rubric dimensions (e.g., wrong bad_fit boundary at 39 instead of 49, missing AI/ML +10 bonus, wrong compensation scale). Would have made all eval metrics meaningless — models scored against a different rubric than the golden labels. Fixed by rewriting all prompts to match the golden rubric exactly before running any evals.
- **Ollama OOM crashes mid-run ("fetch failed")**: Models like qwen3:8b default to a 40,960-token context window, which pre-allocates ~11 GB KV cache on a 16 GB machine. After 2–3 successful jobs, Ollama would crash, causing all subsequent calls to fail with `fetch failed`. Fix: set `num_ctx: 4096` — sufficient for job scoring since each call is stateless and prompts are ~1500–2000 tokens. **Before vs after (qwen3:8b):** crashed at job 4 (early exit, 0 jobs completed) → completed all 10 jobs at 81.2 s/job avg. KV cache dropped from ~11 GB to ~3–4 GB. **qwen2.5:7b-instruct** benefited most from the smaller allocation: all 10 jobs completed at 14.4 s/job avg with 0 parse failures.
- **Fail-fast logic gap**: Initial logic only caught 100% failure rates (`parseFails === testsCompleted`). Partial failures (2 successes then crash) weren't caught, wasting time. Added `consecutiveErrors >= 3` bail-out to detect mid-run Ollama crashes.
- **Model memory bleeding between tests**: Previous models stayed resident in Ollama RAM while the next model loaded, causing OOM. Fixed with `keep_alive: 0` (Ollama API param to immediately evict the model from memory after use).
- **qwen2.5:14b is a hard hardware limit**: 9 GB model on a 16 GB system fails consistently. Graceful failure is handled; no code change can fix a RAM ceiling. Noted as a threshold for what hardware class this system requires.
- **Brute-force was infeasible**: 21 models × 103 jobs = ~2,100 inferences at 30–120 s/job = potentially 12+ hours, with the worst models consuming the same time as the best. Tournament cut this to ~960 inferences with early results after 30–60 minutes.

**Metrics captured per model:**

| Metric             | Why it matters                            |
| ------------------ | ----------------------------------------- |
| Label accuracy     | % of labels matching your ground truth    |
| Score MAE          | Mean absolute error vs your golden scores |
| Parse failure rate | % of outputs that aren't valid JSON       |
| Avg time per job   | End-to-end latency (seconds)              |

---

## Phase 3 — Prompt Optimization

For the surviving models, iterate on:

1. **Few-shot examples** — include 2–3 examples from the golden set directly in the prompt. Biggest single accuracy boost for small models.
2. **Temperature** — try 0.0, 0.3, 0.7. Lower = more consistent JSON, less hallucination.
3. **Rubric clarity** — explicit scoring rubric with point breakdowns (see `prompts/scorer_v2.txt`).
4. **Chain-of-thought** — internal reasoning before final answer (see `prompts/scorer_v3_cot.txt`).

Each iteration is tagged and tracked:

```bash
npm run eval:tagged -- --tag v2_stack_weighting
npm run eval:tagged -- --tag v3_few_shot
npm run eval:tagged -- --tag v4_temp0_cot

npm run iterations:summary
```

Each tagged run saves: eval results, prompt snapshots, run metadata, and a log row in `results/iteration_log.csv`.

**Keep the top 2–3 models after this phase.**

---

## Phase 4 — Synthetic Training Data (Claude as Labeler)

Pull 500–2000 job descriptions from your jobs DB and send them through Claude to generate high-quality labeled data at scale.

**Cost:** ~$1.50–$5 for 1000 jobs using Claude Haiku (~500 token avg JD).

**Output schema** — same as golden_jobs, minus manual fields:

```jsonc
{ "job_id": "...", "jd_text": "...", "label": "...", "score": ..., "reasoning": "..." }
```

```bash
npm run generate:synthetic -- --input data/jobs_export.jsonl --count 1000
# Resumable — skips already-processed records
# Output → data/synthetic_labels.jsonl
```

**Combine** golden (hand-labeled, ~80) + synthetic (Claude-labeled, ~1000) → `data/training_data.jsonl`.

The golden set is weighted more heavily during fine-tuning since it reflects your actual judgement.

---

## Phase 5 — Fine-tuning

Take the best small model from Phase 3 (likely Phi-3 Mini, Llama 3.2 3B, or Gemma 2 2B) and fine-tune it using LoRA on the training data.

**Tool: [Unsloth](https://github.com/unslothai/unsloth)**

- 2× faster than vanilla HuggingFace training
- 60% less VRAM — works on a laptop or free Colab GPU
- Exports directly to GGUF for Ollama

**Training data format (instruction tuning):**

```json
{
  "instruction": "Score this job for the candidate profile below.\n\n[profile]",
  "input": "[jd_text]",
  "output": "{\"label\":\"good_fit\",\"score\":78,\"reasoning\":\"...\"}"
}
```

```bash
# Format training data
python training/format_training_data.py

# Fine-tune (run in Colab or locally with GPU)
python training/finetune.py

# Load into Ollama
ollama create job-scorer -f training/Modelfile
```

**Experiment tracking:** [Weights & Biases](https://wandb.ai) — free, auto-generates loss curves and model comparison charts.

---

## Phase 6 — Final Showdown

Benchmark three things head-to-head on the golden 80:

| Model                   | Accuracy | Score MAE | Speed  | RAM    | Cost/job |
| ----------------------- | -------- | --------- | ------ | ------ | -------- |
| Claude Haiku (API)      | baseline | baseline  | fast   | 0      | ~$0.003  |
| Best Ollama baseline    | ?        | ?         | medium | 4–8 GB | free     |
| Fine-tuned custom model | ?        | ?         | fast   | 2–4 GB | free     |

The story: **a 3B model fine-tuned on Claude's output nearly matches Claude's accuracy, runs locally for free.**

```bash
npm run benchmark:final
# Results → results/final_showdown/
```

---

## Project Structure

```text
ai_eval_harness/
├── data/
│   ├── golden_jobs.jsonl          # Hand-labeled ground truth (80–100 records)
│   ├── synthetic_labels.jsonl     # Claude-generated training labels
│   ├── training_data.jsonl        # Combined (golden + synthetic)
│   └── promptfoo_tests.yaml       # Generated — Promptfoo test cases
├── prompts/
│   ├── scorer_v1.txt              # Baseline prompt
│   ├── scorer_v2.txt              # Rubric-based prompt
│   └── scorer_v3_cot.txt          # Chain-of-thought prompt
├── configs/
│   └── tournament_thresholds.json # Tunable pass/fail thresholds per round
├── results/
│   ├── tournament/                # Phase 2 tournament rounds (smoke/qualifying/full)
│   ├── runs/                      # Tagged iteration runs
│   ├── iteration_log.csv          # Run history
│   └── final_showdown/            # Phase 6 comparison
├── src/cli/
│   ├── sample-golden-from-export.ts
│   ├── validate-golden.ts
│   ├── build-promptfoo-tests.ts
│   ├── sample-test-subset.ts    # Stratified subset sampler for tournament rounds
│   ├── tournament.ts            # Tournament runner (smoke → qualifying → full)
│   ├── eval-tagged.ts
│   └── summarize-iterations.ts
├── training/
│   ├── format_training_data.py    # Prepares Unsloth input
│   ├── finetune.py                # Unsloth LoRA training script
│   └── Modelfile                  # Ollama model definition
├── docs/
│   └── week1_week2_checklist.md
└── promptfooconfig.yaml
```

---

## Requirements

- Node.js 20+
- Ollama with models pulled (for Phase 2/3)
- Python 3.10+ with Unsloth (for Phase 5)
- `ANTHROPIC_API_KEY` env var (for Phase 4 synthetic labeling)

```bash
npm install
```

---

## Commands Reference

| Command                               | Description                                              |
| ------------------------------------- | -------------------------------------------------------- |
| `npm run golden:sample`               | Sample records from a jobs export                        |
| `npm run golden:validate`             | Validate golden dataset schema                           |
| `npm run golden:validate:strict`      | Validate with strict size (80–100 rows)                  |
| `npm run promptfoo:tests`             | Build Promptfoo test cases from golden data              |
| `npm run week1:baseline`              | Run baseline eval (validates + builds tests + runs eval) |
| `npm run eval:tagged -- --tag <name>` | Run a tagged iteration                                   |
| `npm run iterations:summary`          | Show recent run history                                  |
| `npm run tournament:smoke`            | Round 1 — all models vs 10 jobs, drop failures           |
| `npm run tournament:qualifying`       | Round 2 — survivors vs 30 jobs, tighten thresholds       |
| `npm run tournament:full`             | Round 3 — finalists vs all 103 jobs                      |
| `npm run tournament:auto`             | Run all 3 rounds back-to-back                            |

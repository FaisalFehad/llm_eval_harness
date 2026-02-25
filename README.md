# AI Eval Harness

A pipeline for building a fast, locally-running job-fit scoring model that matches the accuracy of large commercial LLMs — trained on your own taste.

The technique is **LLM knowledge distillation**: use a large model (Claude) to generate high-quality labeled data at scale, then fine-tune a small local model to mimic it. End result: a model that runs in Ollama, costs nothing per inference, and scores jobs the way *you* would.

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

The golden dataset is the benchmark everything else is measured against. Each record needs three fields that *you* fill in:

```jsonc
{
  "job_id": "4371838647",
  "title": "Senior Software Engineer",
  "company": "FORT",
  "jd_text": "...",

  // Your verdicts — these are the correct answers:
  "label": "good_fit",   // good_fit | maybe | bad_fit
  "score": 75,           // 0–100
  "reasoning": "Strong TS/Node stack at a product-focused Series A."
}
```

**Target size:** 80–100 records. Already sampled into `data/golden_jobs.jsonl`.

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

## Phase 2 — Naive Benchmark (~20 Ollama Models)

Run all available local models against the golden 80 with no prompt tuning. Establishes a baseline and identifies which models are worth keeping.

**Metrics captured per model:**

| Metric | Why it matters |
| ------ | -------------- |
| Label accuracy | % of labels matching your ground truth |
| Score MAE | Mean absolute error vs your golden scores |
| F1 per class | Does it handle `bad_fit` as well as `good_fit`? |
| Parse failure rate | % of outputs that aren't valid JSON |
| Tokens/sec | Raw generation speed |
| Time per job | End-to-end latency |
| Peak RAM (GB) | Resource cost |

**Cut anything below ~60% label accuracy or above 10% parse failure rate.**

```bash
npm run benchmark:naive
# Results → results/benchmark_naive/
```

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

| Model | Accuracy | Score MAE | Speed | RAM | Cost/job |
| ----- | -------- | --------- | ----- | --- | -------- |
| Claude Haiku (API) | baseline | baseline | fast | 0 | ~$0.003 |
| Best Ollama baseline | ? | ? | medium | 4–8 GB | free |
| Fine-tuned custom model | ? | ? | fast | 2–4 GB | free |

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
├── results/
│   ├── benchmark_naive/           # Phase 2 results
│   ├── benchmark_optimized/       # Phase 3 results
│   ├── runs/                      # Tagged iteration runs
│   ├── iteration_log.csv          # Run history
│   └── final_showdown/            # Phase 6 comparison
├── src/cli/
│   ├── sample-golden-from-export.ts
│   ├── validate-golden.ts
│   ├── build-promptfoo-tests.ts
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

| Command | Description |
| ------- | ----------- |
| `npm run golden:sample` | Sample records from a jobs export |
| `npm run golden:validate` | Validate golden dataset schema |
| `npm run golden:validate:strict` | Validate with strict size (80–100 rows) |
| `npm run promptfoo:tests` | Build Promptfoo test cases from golden data |
| `npm run week1:baseline` | Run baseline eval (validates + builds tests + runs eval) |
| `npm run eval:tagged -- --tag <name>` | Run a tagged iteration |
| `npm run iterations:summary` | Show recent run history |

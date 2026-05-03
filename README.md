# AI Eval Harness

**Can a model 8,000× smaller than GPT-4 score jobs better than GPT-4 — running locally on a personal notebook?**

Yes. **99.6% accuracy.** A 3.1 GB model running on Apple Silicon at zero inference cost, outperforming the teacher it learned from.

This is my journey through LLM knowledge distillation — from hand-labeling 103 jobs to building a production-grade hybrid pipeline. Every technique learned by doing, every decision driven by data, every setback turned into a better solution.

---

## The headline

```
GPT-4.1-mini (teacher)    → labels 860 jobs       → ~91% accuracy, £0.08/batch
Qwen3-4B V15 (student)    → trained via LoRA      → 99.6% hybrid / 91.4% model-only
                            3.1 GB (Q6_K)          0 cost per inference
                            MLX on M5 Pro Max      ~7s/job
```

The student surpassed the teacher. Not by being smarter — by combining a distilled model with surgical regex rules, each handling what it does best.

| Model                  | Hybrid    | Model-only | Size   | Speed    | Best for             |
| ---------------------- | --------- | ---------- | ------ | -------- | -------------------- |
| **V15 4B (iter 700)**  | **99.6%** | **91.4%**  | 3.1 GB | ~7s/job  | **Current champion** |
| V14 4B (step 800)      | 98.7%     | 86.2%      | 3.1 GB | ~32s/job | V15 baseline         |
| V13 0.6B (iter 1500)   | 97.9%     | 67.3%      | 335 MB | ~3s/job  | Low latency          |
| V12.1 1.5B (iter 2000) | 98.3%     | 86.2%      | 839 MB | ~7s/job  | All-rounder          |

_All scores on 239-job audited test set. Hybrid = model + V13.1 regex overrides._

---

## What it does

Scores LinkedIn job postings across 5 dimensions, producing a 0–100 score and label (`good_fit` / `maybe` / `bad_fit`).

```
Job → Model inference (all 5 fields) → Regex override (loc/tech/comp) → Score → Label
```

**Why hybrid?** The model gets tech right 72% of the time; regex gets it 90%. Regex can't judge seniority from context; the model gets it 92%. Each system handles what it's best at.

| Field              | Model   | Regex    | Hybrid uses | Accuracy  |
| ------------------ | ------- | -------- | ----------- | --------- |
| Location           | 97%     | **100%** | Regex       | **100%**  |
| Tech stack         | 72%     | **90%**  | Regex       | **90%**   |
| Compensation       | 79%     | **96%**  | Regex       | **96%**   |
| Seniority          | **92%** | 80%      | Model       | **92%**   |
| Work arrangement   | **91%** | —        | Model       | **90%**   |
| **Combined label** | 86%     | 95%      | **Both**    | **99.6%** |

The model outputs semantic tokens (not numbers). Code converts tokens → scores → labels:

```python
{"loc": "IN_LONDON", "sen": "LEVEL_3", "tech": ["NODE", "JS_TS"], "comp": "RANGE_75_99K"}
score = loc_score + role_score + tech_score + comp_score  # 0-100
label = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
```

**Key design decision:** Separate classification from scoring. The model's job is "this job requires Node.js." How that translates to points is a business decision that lives in code, not a neural network. If NODE goes from 10→15 points, change one line of Python — no retraining needed.

### Why hybrid approach

**Model alone: 86%** — Struggles with mechanical tasks. Tech parsing at 72% because it reads "AI-powered company" in the description and confuses company identity with tech requirements. Comp at 79% because salary strings have edge cases (OTE, daily rates, "up to £X").

**Regex alone: 95%** — Perfect at mechanical tasks. Location is 100% (city list matching). Tech is 90% (pattern matching with boilerplate filters). Comp is 96% (midpoint calculation with disqualifiers). But regex can't judge seniority from context — it gets 80% by title matching alone, missing nuanced JDs where "Software Engineer" actually means senior-level responsibilities.

**Combined: 99.6%** — Each system handles its strength. Regex overrides model on loc/tech/comp because it's more accurate. Model handles sen/arr because it can read context. The result exceeds either system's ceiling.

**Parse failures don't break hybrid** — When the model produces invalid JSON (17 of 239 jobs in V15), regex provides fallback predictions for loc/tech/comp. Hybrid accuracy stays high even with 7% parse failures.

---

## The `_raw` fields: a pedagogical invention

Each field has a `_raw` companion that forces the model to extract evidence before classifying:

```json
{"sen_raw": "Staff Backend Engineer", "sen": "LEVEL_3",
 "tech_raw": "Node.js, TypeScript, React, PostgreSQL", "tech": ["NODE", "JS_TS", "REACT"]}
```

The interleaving is deliberate: `sen_raw` (verbatim text) immediately before `sen` (classification). This is **reasoning-by-construction** — the model copies the evidence, then classifies it.

**The trade-off:** `_raw` fields consume context space (avg 85 tokens/field, 340 tokens total) but provide three compounding benefits:

1. **5× more gradient signal** — With `mask_prompt=true`, loss computes only on the model's response (~430 chars). `_raw` fields provide 5× more training signal than token-only format (430 vs ~90 chars of bare tokens).

2. **+10pp accuracy boost** — Forcing evidence extraction before classification measurably improves both teacher and student accuracy. The model learns "which part of the JD matters" for each field.

3. **Debugging superpower** — When the student misclassifies, `_raw` shows what evidence it extracted. Did it find the wrong text, or find the right text and classify it wrong? Different bugs, different fixes.

**No fixed length cap:** Evidence length varies by field (tech: 15-80 chars, sen: 10-45 chars, comp: 20-120 chars). Solution: add a clear stop signal in the prompt ("End each _raw field with a newline"), then teach the model to respect it via training data. This forces the model to learn "enough evidence" rather than relying on arbitrary limits.

**Evaluation approach:** Student model generates `_raw` fields but isn't scored on them — they're for debugging only. This avoids penalizing the model for extraction quality while preserving the accuracy benefits for token predictions.

**Ablation proof:**

| Format | Accuracy | Delta | Parse failures |
|--------|----------|-------|----------------|
| Full `_raw` + tokens | **84.9%** | baseline | 15 |
| Tokens only, no `_raw` | 73.6% | **−10pp** | 23 (+53%) |

The model doesn't just classify better with `_raw` — it **learns** better. Each `_raw` field is a micro-lesson: "this part of the JD determines this classification."

---

## Token architecture vs regression

**V5 regression approach:** Model predicted numbers directly (`{"score": 75, "label": "good_fit"}`). Result: 39.8% accuracy. The model learned the rubric but wrote wrong numbers.

**V7 token approach:** Model predicts semantic categories, code computes scores:

| Field | Tokens | Score Range | Why tokens win |
|-------|--------|-------------|----------------|
| `loc` | 5 tokens | -50 to +25 | City matching is categorical, not numeric |
| `sen` | 3 tokens | 0 to +25 | Seniority levels are discrete buckets |
| `tech` | 5 tokens (array) | 0 to +30 | Tech stack is a set, not a number |
| `comp` | 7 tokens | -30 to +25 | Salary bands are business rules |
| `arr` | 4 tokens | 0 (info only) | Future-proofing + diagnostic value |

**Benefits:**

- **Classification > regression** — Predicting categories is easier for small models than precise numbers
- **Business logic in code** — If NODE goes from 10→15 points, change Python. No retraining.
- **Deterministic scoring** — Past predictions automatically recalculate if scoring rules change
- **Debuggable** — "LEVEL_3 vs LEVEL_2" is easier to analyze than "score 73 vs 68"

**Result:** 39.8% → 83.9% (+44pp) by switching from regression to tokens.

---

## Knowledge distillation: teacher vs student

**Teacher (GPT-4.1-mini):** 115 lines, 3 worked examples, 25 lines of tech rules, explicit AI_ML validation logic. Scores ~91% on its own labels — but disagrees with itself on 52.5% of AI_ML edge cases at temperature=0.

**Student (Qwen3-4B V15):** 34 lines, one line for tech: `"subset of [NODE, REACT, JS_TS, AI_ML], or exactly [\"OOS\"]."` Everything else learned from 842 training examples. Scores 99.6% hybrid / 91.4% model-only.

**The gap is the curriculum.** The 81-line difference between teacher and student prompts is exactly what the training data teaches. Each training example is a lesson: "given this JD excerpt, extract this evidence, produce these tokens."

**Distillation paradox:** The student surpasses the teacher not by being smarter, but by being **constrained**. GPT-4.1-mini has infinite flexibility — it can generate any text, including inconsistencies. The student is limited to 21 token categories + deterministic regex. That constraint makes it more consistent than its source.

**Iteration arc:**

| Version | Teacher | Student | Ratio | Student Accuracy |
|---------|---------|---------|-------|------------------|
| V5 | 59 lines | 11 lines | 5.4:1 | 83.9% |
| V6 | 185 lines | 12 lines | 15.4:1 | 75.3% (overfit) |
| V7 | 114 lines | 13 lines | 8.8:1 | 84.9% |
| V13 | 114 lines | **34 lines** | 3.4:1 | 97.9% |

The student prompt grew as the model shrank (0.6B needs more help at inference). The teacher prompt shrank as it got more precise. They met in the middle: teacher provides rules, student provides learned patterns.

---

## Technical highlights

### The most surprising discoveries

**1. The 5-minute fix that gave +14pp (Phase 6)**

Real-world eval: 81.9% accuracy. Failure pattern: 9 jobs where model recognized PlayStation/Flexera/Zoom as American companies and overrode location to OUTSIDE_UK, despite JD saying "London, England, United Kingdom."

**Fix:** One worked example added to prompt:
> *"PlayStation... headquartered in San Mateo... Location: London → IN_LONDON (ignore company HQ, trust location field)"*

**Result:** 81.9% → 95.8% in 5 minutes. Prompt before training, always.

**2. Prompt overfitting discovery (Phase 9)**

Three days, 9 prompt iterations. v9.4: **97% on tuning, 76% on held-out** — textbook prompt overfitting.

| Prompt | Tuning | Held-out | Lines | Delta |
|--------|--------|----------|-------|-------|
| v9.4 | **97.0%** | 75.8% | 150 | +21.2pp gap |
| v9.8 | 95.0% | **98.6%** | 138 | -3.6pp gap |

**Pattern:** Prompt length correlated inversely with held-out accuracy. Fine-tuned models develop strong associations with training prompt. **Every line has a cost.**

**3. SHA-256 fingerprint contamination detection (Phase 9)**

Discovered **67% of held-out test set contaminated** (22/33 jobs in training). Built 3-level pipeline: job_id → family ID → JD SHA-256 fingerprint.

**Result:** Fingerprint layer caught 92 overlaps that job_id/family ID missed — jobs scraped twice under different company names but identical JD text. Simple checks miss this; hashing doesn't.

**4. Speed breakthrough scrapped 1,235 lines (Phase 3)**

Ollama: 80 sec/job. llama.cpp: 14 sec/job (5.8× faster). At 14s/job, no need for clever tournament code — test everything directly.

**Result:** Scrapped 1,235 lines of tournament logic, built `eval-runner.ts` (~500 lines). Constraint drove architecture; removing constraint simplified it.

**5. `_raw` fields as pedagogical tool (Phase 11)**

Not chain-of-thought — **reasoning-by-construction**. Model generates `sen_raw` (verbatim) immediately before `sen` (classification). Evidence → decision, per field, strict order.

**Results:**
- +10pp accuracy improvement
- 5× more gradient signal (430 vs ~90 chars)
- Parse failures: 15 → 23 (+53%) without `_raw`
- Debugging: see exactly what evidence model extracted

**6. Student beats teacher through constraint (Phase 13)**

GPT-4.1-mini: 115 lines, infinite flexibility, 52.5% self-disagreement on AI_ML edge cases at temp=0.

Qwen3-4B V15 student: 34 lines, limited to 21 token categories + deterministic regex, **99.6% hybrid accuracy**.

**The win:** Constraint eliminates teacher's self-consistency problem. Student offloads mechanical tasks to regex (near-perfect), leaving only judgment calls to model.

**7. The impact of prompt engineering on inference speed**

Prompt engineering is a free latency optimization. V14 showed that the thinking-mode prompt averaged ~20s/job on the same M1 Mac because it generated compact JSON, while the no-thinking prompt with verbose evidence output took ~32–45s. V15 tightened the production prompt to eliminate output preambles and cut token waste.

**Output tokens cost wall-clock time — the prompt that generates fewer tokens is faster regardless of hardware.**

---

## The journey (condensed)

**Phase 1: Ground truth** — Hand-labeled 103 jobs. Lost data to a bad script. Lesson: always version data, always have a rubric.

**Phase 2: Model selection** — 20 models, 3 runtimes. 4B Qwen beat every 7B and 8B. Size ≠ quality. M1 16 GB constraint drove everything.

**Phase 3: Speed** — llama.cpp: 5.8× faster than Ollama, 0% parse failures. Changed everything. Scrapped 1,235 lines of tournament code.

**Phase 4: Prompt engineering** — More instructions hurt small models. Qwen3-4B dropped from 80%→60% with ~50% more prompt text. Simple prompts win.

**Phase 5: First fine-tune** — 39.8% baseline (golden labels were 20% wrong). LoRA: 39.8% → 90.9%. The model knew the rules; it just wrote wrong numbers.

**Phase 6: Reality check** — 81.9% on real LinkedIn jobs. Location fix: +14pp in 5 minutes. Prompt before training, always.

**Phase 7: OOD testing** — 88.9% on nurses/chefs. Added domain gate: 88.9% → 98.6%. Model wasn't memorizing.

**Phase 8: Teacher audit** — GPT-4.1-mini labels were 39% wrong. Oxford → London, fabricated salaries, math errors. Garbage in, garbage out.

**Phase 9: Teacher retrain** — 9 prompt iterations. Discovered prompt overfitting: 97% on tuning set, 76% on held-out. For small models, tell them WHERE to look, not just WHAT.

**Phase 10: OpenAI pivot** — Local teacher: 81 sec/job, 6-9 hour iteration. GPT-4.1-mini: 0.6 sec/job, £0.08/batch, 5 minutes. Speed unlocked iteration.

**Phase 11: Semantic tokens** — Stopped predicting numbers, started predicting categories. Classification > regression. `_raw` fields force evidence extraction before classification: +10pp improvement. Code as ground truth.

**Phase 12: Hybrid breakthrough** — Regex beats model on mechanical tasks (tech 90% vs 72%, comp 96% vs 79%). Model beats regex on comprehension (seniority 92% vs 80%). Combined: 97.5% → 97.9%.

**Phase 13: V13 production** — 0.6B Qwen3, 351 MB, 97.9% hybrid. Hit the wall: 5 remaining errors all seniority boundary cases, irreducible at 0.6B capacity.

**Phase 14: V14 cloud GPU** — Lambda GH200, 4B model, 41 min training. 98.7% hybrid at full precision. Converted to MLX 6-bit: 98.3% in 3.1 GB. New ceiling.

**Phase 15: V15 data engineering** — Targeted data fixes: "Anywhere" → REMOTE (8 errors), OOS downsampling (bias), NODE/AI_ML distribution gaps, comp boundaries. No architecture changes, pure data. Result: 99.6% hybrid, +7.7pp model-only.

Every dip in the accuracy chart is a moment I raised the bar — harder data, smaller model, stricter eval. The drops aren't regressions; they're ambition.

---

## What I learned

**Data quality beats quantity.** Every training failure traced to data composition — London bias, NODE-biased synthetic jobs, comp imbalance. More data often hurt (V9: +57% data, -16pp). Targeted data always helped (V13: +52 surgical contrastive examples, +0.4pp). ([V6-V11 analysis](docs/V6_V11_DEEP_ANALYSIS_2026-03-13.md))

**Validate before training, not after.** A 5-second pre-training audit catches problems that cost 8+ hours of GPU time to diagnose. Built a 3-stage pipeline: pre-label audit, labeling guards, post-label audit. ([V13 implementation](docs/V13_PLAN.md))

**Prompt before training, always.** Location fix: +14pp in 5 minutes. Domain gate: +10pp. Fine-tuning takes hours and can break things. ([Phase 6](#phase-6--reality-check))

**Hybrid > pure anything.** Model alone: 86%. Regex alone: 95%. Together: 99.6%. Neither system alone exceeds 95%; combined they reach 99.6%. ([Phase 12](#phase-12--the-hybrid-breakthrough))

**Separate what models learn from what code computes.** Semantic tokens + deterministic scoring = bugs fixable without retraining, fields addable without retraining, past predictions automatically recalculate. ([Phase 11](#phase-11--the-architectural-pivot))

**Small models need explicit signal, not bigger prompts.** `_raw` fields provide evidence-before-classification scaffolding — a 10pp improvement. But once LoRA-tuned, the model can't absorb new instructions — only new training data. ([Phase 11](#phase-11--the-architectural-pivot), [V13 vs V13.1](docs/V13_1_IMPLEMENTATION_PROGRESS.md))

**Architecture generation > parameter count.** 0.6B Qwen3 (newer architecture, `<think>` tokens) beat 0.5B Qwen2.5 by 5.8pp despite being only 21% larger. It also beat 1.5B Qwen2.5 by 0.4pp at 40% the size — parse reliability is a first-class accuracy variable. ([V13.1 analysis](docs/V13_1_IMPLEMENTATION_PROGRESS.md))

**Val loss ≠ downstream accuracy.** 0.6B best val loss at iter 1600 (0.165) but best hybrid at iter 1500 (97.9%). 1.5B best val loss at iter 1400 (0.142) but best hybrid at iter 1800 (97.5%). Val loss optimizes token prediction; hybrid accuracy optimizes label boundaries. Different objectives, different optima. ([V13 sweep](docs/V13_PLAN.md#val-loss--best-checkpoint))

**The student can surpass the teacher.** GPT-4.1-mini scores ~91% on its own labels; the hybrid student scores 99.6%. The teacher disagrees with itself on 52.5% of AI_ML edge cases at temperature=0 — the student, constrained to 21 categories + deterministic regex, is more consistent than its source. The win isn't comprehension — it's constrained design. ([Teacher audit](docs/DETAILED_JOURNEY.md#phase-8--knowledge-distillation))

**Hardware constraints are design constraints.** M1's 16 GB eliminated large models, drove the OpenAI pivot, forced one-model-at-a-time eval (two models = OOM). Thermal throttling on 30+ minute evals meant budgeting 15-20 min per 239-job eval. The result: a model that runs anywhere, a pipeline that respects resource limits. ([Phase 10](#phase-10--the-openai-pivot))

**Non-destructive versioning saves you.** Every version lives alongside its predecessors — never overwritten. V13 regex was created next to V12's, not replacing it. After losing V5's eval set to SIGPIPE, I added 4 layers of defense: script guards, `chmod 444`, documentation rules, git tracking. No single safeguard is enough. ([Phase 1](#phase-1--building-ground-truth))

**Trace errors to root causes, not symptoms.** A 50-char hard cap on `tech_raw` taught the model to generate broken JSON boundaries. A field name typo (`job.location` vs `job_location`) made 83% of locations UNK. A single duplicate job with a wrong label created 4× gradient in the wrong direction. These problems all looked like "model capacity" on the surface; the root cause was always data. ([V6-V11 analysis](docs/V6_V11_DEEP_ANALYSIS_2026-03-13.md))

---

## Quick start

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run V15 (best model)

```bash
# Start OMLX server (port 8000)
./eval.sh server

# Run eval with production prompt
./eval.sh run production

# View results
./eval.sh langfuse  # traces at localhost:3000
npx promptfoo view  # UI at localhost:15500
```

### Run other models

```bash
# V13 0.6B (335 MB, ~3s/job, 97.9% hybrid)
harness eval run --version v13 --output-dir eval_v13

# V14 4B MLX 6-bit (3.1 GB, ~32s/job, 98.3% hybrid)
harness eval run --version v14 --output-dir eval_v14

# V12.1 1.5B (839 MB, ~7s/job, 98.3% hybrid)
harness eval run --version v12_1 --output-dir eval_v12_1
```

### Label new data

```bash
npx tsx src/cli/label-jobs-v7.ts \
  --input data/raw.jsonl \
  --output data/labeled.jsonl \
  --model gpt-4.1-mini
```

---

## Production candidates

Three models achieve 97.9–99.6% hybrid accuracy on Mac, each with different trade-offs:

| Model      | Hybrid    | Model-only | Size   | Speed   | Parse | Best for              |
| ---------- | --------- | ---------- | ------ | ------- | ----- | --------------------- |
| **V15 4B** | **99.6%** | **91.4%**  | 3.1 GB | ~7s/job | 17    | **Accuracy champion** |
| V12.1 1.5B | 98.3%     | 86.2%      | 839 MB | ~7s/job | 14    | All-rounder           |
| V13 0.6B   | 97.9%     | 67.3%      | 335 MB | ~3s/job | 19    | Low latency           |

All benchmarked on the same 239-job test set, same V13.1 regex, same scorer.

**V15 4B is the champion** — +1.3pp hybrid vs V14, +7.7pp model-only. Trained locally on M5 Pro Max with targeted data fixes (no architecture changes). The 0.6B remains the best choice when size/latency matter more than absolute accuracy.

---

## Project structure

```
ai_eval_harness/
├── bin/harness                 # CLI entry point
├── finetune/                   # ML scripts
│   ├── cli.py                  # Harness CLI (Typer)
│   ├── registry.py             # Pipeline registry
│   ├── commands/               # Verb implementations
│   ├── adapters_v15_4B/        # V15 LoRA adapters
│   └── semantic_tokens_v7.py   # Token vocab & scoring
├── versions/                   # Per-version layout (reorg 2026-04-19)
│   └── v15/                    # V15 self-contained
│       ├── adapters/           # LoRA checkpoints
│       ├── configs/            # Promptfoo + training configs
│       ├── data/               # Train/valid splits
│       ├── docs/               # V15 plan & runbook
│       ├── eval_results/       # Sweep & production evals
│       ├── prompts/            # Student & teacher prompts
│       └── scripts/            # Frozen V15 pipeline
├── src/cli/                    # TypeScript CLI tools
│   ├── label-jobs-v7.ts        # Teacher labeling
│   ├── audit-training-data-v7.ts
│   └── format-for-mlx-v7.ts
├── data/v12/test_labeled_audited.jsonl  # 239 test jobs (locked)
├── eval.sh                     # Promptfoo wrapper
├── promptfooconfig.yaml        # Root config
└── docs/                       # Deep-dive documentation
    ├── DETAILED_JOURNEY.md     # Full narrative
    ├── V14_IMPLEMENTATION_PROGRESS.md
    ├── V13_1_IMPLEMENTATION_PROGRESS.md
    ├── V6_V11_DEEP_ANALYSIS.md
    └── ARCHITECTURE.md
```

---

## Deep-dive documentation

| Document                                                                       | What's inside                                      |
| ------------------------------------------------------------------------------ | -------------------------------------------------- |
| [docs/DETAILED_JOURNEY.md](docs/DETAILED_JOURNEY.md)                           | Complete narrative — every phase, failure, fix     |
| [versions/v15/docs/PLAN.md](versions/v15/docs/PLAN.md)                         | V15 implementation plan (data engineering)         |
| [versions/v15/docs/RUNBOOK.md](versions/v15/docs/RUNBOOK.md)                   | V15 step-by-step execution                         |
| [docs/V14_IMPLEMENTATION_PROGRESS.md](docs/V14_IMPLEMENTATION_PROGRESS.md)     | V14 full history — 4B training, quantization       |
| [docs/V13_1_IMPLEMENTATION_PROGRESS.md](docs/V13_1_IMPLEMENTATION_PROGRESS.md) | V13.1 — regex, 0.6B corrective, 1.5B training      |
| [docs/V6_V11_DEEP_ANALYSIS.md](docs/V6_V11_DEEP_ANALYSIS.md)                   | 8 experiments that led to hybrid approach          |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)                                   | Semantic tokens, scoring rules, pipeline mechanics |

---

## Technical stack

| Component          | Technology                                                      |
| ------------------ | --------------------------------------------------------------- |
| **Training**       | MLX (Mac) / PyTorch + Unsloth (Lambda GH200)                    |
| **Inference**      | MLX (Apple Silicon native) / llama.cpp (GGUF)                   |
| **Fine-tuning**    | LoRA (rank 16, alpha 32, dropout 0.05)                          |
| **Student models** | Qwen3-4B (V15), Qwen3-0.6B (V13), Qwen2.5-1.5B (V12.1)          |
| **Teacher**        | gpt-4.1-mini                                                    |
| **Pipeline**       | TypeScript CLI tools + Python eval scripts                      |
| **Hardware**       | Apple M5 Pro Max 128GB / M1 16GB / Lambda GH200                 |
| **Model storage**  | [FF-01/qwen3-4b-v15](https://huggingface.co/FF-01/qwen3-4b-v15) |

---

## Further reading

This project is a practical application of ideas from the following papers and resources.

### Knowledge distillation & small models

| Paper | Link |
|-------|------|
| **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **Neural-Symbolic Collaborative Distillation** — Liao et al., AAAI 2025 | [arXiv:2409.13203](https://arxiv.org/abs/2409.13203) |
| **On the Surprising Efficacy of Distillation** — Farhat & Chen, ICLR 2024 | [arXiv:2404.03263](https://arxiv.org/abs/2404.03263) |
| **Big Reasoning with Small Models** — Alkiek et al., 2025 | [arXiv:2510.13935](https://arxiv.org/abs/2510.13935) |

### Chain-of-thought & reasoning

| Paper | Link |
|-------|------|
| **Chain-of-Thought Prompting Elicits Reasoning in LLMs** — Wei et al., NeurIPS 2022 | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| **Towards Reasoning in Large Language Models: A Survey** — Huang & Chang, ACL 2023 | [arXiv:2212.10403](https://arxiv.org/abs/2212.10403) |

### Hybrid neural-symbolic systems

| Paper | Link |
|-------|------|
| **Neural-Symbolic Collaborative Distillation** — Liao et al., AAAI 2025 | [arXiv:2409.13203](https://arxiv.org/abs/2409.13203) |
| **From Regex to Transformers: A Hybrid Framework** — Jideani & Gerber, Springer ICC 2025 | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-88720-8_5) |

### Prompt engineering & latency

| Paper | Link |
|-------|------|
| **Compress, then prompt** — Xu et al., 2023 | [arXiv:2303.05732](https://arxiv.org/abs/2303.05732) |
| **Parallel Prompt Decoding** — Chen et al., FPGA 2024 | [PDF](https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/eee/csc-group/ChenH_ParallelPromptDecoding_FPGA24.pdf) |
| **Survey on Inference Engines for LLMs** — Park et al., ACM TOCS 2025 | [ACM](https://dl.acm.org/doi/10.1145/3725362) |
| **Advanced prompt engineering** — Huttula, 2025 | [PDF](https://www.theseus.fi/bitstream/handle/10024/835000/Huttula_Antti.pdf?sequence=2) |

### Data contamination & deduplication

| Paper | Link |
|-------|------|
| **Deduplicating Training Data** — Lee et al., 2021 | [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) |
| **Quantifying Memorization** — Carlini et al., 2022 | [arXiv:2202.07646](https://arxiv.org/abs/2202.07646) |

### Quantization & deployment

| Paper | Link |
|-------|------|
| **TinyUSFM: Compact Foundation Models** — Ma et al., 2025 | [arXiv:2510.19239](https://arxiv.org/abs/2510.19239) |
| **BrainDistill: Quantization-Aware Distillation** — Xie et al., 2026 | [arXiv:2601.17625](https://arxiv.org/abs/2601.17625) |
| **Qwen3 Technical Report** — Alibaba Cloud | [HuggingFace](https://huggingface.co/Qwen/Qwen3-4B) |

## License

MIT

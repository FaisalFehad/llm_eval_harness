# AI Eval Harness

A pipeline for building a fast, locally-running job-fit scoring model that matches the accuracy of large commercial LLMs — trained on your own taste.

The technique is **LLM knowledge distillation**: use a large model to generate high-quality labeled data, then train a tiny model to mimic it. End result: a model that runs locally via llama.cpp, costs nothing per inference, and scores jobs the way _you_ would.

---

## Progress

```text
1. Ground truth dataset          ✅  103 hand-scored jobs with 4-category rubric
2. Tournament model selection    ✅  20 models tested across 3 runtimes, narrowed to 1
3. llama.cpp migration           ✅  5.8× faster than Ollama, 0% parse failures
4. Prompt engineering            ✅  5 models tested, Qwen3-4B best (80%)
5. Data correction               ✅  Fixed 20 golden jobs (loc/comp scoring errors)
6. Fine-tune best 4B model       ✅  LoRA fine-tuned Qwen3-4B: 39.8% → 90.9% on held-out test set
7. Generate distillation data    ⬜  Run teacher on large batch of real jobs
8. Train student (Granite:350M)  ⬜  On teacher outputs, measure distillation gap
```

---

## How It Works

```text
golden_jobs.jsonl (103 hand-labeled jobs)
    │
    ▼
sample-test-subset → balanced test sets (10 / 30 / 103 jobs)
    │
    ▼
prompt-lab (quick A/B)  or  eval-runner (full eval)
    │  node-llama-cpp · grammar-constrained JSON · GGUF models
    ▼
results/{prompt-lab,runs}/{timestamp}_{tag}/
    ├── report.md            (accuracy, MAE, bias, confusion matrix)
    ├── eval_results.json    (machine-readable metrics)
    ├── prompt_snapshot.txt  (exact prompt used)
    └── details/*.json       (per-job scoring breakdown)
```

Every eval run is snapshotted — config, prompts, and results — so iterations are fully reproducible.

---

## Phase 1 — Ground Truth Dataset

Before any model touches a job description, I need ground truth to measure against. I hand-labeled 103 jobs against a 100-point scoring rubric. Distribution is intentionally skewed (52 bad_fit, 40 maybe, 11 good_fit) because that reflects the real job market — most jobs aren't a great fit.

Each job is scored across four categories (25 pts each):

| Category         | What scores high                      | Key penalties                 |
| ---------------- | ------------------------------------- | ----------------------------- |
| Role & Seniority | Senior/Staff/Lead Engineer            | Junior, management, unrelated |
| Tech Stack       | Node.js, TypeScript, AI/ML experience | No relevant stack             |
| Location         | Remote UK/global, hybrid London       | **-50** for outside UK        |
| Compensation     | £100k+ base                           | **-30** for below £45k        |

Labels: 70–100 = `good_fit`, 50–69 = `maybe`, 0–49 = `bad_fit`.

Hand-labeling 103 records took longer than expected. Having the explicit rubric was essential — without it, scores drift as fatigue sets in. Midway through, a bad script run wiped all labels from `golden_jobs.jsonl` — went from 88 labeled records to 60 unlabeled ones with no git history to restore from. I had to re-label the surviving 57 real records and generated 46 synthetic jobs using Faker.js to fill gaps and cover edge cases (non-UK locations, missing salary, management roles).

---

## Phase 2 — Model Selection

### Starting point: Promptfoo + Mistral cloud API

I started with [Promptfoo](https://promptfoo.dev) as the eval framework, running evals against Mistral's cloud API (`mistral-large-latest`). The setup was simple — three prompt variants, one cloud provider, `npx promptfoo eval` to run. But it had problems: cloud API calls are slow and expensive for iterating, and I could only test one model at a time. I needed to test local models to find something I could actually fine-tune later.

So I switched to **Ollama** for local model inference. This let me test many models for free, but immediately introduced new problems on my 16 GB MacBook Air.

### Tournament approach

The plan was to run all ~20 Ollama models against all 103 jobs — roughly 2,100 inferences. On my hardware, that's 8–12 hours with no early feedback.

I could have just picked a few models I thought would work, but I honestly had no idea how each one would perform. I could have also kept only the largest models — they're more likely to score well — but given my hardware constraints, I'm happy to trade a little accuracy for speed and not crashing my machine. Many of these models are obviously bad candidates (350M param models, old LLaMA 2, coding-only models). Why wait for a model to finish 103 jobs when it could be dropped after just 10?

So instead: a **3-round tournament** that drops bad models early.

- **Round 1 — Smoke test**: All models, 10 balanced jobs. Drop if accuracy < 40%, parse failures > 30%, or > 120s/job.
- **Round 2 — Qualifying**: Survivors, 30 jobs. Tighter thresholds.
- **Round 3 — Full eval**: Finalists only, all 103 jobs.

### Things that broke along the way

**Prompt-to-golden mismatch.** Before any models ran, I found a critical bug: the scoring prompts had different label thresholds than the golden truth data. The prompt said `bad_fit = 0–39` but the golden data used `bad_fit = 0–49`. A model that scored a job 45 would correctly output `bad_fit` per the golden truth, but the prompt was telling it to output `maybe`. Every model was being evaluated against a rubric it wasn't given. Had to rewrite all prompts to match the golden rubric before any eval results were meaningful.

**Ollama OOM crashes.** Models would run 2–3 jobs fine, then crash with `fetch failed`. Turned out Ollama's default context window (40,960 tokens for qwen3:8b) pre-allocates ~11 GB of KV cache on a 16 GB machine — leaving almost nothing for macOS. Each job only needs ~2,000 tokens, so I was wasting 9 GB on capacity I'd never use. Setting `num_ctx: 4096` dropped RAM usage from ~11 GB to ~3–4 GB and eliminated the crashes.

**Model memory bleeding.** Even after the context fix, models kept crashing when swapping. Ollama holds the previous model in RAM for 5 minutes by default. When the 7B model finished and the 14B tried to load, both were in memory simultaneously — 4.7 GB + 9 GB = game over on 16 GB. Fixed with `keep_alive: 0` to force immediate eviction after each model's test.

**Fail-fast gap.** The initial logic only caught 100% failure rates. A model could succeed on 2 jobs then crash on the next 8, wasting time. Added a `consecutiveErrors >= 3` bail-out to detect mid-run Ollama crashes and move on.

### Smoke test results

**6 of 16 models advanced. 10 eliminated.**

| Model                  | Acc | Parse Fail | MAE  | Bias  | Speed | Result |
| ---------------------- | --- | ---------- | ---- | ----- | ----- | ------ |
| minimax-m2.5 (cloud)   | 70% | 0%         | 20.5 | +13.5 | 20.1s | PASS   |
| qwen3:8b               | 50% | 20%        | 23.1 | +23.1 | 81.2s | PASS   |
| wizardlm2 (~7B)        | 40% | 0%         | 36.7 | +23.7 | 23.5s | PASS   |
| mistral-openorca:7b    | 40% | 0%         | 31.5 | +29.5 | 14.1s | PASS   |
| dolphin-llama3 (~8B)   | 40% | 0%         | 37.0 | +37.0 | 13.5s | PASS   |
| llama2:7b              | 40% | 0%         | 38.5 | +37.5 | 15.6s | PASS   |
| _10 models eliminated_ |     |            |      |       |       | FAIL   |

Some things I didn't expect: all sub-5B models failed outright (~30% accuracy, basically random guessing). The largest local model (qwen2.5:14b) had the _worst_ accuracy — aggressive quantization to fit in 16 GB erased the size advantage entirely. And 14 of 16 models systematically over-scored everything. A model that rates every job as "good" is useless for filtering. Only the cloud model could correctly identify `maybe` jobs — every local model mapped them straight to `good_fit`.

### The latency problem → llama.cpp

Although I was pretty happy with qwen3:8b at 50% accuracy, 81.2s/job is still a long time to wait for iterative prompt engineering. It would be such a pain to wait hours to test each prompt on a decent number of jobs. So I kept trying other models hoping to find a sweet spot of similar accuracy but faster speed.

Then I came across llama.cpp and llama-server as runtimes instead of Ollama. This meant getting new GGUF models (Ollama models aren't compatible), but the speed difference was immediately obvious:

| Runtime   | Model    | Acc | Speed | Parse failures |
| --------- | -------- | --- | ----- | -------------- |
| Ollama    | qwen3:8b | 50% | 81.2s | 20%            |
| llama.cpp | qwen3:8b | 50% | 14.1s | 0%             |

Same model, same accuracy, **5.8× faster**, zero parse failures. That extra headroom also opens the door to using bigger models than what I could previously run on Ollama, and potentially running multiple jobs in parallel.

It wasn't completely smooth though. Running Qwen3 through llama-server, the model started returning empty `{}` responses — valid JSON but no actual content. The issue was Qwen3's thinking mode: the model was spending all its tokens on internal reasoning (`<think>` blocks) and the JSON grammar sampler was only getting the leftovers. Fixed by injecting `/no_think` as a system message. I also hit GPU OOM from prompt cache accumulation — llama-server was caching prompts across slots (~175 MB each), and after 5 jobs it had 781 MB of useless cache that crashed the process. Fixed with `--no-cache-prompt` since each job is independent anyway.

### Ditching Promptfoo and the tournament

At this point I realised the speed improvement changed everything. The tournament was a solution to Ollama's latency — if each job takes 80 seconds, you need to be clever about which models you test. But at 14 seconds per job, I could just test everything. A 10-job eval gives a pretty accurate picture of model performance, and now it only takes 2 minutes instead of 13.

So I rebuilt the whole eval pipeline. Promptfoo doesn't support node-llama-cpp as a provider — it talks to HTTP APIs, not native bindings. I had a choice: write a custom Promptfoo provider plugin, or just build my own eval runner. I built `eval-runner.ts` — it reads the same Promptfoo config YAML (keeping the nice schema and test format) but runs inference directly through node-llama-cpp. Grammar-constrained decoding means 0% parse failures — the model can only produce tokens that form valid JSON matching the expected schema. Parse failures had been a constant headache with Ollama; with grammar constraints they just disappeared.

I deleted `tournament.ts` (1,235 lines of Ollama client code, llama-server spawning, and 3-round tournament logic) and replaced it with `eval-runner.ts` (~500 lines). Promptfoo is still in the project — I use it for the YAML config schema and test generation — just not as the runtime anymore.

### Baseline results (node-llama-cpp)

Re-ran all candidates through node-llama-cpp with grammar-constrained JSON. 10 jobs, seed 42, balanced sampling.

| #   | Model                  | Params | Acc     | MAE      | Bias     | Speed |
| --- | ---------------------- | ------ | ------- | -------- | -------- | ----- |
| 1   | **gemma-3-4b-it**      | 4B     | **60%** | **26.8** | +26.8    | 11.7s |
| 2   | qwen3-8b-official      | 8B     | 50%     | 36.0     | +36.0    | 25.1s |
| 3   | qwen3-4b-instruct-2507 | 4B     | 50%     | 35.0     | +35.0    | 16.8s |
| 4   | meta-llama-3.1-8b      | 8B     | 40%     | 36.0     | +16.4    | 19.8s |
| 5   | ministral-3-8b-2512    | 8.4B   | 30%     | **24.5** | +7.5     | 22.4s |
| 6   | qwen2.5-7b-instruct    | 7.6B   | 30%     | 30.4     | **-2.6** | 16.0s |
| 7   | glider (PatronusAI)    | 3.8B   | 30%     | 31.6     | +23.4    | 29.7s |
| 8   | mistral-7b-instruct    | 7B     | 10%     | 22.8     | +5.2     | 16.7s |

**Gemma-3-4B-IT** is the clear winner — and it wasn't even supposed to be. It's the only model with both 100% good_fit detection and non-zero bad_fit detection. Every other model is either a "yes-man" that approves everything or a pessimist that rejects everything:

| Model             | good_fit (of 4) | maybe (of 3) | bad_fit (of 3) |
| ----------------- | --------------- | ------------ | -------------- |
| **gemma-3-4b-it** | 4/4 (100%)      | 1/3 (33%)    | 1/3 (33%)      |
| qwen3-8b          | 4/4 (100%)      | 1/3 (33%)    | 0/3 (0%)       |
| ministral-3-8b    | 0/4 (0%)        | 1/3 (33%)    | 2/3 (67%)      |
| qwen2.5-7b        | 0/4 (0%)        | 2/3 (67%)    | 1/3 (33%)      |

A few other things I found along the way: thinking mode (`think` vs `no_think`) never helped on any model — zero accuracy improvement, up to 38% slower. Model size doesn't predict quality — the 4B Gemma beat every 7B and 8B model tested. And Ministral has the best score calibration (24.5 MAE) and best bad_fit detection (67%) but can't recognise a good job to save its life — worth revisiting with prompt work.

### Verdict

```text
20 models (Ollama + cloud)
    │  Round 1: 10 jobs, drop <40% acc / >30% parse / >120s
    ▼
6 survivors
    │  Migrate to node-llama-cpp (5.8× faster, 0% parse fail)
    ▼
8 GGUF models re-baselined
    │  10 jobs, seed 42, grammar-constrained JSON
    ▼
Gemma-3-4B-IT (60%) ← best baseline, sent to prompt engineering
```

After testing and benchmarking tens of models across three runtimes and running hundreds of eval tests, I've gathered enough data to make decisions about which model to tune and what their weaknesses are. The prompt is still the bottleneck — all models struggle with the same failure modes. So I'll start optimising my prompt to get as much as possible from Gemma and see how far it can go before fine-tuning.

---

## Phase 4 — Prompt Engineering

Starting from the v9 prompt (the best from Gemma optimization), each model was tested with the same baseline, then iterated with targeted prompt changes until accuracy plateaued. The pattern: most models hit ceiling on v9 itself — additional prompt complexity consistently degraded accuracy on small models.

### Gemma-3-4B-IT

Gemma-3-4B-IT tops out at **70% label accuracy** (v6c prompt, 10-job sample). Tested 5 prompt variants (v6–v9) — adding more examples, IF-THEN rules, or zero-point demonstrations all degraded accuracy. The remaining 30% misses are model-level limitations: hallucinating keywords not in the input, arithmetic errors on negative sums, and ignoring "Up to £X" comp rules. 70% appears to be this model's ceiling for prompt-only optimisation.

### Qwen3-4B-Instruct-2507

Tested Qwen3-4B as an alternative to Gemma, starting from the same v6 prompt that was Gemma's best. Qwen3-4B immediately outperformed Gemma — **80% label accuracy** on the first run vs Gemma's 70% ceiling — using the exact same prompt and rubric.

| Version              | Prompt change                                         | Accuracy | MAE      | Bias      | Speed     | Status           |
| -------------------- | ----------------------------------------------------- | -------- | -------- | --------- | --------- | ---------------- |
| v9 (baseline)        | Gemma v6 prompt, no changes                           | **80%**  | 12.5     | +12.5     | 55.4s     | Best prompt      |
| v10                  | Added CAUTION gate, Example C, empty-field extraction | 60%      | 14       | +14       | 111.3s    | Regressed        |
| v11                  | v9 + rewording "Up to" rule only                      | 70%      | 15       | +15       | 27.4s     | Regressed        |
| **v9 + infra fixes** | **Same v9 prompt, GC between jobs**                   | **80%**  | **12.5** | **+12.5** | **28.1s** | **Best overall** |

**Key finding:** more verbose prompts hurt this 4B model. v10 added ~50% more prompt text and accuracy dropped by 20 percentage points. v11 changed a single line and still introduced a regression. The v9 baseline (identical to Gemma v6) remains the best prompt for both models.

#### Issues and fixes

**RAM pressure causing timeouts and degraded inference.** The first v10 run got 5/5 correct on completed jobs, then timed out at 421s on job 6. The second run completed all 10 but only hit 60% accuracy. Same prompt, different results — RAM pressure was degrading inference quality. Fixed by adding `global.gc?.()` + 100ms pause between jobs and `--expose-gc` to the tsx invocation. This cut latency from 55.4s to 28.1s per job — a **2× speedup**.

**Context size truncation.** Reducing `context_size` from 4096 to 2048 caused accuracy to drop. The longest JD in the test set is 6,295 characters (~1,574 tokens); combined with the ~470-token prompt, total context hits ~2,044 tokens — right at the 2048 limit. Reverted to 4096.

#### Remaining misses (2/10)

Both misses are consistent across all runs — the model gets them wrong every time regardless of prompt wording:

1. **Spencer Inc** (expected: maybe 50, predicted: good_fit 75): The model's reasoning is correct — it computes loc=10, role=0, tech=15, comp=25 = 50 = maybe. But its JSON output says 75/good_fit. This is a reasoning-to-output mismatch: the model talks itself to the right answer then writes the wrong number.

2. **Fintech "Up to £150K"** (expected: bad_fit 25, predicted: good_fit 90): The model ignores the "Up to £X = comp 0" rule entirely, treating "Up to £170K" as a £170K midpoint and scoring comp=25. Three different prompt wordings for this rule all failed — the model simply doesn't apply negative/zero-scoring rules to high-looking numbers.

Both are model-level limitations, not prompt issues — candidates for fine-tuning.

#### Results

Qwen3-4B-Instruct-2507 with the v9 prompt is the best local model tested so far: **80% accuracy at 28s/job**, beating Gemma's 70% ceiling with the same prompt. The remaining 20% errors are model-level limitations (reasoning/output mismatch, ignoring zero-score rules) that prompt engineering alone can't fix. I would like to try my luck with a larger model, hopefully it will be better at applying the rules and doing arithmetic, but for now Qwen3-4B has gone a long way!.

### Meta-Llama-3.1-8B-Instruct

Tested Llama 3.1 8B as a larger (8B) alternative, hoping the extra parameters would help with rule-following and arithmetic. Previously scored 40% in the baseline tournament — started from the same v9 prompt.

| Version       | Prompt change                                  | Accuracy | MAE  | Bias  | Speed | Status  |
| ------------- | ---------------------------------------------- | -------- | ---- | ----- | ----- | ------- |
| v9 (baseline) | Gemma v6 prompt, no changes                    | 60%      | 15.5 | -15.5 | 23.1s | Best    |
| v10           | Explicit if/then keyword matching + 4 examples | 60%      | 18.5 | -14.5 | 20.9s | No gain |

**Key finding:** Llama 3.1 8B has the opposite problem to Qwen — it **underscores** (bias -15.5) instead of overscoring (+12.5). It has perfect bad_fit detection (3/3) and correctly handles the "Up to £X = comp 0" rule that Qwen can't learn. But it can't reliably detect keywords in job titles — scoring "Senior Backend Engineer" as role=0 and "London Area, United Kingdom" as loc=0.

| Trait            | Qwen3-4B       | Llama 3.1 8B   |
| ---------------- | -------------- | -------------- |
| Accuracy         | **80%**        | 60%            |
| Bias             | +12.5          | -15.5          |
| good_fit recall  | **4/4 (100%)** | 1/4 (25%)      |
| bad_fit recall   | 2/3 (67%)      | **3/3 (100%)** |
| "Up to £X" rule  | Ignores it     | **Correct**    |
| Keyword matching | **Solid**      | Broken         |

v10 added explicit if/then keyword matching with worked examples for the exact failing patterns ("Senior Backend Engineer" → 25, "Lead DevOps Architect" → 25). This fixed role detection for some jobs but broke tech stack scoring for others — net zero. The ScaleXP failure is particularly bad: the model scores 0 on a job where the title is "Senior Backend Engineer" and the location is "London Area, United Kingdom" — it can't even read the input fields.

Model size doesn't help here — the 8B Llama is worse than the 4B Qwen at this task. Not worth further prompt iteration.

### Qwen2.5-7B-Instruct

Tested Qwen2.5-7B as a middle ground — same architecture family as the winning Qwen3-4B but larger (7B) and an older generation. Previously scored 30% in the baseline tournament with the old prompt.

| Version       | Prompt change                                          | Accuracy | MAE  | Bias | Speed | Status  |
| ------------- | ------------------------------------------------------ | -------- | ---- | ---- | ----- | ------- |
| v9 (baseline) | Gemma v6 prompt, no changes                            | 70%      | 10.5 | +8.5 | 21.5s | Best    |
| v10           | Stricter comp rules, "Up to" example, 4 CRITICAL rules | 70%      | 12.0 | +6.0 | 23.5s | No gain |

**Key finding:** v9 jumped from 30% → 70% (the structured prompt is doing most of the work). Bias is positive (+8.5) like Qwen3-4B, but the model has two unfixable problems:

1. **Salary hallucination** — fabricates GBP salary figures for jobs that don't have them (Oracle US job predicted as having "£79,800-£178,100")
2. **Ignores "Up to £X" rule** — despite a worked example showing exactly this pattern (Owen Thomas, +60 error), the model still scores comp=25

v10 added CRITICAL rules and Example C targeting these failures. Result: fixed Happl but regressed Lead DevOps — net zero. The Owen Thomas miss (+60 error) persisted unchanged, proving the model can't learn this rule from prompt alone.

70% is this model's ceiling — same issues as other models, just different flavours.

### WizardLM-2-7B

Tested WizardLM-2-7B as a creative/reasoning-oriented alternative. Previously scored 40% in the baseline tournament.

| Version       | Prompt change               | Accuracy | MAE  | Bias  | Speed | Status  |
| ------------- | --------------------------- | -------- | ---- | ----- | ----- | ------- |
| v9 (baseline) | Gemma v6 prompt, no changes | 60%      | 20.0 | +20.0 | 23.0s | Yes-man |

**Key finding:** the model is a **yes-man** — all 5 completed predictions were identical: score 90, good_fit, with the reasoning copy-pasted verbatim from Example A ("Senior role in London with Node.js/TS and £100k midpoint salary"). Only 5 of 10 jobs completed before timeout.

The model isn't applying the scoring rubric — it's parroting the worked example regardless of input. This is worse than overscoring; it's zero discrimination between jobs. Not worth iterating.

### Cross-model comparison

| Model                      | Size | Best Accuracy | MAE  | Bias  | Key weakness                      |
| -------------------------- | ---- | ------------- | ---- | ----- | --------------------------------- |
| **Qwen3-4B-Instruct-2507** | 4B   | **80%**       | 12.5 | +12.5 | "Up to £X" overscoring            |
| Gemma-3-4B-IT              | 4B   | 70%           | 26.8 | +26.8 | Keyword hallucination, arithmetic |
| Qwen2.5-7B-Instruct        | 7B   | 70%           | 10.5 | +8.5  | Salary hallucination              |
| Meta-Llama-3.1-8B-Instruct | 8B   | 60%           | 15.5 | -15.5 | Can't detect keywords             |
| WizardLM-2-7B              | 7B   | 60%           | 20.0 | +20.0 | Yes-man (parrots example)         |

Model size doesn't predict accuracy — the 4B Qwen3 beats every 7B and 8B model. All models share the same v9 prompt baseline; additional prompt complexity consistently hurts small models. The remaining errors are model-level limitations (hallucination, arithmetic, rule-ignoring) that only fine-tuning can address.

### Fine-tuning candidate: Qwen3-4B

Looking at the data, Qwen3-4B is the clear choice for fine-tuning — and it changes the original plan.

| Why Qwen3-4B                | Detail                                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------------------------- |
| Highest baseline            | 80% — less to fix, more likely to push past 90%                                                          |
| Systematic errors           | Same 2 misses every run ("Up to £X" rule, reasoning-to-output mismatch) — exactly what fine-tuning fixes |
| Smallest model              | 4B params — cheapest/fastest to fine-tune, already beats every 7B and 8B                                 |
| Others have harder problems | Qwen2.5 hallucinates salaries, Llama can't parse inputs, WizardLM parrots examples                       |

---

## Phase 5 — Fine-tuning (in progress)

### First attempt: MLX LoRA on Qwen3-4B

Built a full LoRA fine-tuning pipeline targeting Qwen3-4B on Apple M1 (16GB):

```text
golden_jobs.jsonl (103 hand-labeled)
    │
    ├─ compute-breakdowns.ts → add loc/role/tech/comp per job
    ├─ split-train-test.ts   → 70 train / 33 held-out test (stratified, seed=42)
    ├─ format-for-mlx.ts     → chat-format JSONL for MLX LoRA
    │
    ▼
MLX LoRA training (rank=8, 400 iters, lr=1e-5)
    │
    ▼
Fuse → GGUF → eval on held-out 33
```

**Training config** (MLX LoRA on M1 16GB):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| rank | 8 | 70 examples + narrow task = low rank prevents overfitting |
| lr | 1e-5 | Model already 80% correct, small adjustments needed |
| iters | 400 | ~5.7 epochs over 70 examples |
| batch_size | 1 | M1 16GB constraint |
| mask_prompt | true | Only train on assistant responses (not the 470-token prompt) |
| grad_checkpoint | true | Required — without it, OOMs on M1 16GB |

### Things that broke

**mlx-lm API change.** mlx-lm v0.30+ changed `alpha` to `scale` in LoRA config. Training crashed with `KeyError: 'scale'`. Fixed by replacing `alpha: 16` with `scale: 2.0` (= alpha/rank = 16/8).

**NaN loss from sequence truncation.** Default `max_seq_length` is 2048 tokens. The longest training example is ~2,350 tokens. MLX silently truncated examples mid-response, creating invalid training targets. Loss went to NaN at iter 60. Fixed with `max_seq_length: 4096`.

**OOM crashes.** With `max_seq_length: 4096`, the training used ~5.6 GB peak memory with `grad_checkpoint: true`. Tried disabling grad_checkpoint for speed (~30-40% faster) — immediately OOMed. Also tried `max_seq_length: 2560` without grad_checkpoint — still OOMed. On M1 16GB, `grad_checkpoint: true` is mandatory. Training takes ~3 hours.

**Loss curve before crash** (looked healthy):

```
Iter 10: loss 3.508
Iter 20: loss 1.766
Iter 30: loss 0.974  ← learning format + patterns quickly
Iter 40: loss 0.731
Iter 50: loss 0.505
Iter 60: NaN         ← truncation corruption
```

### Golden data quality problem

Before restarting training, reviewed the training data quality. Found **20 of 103 golden jobs had wrong labels**:

- **13 location errors:** Jobs with "London Area, United Kingdom" in the location field were hand-scored as "Outside UK (-50)" instead of "London, UK (+25)". A 75-point swing per job.
- **8 compensation errors:** Jobs with no salary stated were scored as "Salary <£45k (-30)" instead of "No salary (0)". One overlap with location errors.

These aren't judgment calls — they're mistakes in applying the rubric. The rubric (v9) is unchanged.

**Impact on previous work:**

| What | Status | Why |
|------|--------|-----|
| Scoring rubric (v9) | Still valid | Rules didn't change |
| Eval infrastructure | Still valid | Pipeline is reusable |
| Relative model rankings | Likely still valid | All models measured against the same data |
| Prompt engineering insights | Still valid | "Simple prompts beat complex ones" is still true |
| Absolute accuracy numbers | Invalid | 80% was measured against wrong answers |
| Training data | Must regenerate | Can't train on wrong labels |

Label distribution shifted after fixes: bad_fit 52→39, maybe 40→51, good_fit 11→13. Several jobs that were penalised for "wrong location" were actually London-based and should have scored higher.

### Revised approach

Rather than just retraining on corrected data, taking a step back to re-validate model selection:

```text
golden_jobs.jsonl (103 corrected)
    │
    ├─ Old pipeline preserved (original labels, for reference)
    │
    ├─ New tests generated from corrected labels
    │
    ▼
Head-to-head eval: Qwen3-4B vs Gemma-3-4B
    │  Both on v9 prompt, all 103 jobs
    │
    ▼
Failure analysis
    │  Categorise misses: systematic (trainable) vs random (not trainable)
    │
    ▼
Select winner → LoRA fine-tune on 70 train jobs
    │
    ▼
Eval on 33 held-out → target 90%+
```

**Why re-evaluate instead of just retraining Qwen3-4B?**

1. The old 80% accuracy number is invalid — Qwen3 might not still be the winner
2. Training larger models (7B/8B) is impossible on M1 16GB — only 4B fits, so it's between these two
3. The model with more *systematic* (trainable) failures is the better candidate, even if its raw accuracy is slightly lower
4. Both pipelines (old and new) are preserved — can always go back and verify

### Qwen3-4B on corrected data: 39.8%

Ran Qwen3-4B-Instruct-2507 on all 103 corrected jobs with v9 prompt. The results were a shock:

| Metric | Old data (10 jobs) | Corrected data (103 jobs) |
|--------|-------------------|--------------------------|
| Accuracy | 80% | **39.8%** |
| Parse fail | 0% | 0% |
| MAE | 12.5 | 25.9 |
| Bias | +12.5 | **+23.1** |

The 80% → 40% drop makes sense in hindsight. Previously tested on only 10 jobs — each worth 10 percentage points. And those 10 were measured against wrong labels, so "correct" predictions may have been wrong answers matching wrong data. Testing on all 103 with corrected labels reveals the true accuracy.

**Confusion matrix:**

```text
Actual \ Predicted  | good_fit | maybe | bad_fit
--------------------|----------|-------|--------
good_fit (13)       |    13    |   0   |    0     ← 100% correct
maybe    (51)       |    32    |  19   |    0     ← 37% correct
bad_fit  (39)       |    13    |  17   |    9     ← 23% correct
```

The model **never under-predicts** — zero cases of predicting a lower label than actual. It always scores too high. It correctly identifies every good_fit but can't recognise bad jobs.

### Finding: the model thinks correctly but writes wrong answers

This is the most interesting failure pattern I've found. Looking at individual results, the model's reasoning is often **correct** but the JSON values are **wrong**:

**Example — Backend Systems Developer (Edinburgh):**
- Reasoning: *"no Senior keyword, role=0... no Node.js, tech=0... salary in USD, comp=0... total=10 → bad_fit"*
- JSON output: `{"loc":10,"role":25,"tech":10,"comp":25,"score":70,"label":"good_fit"}`
- The reasoning says bad_fit. The JSON says good_fit. **Both from the same response.**

**Example — Senior SWE in Bangalore, India:**
- Reasoning: *"India, outside UK, loc=-50... total=0 → bad_fit"*
- JSON output: `{"loc":10,"role":25,"tech":10,"comp":25,"score":70,"label":"good_fit"}`
- Knows it's India (-50). Writes loc=10.

This happens on ~45 of the 62 wrong answers. The model walks through the rules correctly in its reasoning, then outputs inflated numbers in the JSON.

**Why?** The JSON format puts scores **before** reasoning: `{"loc":X,"role":X,...,"reasoning":"..."}`. The model generates the numbers first (defaulting to high values), then reasons about them after. By the time it works out the correct answer, the numbers are already committed. It's writing the answer before reading the question.

**Specific failure patterns across all 103 jobs:**

| Pattern | Count | What happens |
|---------|-------|-------------|
| role defaults to 25 | ~30 | "Software Engineer", "Product Developer", "Engineering Manager" all get role=25 despite no senior keyword |
| tech defaults to 10-15 | ~25 | Java, Rust, Apollo/GraphQL, AWS all get tech points despite not being Node.js/TS/AI |
| comp defaults to 25 | ~20 | "Up to £X", USD salaries, and sub-£55k all get comp=25 |
| loc won't go negative | ~8 | Bangalore, Prague, Berlin, Vancouver all get loc=10 instead of -50 |
| Geography error | ~2 | Dublin, Ireland scored as "UK city" — Ireland is not in the UK |

The model has a "default high" bias — when uncertain, it writes 25 for every category. It can reason its way to 0 or -50, but can't bring itself to actually write those values.

### Approaches considered

**1. Train directly on corrected data (keep current format)**

Just regenerate training data with correct labels and train. Simple, but doesn't address the root cause — the model would still generate scores before reasoning. Training might force correct patterns through repetition, but we'd be fighting the format.

**2. Change output format (reasoning before scores) + train**

Move reasoning before the score values in the JSON, so the model thinks first. But this requires updating the training data format, the eval parser, the prompt, and the test assertions — significant rework. And changing the format is prompt engineering, not training strategy. Should test it separately first.

**3. Test format change first, then train without changing data** ← chosen

The key insight: a format change is just prompt engineering and can be tested in 15 minutes with zero code changes to the training pipeline. If putting reasoning first improves accuracy from 40% to 60%, that tells us the model genuinely understands the rules (the "correct reasoning" isn't post-hoc rationalisation). If it doesn't help, we know the model's understanding is shallower than the reasoning text suggests.

Either way, we get **better diagnostic information** — genuine reasoning rather than afterthought — which directly shapes what training data we need. Then we train with the existing simple JSON format (no reasoning in training data), keeping the data pipeline unchanged.

**Why this is the right order:**
- Prompt change: 15 minutes to test, free information about model comprehension
- Training: 1-3 hours on M1, expensive to iterate, need to get it right first
- The diagnostic eval tells us *what* to train on — don't train blind

### Gemma-3-4B on corrected data: 44.7%

| Metric | Qwen3-4B | Gemma-3-4B |
|--------|----------|------------|
| Accuracy | 39.8% | **44.7%** |
| MAE | 25.9 | **20.2** |
| Bias | +23.1 | **+18.8** |
| Speed | 82s | **22.4s** (4× faster) |

Gemma wins on raw numbers. But the confusion matrices reveal very different failure modes:

```text
QWEN3-4B                              GEMMA-3-4B
Actual\Pred  | good | maybe | bad     Actual\Pred  | good | maybe | bad
-------------|------|-------|----     -------------|------|-------|----
good_fit(13) |  13  |   0   |  0     good_fit(13) |  10  |   3   |  0
maybe   (51) |  32  |  19   |  0     maybe   (51) |  18  |  33   |  0
bad_fit (39) |  13  |  17   |  9     bad_fit (39) |   8  |  28   |  3
```

Gemma is better at `maybe` (65% vs 37%) but worse at `good_fit` (77% vs 100%) and `bad_fit` (8% vs 23%). Neither model ever under-predicts — both have a positive bias.

### Finding: same accuracy, completely different failure modes

**Qwen understands the rules but writes wrong values.** Its reasoning walks through the rubric correctly — "no Senior keyword, role=0" — then the JSON says `role:25`. The model thinks correctly but outputs wrong numbers. This is a structural problem (JSON values come before reasoning in the output format).

**Gemma doesn't understand the rules at all.** Its reasoning is brief, confident, and **consistent** with its wrong JSON. When Gemma gives `tech=25` for a Ruby on Rails job, its reasoning says "Ruby on Rails, PostgreSQL are listed (+25)." Gemma genuinely believes any tech stack deserves points.

Gemma-specific errors not seen in Qwen:
- **Awards tech points for ANY tech** — Ruby on Rails gets `tech=25`, Rust/WASM gets `tech=25`, Golang/gRPC gets `tech=20`. Gemma doesn't understand only Node.js/TS/AI qualify.
- **Scores US locations as +25** — DigniFi in "United States" gets `loc=25`. Reasoning says "in the US" but still gives maximum location score.
- **Converts USD to GBP** — "$120k-$150k, which equates to approximately £96k-£120k" → `comp=25`. The rubric says ignore USD entirely. Gemma invents a currency conversion rule.

Both models share: "Up to £X" ignored, Dublin=UK geography error, general positive bias.

### Training candidate: Qwen3-4B (confirmed)

Despite lower raw accuracy, Qwen is the better training candidate:

| | Qwen3-4B | Gemma-3-4B |
|---|---|---|
| Root cause | Understands rules, outputs wrong values | Doesn't understand the rules |
| Training task | Pattern correction (easier) | Comprehension improvement (harder) |
| Failure uniformity | One fix (learn to write zeros) covers most errors | Many diverse error types |

Teaching a model to **output** different values from 70 examples is easier than teaching it to **understand** new rules from 70 examples. Qwen already has the knowledge — it just needs to learn the correct output patterns.

### Next step: diagnostic prompt before training

Before training, one more experiment. The v9 prompt puts JSON values before reasoning — the model writes the answer before thinking. A v10 prompt with reasoning-before-scores might:

1. **Improve accuracy** — model thinks before committing to numbers
2. **Reveal true comprehension** — is Qwen's "correct" reasoning genuine thinking or post-hoc rationalisation?

Either outcome gives us better information for training. This is a 15-minute prompt experiment vs hours of training — worth testing first.

If the format change helps, we train with the current simple JSON format (no reasoning in training data) because:
- Training teaches the model correct output patterns through repetition
- A well-trained model doesn't need to reason at inference time
- Simpler format = shorter sequences = faster training = less memory

```text
1. Create v10 prompt (reasoning-first format)         ← next
2. Test v10 on Qwen3-4B (103 jobs)                    ← does format alone help?
3. Train with simple JSON format, oversampling failures
4. Eval on 33 held-out test set → target 90%+
```

### Training run: LoRA fine-tune on Qwen3-4B

Skipped the reasoning-first prompt experiment and trained directly on the corrected data with oversampling of failure cases.

#### Why LoRA?

LoRA (Low-Rank Adaptation) freezes most of the model's weights and adds small trainable matrices to a few attention layers. Instead of updating all ~4 billion parameters, you update ~0.3% of them. This matters for three reasons:

1. **It fits on M1 16GB.** Full fine-tuning a 4B model requires storing gradients for every parameter — roughly 3–4× the model size in memory. That's 12–16 GB just for the optimizer state, before the model itself. Impossible. LoRA keeps optimizer state tiny.

2. **It's appropriate for what we're fixing.** The diagnostic found Qwen already *knows* the rules — it reasons correctly, then writes wrong values. We're not teaching new knowledge; we're correcting output patterns. LoRA is designed for adaptation, not re-teaching. A large update to the whole model would risk damaging what already works.

3. **70 training examples is tiny.** Full fine-tuning with 70 examples would overfit catastrophically — the model would memorise the training set and forget everything else. LoRA's low-rank constraint acts as strong regularisation, limiting how much the model can change.

#### Why these specific hyperparameters?

**rank=4** (not 8, not 16): Rank controls how many new "directions" the adapter can learn. Rank 8 was in the first (failed) attempt — this task is even simpler (pattern correction, 70 examples, narrow domain), so we went lower. Lower rank = fewer parameters = harder to overfit = more likely to generalise.

**lr=1e-5** (conservative): The model is already 80% correct. Small learning rate means small corrections. A larger rate (1e-4) would risk overwriting the knowledge the model already has.

**mask_prompt=true**: The training prompt is ~470 tokens of instructions and examples. Without masking, the model would compute loss on those tokens too — trying to "learn" to reproduce the prompt itself, which is useless. With masking, loss is computed only on the assistant's JSON response (~100 tokens). We're training the output, not the instructions.

**grad_checkpoint=true**: Normally, the backward pass caches all intermediate activations (forward pass results) to compute gradients efficiently. On M1 16GB, storing those caches for a 4B model with `max_seq_length: 2560` requires more memory than available. Gradient checkpointing trades compute for memory: recomputes activations on the fly instead of storing them, using ~40% more compute but reducing peak memory enough to fit. Without it, training OOMs every time.

**Stopped at iter 200**: The original plan was 400 iterations. But training loss (0.108) and validation loss (0.114) had converged by iter 200 — a gap of only 0.006. Continuing to 400 would widen this gap as the model begins memorising training examples rather than learning the pattern. Train/val gap is the overfitting signal — stop when it's near zero.

#### Data preparation

- 70/33 stratified split (seed=42) — both sets have proportional label distribution
- **Oversampling** hard failure cases → 93 training examples from 70 jobs:
  - Non-UK locations (India, US, Europe): the `loc=-50` case the model consistently got wrong
  - Non-senior titles ("Software Engineer", "Engineer", "Product Developer"): `role=0` case
  - "Up to £X" compensation: `comp=0` case the model always ignored
  - Non-qualifying tech (Java, Rust, Ruby, AWS-only): `tech=0` case

  Oversampling means these patterns appear more frequently in training so the model sees enough examples of each failure mode to correct the habit. Without oversampling, `loc=-50` cases are rare in the 70-job set — the model could easily learn to approximate everything without ever correctly learning the penalty.

- `mask_prompt: true` — loss computed on assistant responses only (not the 470-token prompt)

**Training config (corrected from earlier attempt):**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| rank | 4 | Narrow pattern-correction task — low rank prevents overfitting |
| scale | 4.0 | Effective alpha = 16 (mlx-lm v0.30+ uses scale, not alpha) |
| lr | 1e-5 | Conservative — correcting output patterns, not teaching new knowledge |
| iters | 200 | Stopped at 200 (of 400 planned) when val loss converged |
| batch_size | 1 | M1 16GB constraint |
| grad_checkpoint | true | Required — without it, OOMs on M1 16GB |
| max_seq_length | 2560 | Longest example ~2350 tokens + headroom |

**Loss curve (stopped at iter 200):**

```
Iter   1: val  loss 3.846
Iter  10: train loss 3.690
Iter  50: train loss 0.573
Iter 100: train loss 0.222  ← checkpoint saved
Iter 150: train loss 0.152
Iter 200: train loss 0.108 / val loss 0.114  ← checkpoint saved, training stopped
```

Train/val gap at iter 200: **0.006** — near-zero, no overfitting. Stopped here rather than continuing to 400 iterations where the gap would likely widen.

### Fine-tune results: 39.8% → 90.9%

Eval on 33 held-out test jobs (never seen during training):

| Metric | Baseline (v9 prompt) | Fine-tuned (iter 200) |
|--------|---------------------|----------------------|
| Label accuracy | 39.8% (103 jobs) | **90.9%** (33 jobs) |
| Parse failures | 0% | 0% |

**Field accuracy:**

| Field | Accuracy | Notes |
|-------|----------|-------|
| loc | 97.0% | Correctly writes -50 for non-UK locations |
| role | 93.9% | Stopped defaulting to 25 for non-senior titles |
| comp | 87.9% | Handles "Up to £X" = 0 correctly |
| tech | 75.8% | Weakest — requires judgement on "required vs mentioned" |

**Per-label breakdown:**

| Label | Accuracy | Notes |
|-------|----------|-------|
| maybe | 17/17 = **100%** | Was the hardest class before fine-tuning |
| bad_fit | 11/12 = **92%** | Systematic over-scoring bias eliminated |
| good_fit | 2/4 = 50% | Only 4 examples — too few to draw conclusions |

**What changed:** The model stopped defaulting to high values (role=25, comp=25, loc=10) when uncertain. It now correctly writes zeros and negatives when the rules require it. The reasoning was already correct before fine-tuning — the training corrected the output patterns to match.

**Remaining weakness:** tech scoring at 75.8% requires understanding whether a skill is "explicitly required/core" vs just mentioned in passing — a judgement call that benefits from more training examples.

### Generalisation eval: real unseen UK jobs (81.9%)

The 90.9% result came from the held-out test set — 33 jobs drawn from the same 103-job pool used for training. A valid test, but the data distribution is closely related to what the model trained on. To check whether the improvement is real or just memorisation, I ran it on a completely separate dataset: 72 real UK jobs scraped fresh from LinkedIn, never touched during any part of this project.

**Dataset:** 72 LinkedIn UK jobs, labels auto-generated by the deterministic scorer. Distribution: 28 `maybe`, 44 `bad_fit`, 0 `good_fit` (most jobs have no GBP salary in the JD, capping max score at 50).

| Metric | Baseline (v9, no fine-tune) | Fine-tuned (iter 200) |
|--------|-----------------------------|-----------------------|
| Label accuracy | ~40% (extrapolated) | **81.9%** (59/72) |
| Parse failures | 0% | 0% |

**Field accuracy:**

| Field | Accuracy | Notes |
|-------|----------|-------|
| comp | 97.2% | Nearly perfect — correctly scores missing GBP salary as 0 |
| tech | 88.9% | Good — Node.js/TS detection holding up on real data |
| role | 86.1% | Good — seniority keyword matching reliable |
| loc  | 65.3% | **Weakest** — struggles with ambiguous UK location strings |

**Per-label breakdown:**

| Label | Accuracy | Notes |
|-------|----------|-------|
| bad_fit | 41/44 = **93%** | Correctly rejects non-qualifying UK jobs |
| maybe   | 18/28 = **64%** | 10 misses — mostly location misscoring |

**What the drop from 90.9% → 81.9% means:** expected and honest. The held-out test set shares the same data distribution as the training set (same 103 jobs, same rubric, same scoring patterns). The LinkedIn dataset is genuinely different — varied location strings, sparse JDs, real-world noise. 81.9% on completely unseen real-world data confirms the fine-tuning generalised, not just memorised.

**Main remaining weakness:** location scoring at 65.3%. The `maybe` misses are mostly location-driven — a job that should score `loc=25` (London hybrid) gets `loc=0` (unclear), dropping the total below 50 into `bad_fit`. The model struggles with varied UK location string formats: "Greater London, England, United Kingdom (Hybrid)", "London Area, United Kingdom", etc.

---

## Running It

```bash
# Build test sets from golden data
npm run golden:sample -- --input data/jobs_export.jsonl --count 100 --seed 42
npm run golden:validate:strict

# Run eval (node-llama-cpp, auto-downloads models)
npm run eval:runner

# Tagged eval run (snapshots config + results)
npm run eval:tagged -- --tag "prompt-v2"

# Compare runs
npm run compare:runs
```

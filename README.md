# AI Eval Harness

A pipeline for building a fast, locally-running job-fit scoring model that matches the accuracy of large commercial LLMs — trained on your own taste.

The technique is **LLM knowledge distillation**: use a large model to generate high-quality labeled data, then train a tiny model to mimic it. End result: a model that runs locally via llama.cpp, costs nothing per inference, and scores jobs the way _you_ would.

---

## Progress

```text
1. Ground truth dataset          ✅  103 hand-scored jobs with 4-category rubric
2. Tournament model selection    ✅  20 models tested across 3 runtimes, narrowed to 1
3. llama.cpp migration           ✅  5.8× faster than Ollama, 0% parse failures
4. Prompt engineering            ✅  Gemma-3-4B-IT (70%), Qwen3-4B (80%)
5. Fine-tune teacher (Qwen3:8B)  ⬜  On 70 hand-scored jobs, eval against held-out 30
6. Generate distillation data    ⬜  Run teacher on large batch of real jobs
7. Train student (Granite:350M)  ⬜  On teacher outputs, measure distillation gap
```

---

## How It Works

```text
golden_jobs.jsonl (103 hand-labeled jobs)
    │
    ▼
build-promptfoo-tests → test subsets (10 / 30 / 103 jobs)
    │
    ▼
eval-runner (node-llama-cpp, grammar-constrained JSON)
    │
    ▼
results/runs/{timestamp}_{tag}/
    ├── eval_results.json     (accuracy, MAE, bias, speed)
    ├── details/*.json        (per-model confusion matrices)
    └── promptfooconfig.yaml  (config snapshot for reproducibility)
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

After testing and benchmarking tens of models across three runtimes and running hundreds of eval tests, I've gathered enough data to make decisions about which model to tune and what their weaknesses are. The prompt is still the bottleneck — all models struggle with the same failure modes. So I'll start optimising my prompt to get as much as possible from Gemma and see how far it can go before fine-tuning.

---

# Prompt Optimisation

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

| Version       | Prompt change                                    | Accuracy | MAE  | Bias  | Speed  | Status  |
| ------------- | ------------------------------------------------ | -------- | ---- | ----- | ------ | ------- |
| v9 (baseline) | Gemma v6 prompt, no changes                      | 60%      | 15.5 | -15.5 | 23.1s  | Best    |
| v10           | Explicit if/then keyword matching + 4 examples   | 60%      | 18.5 | -14.5 | 20.9s  | No gain |

**Key finding:** Llama 3.1 8B has the opposite problem to Qwen — it **underscores** (bias -15.5) instead of overscoring (+12.5). It has perfect bad_fit detection (3/3) and correctly handles the "Up to £X = comp 0" rule that Qwen can't learn. But it can't reliably detect keywords in job titles — scoring "Senior Backend Engineer" as role=0 and "London Area, United Kingdom" as loc=0.

| Trait              | Qwen3-4B       | Llama 3.1 8B   |
| ------------------ | -------------- | -------------- |
| Accuracy           | **80%**        | 60%            |
| Bias               | +12.5          | -15.5          |
| good_fit recall    | **4/4 (100%)** | 1/4 (25%)      |
| bad_fit recall     | 2/3 (67%)      | **3/3 (100%)** |
| "Up to £X" rule    | Ignores it     | **Correct**    |
| Keyword matching   | **Solid**      | Broken         |

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

| Model | Size | Best Accuracy | MAE | Key weakness |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 | 4B | **80%** | 12.5 | "Up to £X" overscoring |
| Qwen2.5-7B-Instruct | 7B | 70% | 10.5 | Salary hallucination |
| Meta-Llama-3.1-8B-Instruct | 8B | 60% | 15.5 | Can't detect keywords |

Qwen3-4B remains the winner — smaller, faster, and more accurate than both larger models.

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

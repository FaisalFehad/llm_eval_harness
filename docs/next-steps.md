# Next Steps — Options Considered

After reaching 95.8% accuracy on real-world UK LinkedIn jobs, two main directions were evaluated.

---

## Option A: Knowledge Distillation

### The core idea

Use the fine-tuned Qwen3-4B as a **teacher model** to auto-label thousands of new jobs, then train a tiny **student model** (Granite 350M, SmolLM 360M) to mimic it.

```
Fine-tuned Qwen3-4B (teacher)
    │
    │  Run on 1,000+ new jobs
    │  → generates labeled outputs (loc, role, tech, comp, label)
    ▼
Large labeled dataset
    │
    │  Train tiny model on this data
    ▼
Granite-350M or SmolLM-360M (student)
    → ~10x smaller, ~10x faster, runs on anything
```

The student never sees the original golden labels — it only trains on the teacher's outputs. You're training it to mimic your fine-tuned model's *judgement*, not to score jobs from first principles.

### What to measure: the distillation gap

How much quality survives compression? A 350M model trained on 10,000 teacher-labeled examples on a narrow task often hits 85-88% — because the decision boundaries are clear. This gap is real research — companies publish papers about it.

### What you'd build

1. **Data generation script** — run fine-tuned model on a large batch of fresh LinkedIn jobs, save predictions as new ground truth
2. **Student training pipeline** — same LoRA setup, different (smaller) base model
3. **Comparison eval** — teacher vs student side by side on a held-out set

### Why it's industry-relevant

This is literally how Meta, Google, and Mistral build their smaller models. GPT-4 generated a lot of the data that trained smaller open models. Key concepts:
- Teacher-student training
- Model compression
- Deployment trade-offs
- Specialist small models beating generalist large ones on narrow tasks

### Hardware

Granite-350M trains comfortably on M1 16GB. Inference fast enough to score 100 jobs in the time Qwen3-4B takes for 10.

---

## Option B: Serving as a Real API

### The core idea

Turn the model from a terminal script into a background service that accepts job descriptions over HTTP and returns scores as JSON.

```
Job description (text)
    │
    │  HTTP POST /score
    ▼
FastAPI server
    │  loads model once at startup, keeps it in memory
    ▼
MLX inference (fine-tuned adapter)
    │
    ▼
{"loc":25,"role":25,"tech":0,"comp":0,"score":50,"label":"maybe"}
    │
    ▼
Browser, script, Chrome extension, spreadsheet...
```

### What you'd build

1. **FastAPI app** — `POST /score` accepts `{title, location, jd_text}`, returns scored JSON
2. **Model loading** — load fine-tuned adapter once at startup, keep in memory
3. **Dockerfile** — package everything into a portable container
4. **Simple web UI** — plain HTML form to paste a job description and see the score (optional but satisfying)

### Key concepts you'd learn

**Startup vs request time.** Model loads in ~5 seconds — once, at startup. Requests then feel instant. This is the difference between a service that works and one that feels broken.

**Concurrency.** Two requests arriving simultaneously both need the GPU. You can't run two MLX inferences at once — you need a queue. Every model serving team solves this.

**Containerisation.** Instead of "a script on my machine", you have "a portable black box that runs anywhere". This is how every production ML model gets deployed.

**Monitoring.** `/health` endpoint, latency logging per request, label distribution tracking. Basic observability — skipped by most ML courses, required by every production team.

### The interesting design decision: sync vs async

- **Synchronous**: HTTP request waits ~2-3 seconds for inference, returns result. Simple, works for one user.
- **Async with job queue**: request returns a job ID immediately, client polls for result. Handles many concurrent users. Teaches Celery/RQ — patterns used everywhere in industry.

Start with sync, understand why async exists, implement async if it gets interesting.

### Why it's industry-relevant

The jump from "script" to "service" teaches things training never does. Key concepts:
- Model serving
- REST APIs
- Docker / containerisation
- Production deployment
- Basic observability

---

## How they compare

| | Distillation (A) | Serving (B) |
|--|--|--|
| What you build | Smaller, faster version of the model | Usable service around the model |
| Core new concept | Model compression, teacher-student training | Productionisation, APIs, Docker |
| End result | 350M model that runs on anything | A URL you can call to score jobs |
| Most surprising moment | How little quality you lose going 10× smaller | Loading once and having requests feel instant |
| Industry relevance | Core ML engineering | Core software engineering |

**They're complementary.** Build the API first (around the current model), then swap in the distilled student model later. The API doesn't care which model runs behind it.

---

## Decision pending

Both options are valid next steps. A natural order would be:
1. Build the API (B) — immediately useful, teaches productionisation
2. Run distillation (A) — swap the smaller student model into the same API

This way you get the industry ML experience (distillation) AND the industry engineering experience (serving), and the second step slots into the infrastructure from the first.

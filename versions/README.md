# versions/

Self-contained per-version artifacts. Each `versions/vN/` fully reproduces that experiment.

## Layout (per-V)

```
vN/
  manifest.json     # machine-readable: base model, HF ref, metrics, shas, commands
  README.md         # human-readable summary
  prompts/          # student.txt (champion), student.fix1.txt (variants), teacher.txt
  configs/          # lora.yaml, promptfoo.yaml, scorer.py, hooks.py, tests.jsonl
  data/             # train.jsonl, valid.jsonl (test set shared at data/v12/test_labeled_audited.jsonl)
  scripts/          # train.py, eval.py, sweep.py — frozen copies per V
  eval_results/     # sweep_summary.json, hybrid_summary.json, predictions.jsonl, promptfoo.json
  adapters/
    HF_REF.md       # FF-01/qwen3-{size}-{v} + checkpoint list. Binaries NOT in git.
  docs/             # V-specific docs (IMPLEMENTATION.md, PLAN.md, RUNBOOK.md)
  experiments/      # sub-experiments (exp1-fix1, exp2-fix3, etc.)
```

## Shared

- `lib/vN/` — frozen copies of shared lib code (semantic_tokens, compute_hybrid, deterministic_baseline) at V-time
- `data/v12/test_labeled_audited.jsonl` — 239-job golden test set, chmod 444, shared across all Vs
- `./models/` — gitignored local cache (base models, downloaded adapters, merged, GGUF) populated by `harness eval run --version vN` via `hf download`

## Reproduce any V

```bash
git checkout <git_tag_from_manifest>
harness eval run --version vN   # auto-downloads from FF-01/ HF repo
```

## Legacy

`versions/legacy/` — pre-V5 base-model comparison configs, old promptfoo v1..v8 iterations.

## Experiments

`experiments/` — cross-V experiments that don't belong to single V (e.g. base-model-comparison, promptfoo-iterations).

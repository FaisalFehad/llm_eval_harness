# V15 — Qwen3-4B Student Model

**Status**: COMPLETE (2026-04-13). Champion.
**Hybrid**: 99.6% (238/239, 1 error — Job 14 teacher NODE mis-label)
**Model-only label accuracy**: 91.4%
**Parse failures**: 17
**Base model**: `mlx-community/Qwen3-4B-bf16`
**Champion checkpoint**: `iter_700`
**Champion prompt**: `student.fix4.txt` (symlinked as `student.txt`)

## Reproduce

```bash
harness eval run --version v15             # auto-downloads adapter from HF (once wired)
./eval.sh run production                   # promptfoo via OMLX server
```

## Key files

| Path | Purpose |
|------|---------|
| `manifest.json` | Machine-readable config (base model, HF ref, metrics, git SHA) |
| `prompts/student.fix4.txt` | Champion prompt — fix4 variant |
| `prompts/student.{base,fix1,fix2,fix3,fix5,production}.txt` | Variants for A/B |
| `prompts/teacher.txt` | gpt-4.1-mini labeling prompt |
| `configs/lora.yaml` | LoRA training config (rank 16, alpha 32, LR 2e-5, 1400 steps) |
| `configs/promptfoo.yaml` | Master multi-prompt eval config with Langfuse tracing |
| `configs/promptfoo.single.yaml` | Single-prompt eval |
| `configs/scorer.py` `hooks.py` `tests.jsonl` `wrap_prompt.cjs` | Promptfoo scoring infra |
| `data/` | V15 training data (~757 train / 84 valid, OOS downsampled) |
| `scripts/` | Frozen V15 pipeline: train, eval, sweep, generate_data, merge_data, downsample_oos, mlflow_{eval,model}, quantize_oq |
| `adapters/` | LoRA checkpoints iter_100..iter_900 (binaries gitignored; HF: `FF-01/qwen3-4b-v15`) |
| `eval_results/` | sweep_4B, production (champion), fix{1..5}, master, oQ6, mlx6bit, gguf_Q6_K |
| `docs/PLAN.md` `docs/RUNBOOK.md` | V15-specific planning + execution notes |

## Variants registered

| Key | Model | Purpose |
|---|---|---|
| `v15` | `mlx-community/Qwen3-4B-bf16` + adapter iter 700 | Training/reference |
| `v15-oq6` | `~/MLX Models/qwen3_4B_v15_oQ6` | Mac deployment (merged MLX 6-bit, 3.1 GB) |
| `v15-mlx6` | `~/qwen3_4B_v15_mlx6bit` | Pre-oQ6 MLX 6-bit alternate |
| `v15-gguf` | `~/qwen3_4B_v15_Q6_K.gguf` | llama-cpp-python backend |

## Metric provenance (verified 2026-04-19)

- Hybrid 99.6%: `compute_hybrid_v13_1.py --v12` on `eval_results/production/adapters_v15_4B/2026-04-13_152613_*.predictions.jsonl`
- Model-only 91.4%: `label_accuracy` from same run's summary.json
- Sweep iter 700 (base prompt, not fix4): 99.2% / 86.2% — fix4 prompt adds +0.4pp hybrid / +5.2pp model-only

## Frozen lib

`lib/v15/` — frozen copies of `semantic_tokens.py`, `compute_hybrid.py`, `deterministic_baseline.py` at V15 release time. Per-V frozen per reorg spec (A3).

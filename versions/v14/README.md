# V14 — Qwen3 Multi-Size Reference

**Status**: Reference (superseded by V15).
**Champion (full precision)**: 4B step 800 = 98.7% hybrid / 86.2% model-only
**Mac deployment champion**: MLX 6-bit exp1 fix1 = 98.3% hybrid / 82.7% model-only (13 parse)

## Variants trained

| Size | Adapter path | Best ckpt | Hybrid | Model-only |
|---|---|---|---|---|
| 0.6B | `adapters/0.6B/` | checkpoint-2000 | 97.9% | 49.4% |
| 1.5B | `adapters/1.5B/` | checkpoint-1200 | 96.7% | 41.4% |
| 3.5_4B | `adapters/3.5_4B/` | — | — | — |
| 4B | `adapters/4B/` | **checkpoint-800** | **98.7%** | **86.2%** |

## Layout

```
v14/
  README.md            (this)
  manifest.json
  prompts/student.txt              base V14 prompt
  scripts/                         train, sweep, eval, eval_gguf, run_training.sh, requirements.txt
  configs/                         (lora configs — moved when located)
  data/                            train.jsonl (774), valid.jsonl (86)
  adapters/{0.6B,1.5B,3.5_4B,4B}/  LoRA checkpoints (binaries gitignored; HF: FF-01/qwen3-4b-v14)
  eval_results/
    sweep/{0.6B,1.5B,4B_fixed}/    checkpoint sweep outputs
    gguf/{f16,Q6_K,Q4_K_M,Q4_K_M_v3}/   GGUF inference results
    mlx6bit/                       MLX 6-bit baseline
  experiments/
    exp1/                          Mac MLX no-think experiment
      prompts/student.txt
      scripts/eval_no_think.py
      fix1/, fix_loop/, fixes/, no_think/
    exp2/                          Mac MLX thinking experiment (worse)
      prompts/student.{fix1,fix1b,fix2,fix3,fix4,fix5,no_raw}.txt
      scripts/eval_{think,thinking}.py
      no_raw/, think/, thinking/
    anyfix/                        abandoned
    comp_midpoint/                 salary midpoint micro-experiment
    targeted_issues/               targeted error-class tests
  docs/
    IMPLEMENTATION_PROGRESS.md
    REPRODUCTION_GUIDE.md
    EXPERIMENTS_PLAN.md
    lambda_logs/                   Lambda GH200 training + eval logs + imatrix.dat
```

## Reproduce

```bash
harness convert from-hf-adapter --version v14 \
  --adapter-dir versions/v14/adapters/4B/checkpoint-800 \
  --output-dir ~/merged_v14_4B \
  --mlx-out ~/qwen3_4B_v14_mlx6bit
harness eval run --version v14
```

## Key findings (pinned)

- `Q6_K` is minimum viable quantization for fine-tuned JSON schema. Q4_K_M hallucinates field names (loc_field) → 62.3% model-only floor.
- Thinking mode WORSE than no-think on both hybrid (96.7 vs 98.3) + model-only (68.6 vs 82.7). 95%+ model-only requires training-time fix, not prompting. → V15.
- `eval_student_v14_gguf.py` had bug: label-match compared to self at line 157. Model-only `correct=0` always zero in Q6_K/Q4_K_M hybrid_summary.json. Per-field accuracy correct.

## Metric provenance (verified 2026-04-19)

- Exp1 fix1: recomputed via `compute_hybrid_v13_1.py --v12` on `experiments/exp1/fix1/baseline/2026-03-20_002150_*.predictions.jsonl` → 98.3% hybrid (235/239, 4 errors, 13 parse).
- Step 800 4B: `eval_results/sweep/4B_fixed/sweep_summary.json` → 98.7%.
- Q6_K hybrid: `eval_results/gguf/Q6_K/hybrid_summary.json` → 98.3% v12_hybrid.

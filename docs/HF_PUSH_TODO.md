# HuggingFace Push Handoff

**Status**: DEFERRED — to run in standalone session.
**Auth**: `hf auth whoami` → `FaisalFehad` ✓ (verified 2026-04-19).
**Tag before push**: `pre-reorg-2026-04-19`.

## Scope

Push all custom-trained adapters + merged HF + GGUF variants per version, all sweep checkpoints, to `FaisalFehad/` org on HuggingFace. Source of truth for local auto-download in `harness eval run --version vN`.

## Target repo layout (per-V)

```
FaisalFehad/qwen3-4b-v15/                 # one repo per version
  checkpoints/
    iter_100/<adapter-files>
    iter_200/...
    iter_700/          # champion
    iter_900/
  merged/              # full-precision HF model (after LoRA merge)
  gguf/
    f16.gguf
    Q6_K.gguf
    Q4_K_M.gguf
  mlx/
    mlx-4bit/
    mlx-6bit/
  README.md            # auto-render or write manually
```

## Per-V inventory

| V | Base | Adapter size | Merged | GGUF | MLX | Sweep ckpts | Est. total |
|---|---|---|---|---|---|---|---|
| v13 | Qwen3-0.6B-4bit | 335 MB | — | — | — | 9 (1500–1900 + extras) | ~360 MB |
| v13_1 | Qwen2.5-1.5B-Instruct-4bit | 839 MB | — | — | — | 10 (200–2000) | ~1.3 GB |
| v13_1 (0.6B corrective) | Qwen3-0.6B-4bit | 335 MB | — | — | — | 8 (50–400) | ~280 MB |
| v14 (4B) | Qwen3-4B | ~450 MB | 7.5 GB (`fused_model/` / `~/merged_v14_4B/`) | F16 7.5 GB + Q6_K 3.1 GB + Q4_K_M 2.3 GB | 6bit 3.1 GB | ~15 | **~25 GB** |
| v14 (1.5B) | Qwen2.5-1.5B | 839 MB | — | — | — | ? | ~1.5 GB |
| v14 (0.6B) | Qwen3-0.6B | 335 MB | — | — | — | ? | ~500 MB |
| v15 (4B) | mlx-community/Qwen3-4B-bf16 | 560 MB (adapter dir) | — (to create) | oQ6 3.1 GB (exists `eval_results/v15_oQ6/`?) | 6bit 3.1 GB | 9 (100–900) | ~5 GB |

**Est. total ~35 GB.** Check HF Pro quota before push.

## Per-V repo checklist

- [ ] `FaisalFehad/qwen3-0.6b-v13` — adapters only (no merged/gguf done)
- [ ] `FaisalFehad/qwen2.5-1.5b-v13.1` — adapters only
- [ ] `FaisalFehad/qwen3-0.6b-v13.1` — corrective adapters
- [ ] `FaisalFehad/qwen3-4b-v14` — **exists** (has GGUFs + merged_v14_4B). Add: all sweep ckpts + MLX 6bit. Restructure to `checkpoints/iter_N/` scheme.
- [ ] `FaisalFehad/qwen2.5-1.5b-v14`
- [ ] `FaisalFehad/qwen3-0.6b-v14`
- [ ] `FaisalFehad/qwen3-4b-v15` — **new**. All sweep ckpts + merged (need to build) + MLX 6bit + GGUFs (need to build).

## Push script skeleton

```python
# scripts/hf_push.py (to write)
from huggingface_hub import HfApi, create_repo, upload_folder
api = HfApi()
for v, spec in VERSIONS.items():
    repo_id = spec["hf_repo"]
    create_repo(repo_id, private=True, exist_ok=True)
    for ckpt in spec["checkpoints"]:
        upload_folder(
            folder_path=ckpt["local_path"],
            path_in_repo=f"checkpoints/iter_{ckpt['iter']}",
            repo_id=repo_id,
        )
    # merged, gguf, mlx similarly
```

## Pre-flight

1. `hf auth whoami` → confirm FaisalFehad
2. Check HF Pro plan: `hf repos ls --format json | grep storage` — verify quota >50 GB private
3. Dry-run one ckpt: `hf upload --dry-run FaisalFehad/qwen3-4b-v15 finetune/adapters_v15_4B/adapter_config.json`
4. Real run per-V, background + log to `logs/hf_push_vN.log`
5. After push: verify `hf repos info FaisalFehad/qwen3-4b-v15` shows expected tree
6. Write `hf_repo: FaisalFehad/qwen3-4b-v15` + `checkpoints: [iter_100, ...]` into `versions/v15/manifest.json`

## Only after push confirmed

- Delete local `fused_model/` (7.5 GB)
- Delete local `finetune/adapters_v*/` binaries (~36 GB)
- Wire auto-download in `finetune/registry.py` via `huggingface_hub.snapshot_download(repo_id, revision=f"checkpoints/iter_{iter}", cache_dir="./models/adapters")`

## Rollback

If push fails mid-way: local state intact. Just restart from last successful V.
If push verified but then user wants local back: `hf download FaisalFehad/qwen3-4b-v15 --local-dir ./models/adapters/v15`.

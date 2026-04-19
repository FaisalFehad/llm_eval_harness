#!/usr/bin/env python3
"""
HF push for a single V's artifacts.

Layout per repo (FF-01/qwen3-{size}-{v}):
  checkpoints/iter_{N}/<adapter-files>
  merged/<hf-files>
  gguf/{f16,Q6_K,Q4_K_M}.gguf
  mlx/{4bit,6bit,oQ6}/

Usage:
  HARNESS_ALLOW_HF_PUSH=1 .venv/bin/python3 scripts/hf_push.py v15
  HARNESS_ALLOW_HF_PUSH=1 .venv/bin/python3 scripts/hf_push.py v15 --only adapters
  HARNESS_ALLOW_HF_PUSH=1 .venv/bin/python3 scripts/hf_push.py v13
"""
from __future__ import annotations
import argparse
import os
import re
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
except ImportError:
    print("✗ huggingface_hub not installed", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()

# ── Per-V artifact registry ─────────────────────────────────────────
SPECS = {
    "v13": {
        "hf_repo": "FF-01/qwen3-0.6b-v13",
        "adapters_dir": REPO_ROOT / "versions/v13/adapters",
        "merged": None,
        "gguf": [],
        "mlx": [],
    },
    "v13_1_1.5B": {
        "hf_repo": "FF-01/qwen2.5-1.5b-v13.1",
        "adapters_dir": REPO_ROOT / "versions/v13_1/adapters/1.5B",
        "merged": None,
        "gguf": [],
        "mlx": [],
    },
    "v13_1_0.6B": {
        "hf_repo": "FF-01/qwen3-0.6b-v13.1",
        "adapters_dir": REPO_ROOT / "versions/v13_1/adapters/0.6B",
        "merged": None,
        "gguf": [],
        "mlx": [],
    },
    "v14": {
        "hf_repo": "FF-01/qwen3-4b-v14",  # exists — augment
        "adapters_dir": REPO_ROOT / "versions/v14/adapters/4B",
        "merged": REPO_ROOT / "fused_model",  # 7.5 GB merged HF
        "gguf": [],
        "mlx": [HOME / "qwen3_4B_v14_mlx6bit"],
    },
    "v15": {
        "hf_repo": "FF-01/qwen3-4b-v15",
        "adapters_dir": REPO_ROOT / "versions/v15/adapters",
        "merged": HOME / "merged_v15_4B",
        "gguf": [HOME / "qwen3_4B_v15_f16.gguf", HOME / "qwen3_4B_v15_Q6_K.gguf"],
        "mlx": [
            HOME / "qwen3_4B_v15_mlx4bit",
            HOME / "qwen3_4B_v15_mlx6bit",
            HOME / "MLX Models" / "qwen3_4B_v15_oQ6",
        ],
    },
}


def push_adapters(api: HfApi, repo: str, adapters_dir: Path, dry_run: bool = False) -> None:
    """Push each iter_NNN_adapters.safetensors as checkpoints/iter_N/."""
    if not adapters_dir.exists():
        print(f"  ✗ adapters_dir missing: {adapters_dir}")
        return
    ckpts = sorted(adapters_dir.glob("*_adapters.safetensors"))
    print(f"  Found {len(ckpts)} checkpoints")
    cfg = adapters_dir / "adapter_config.json"

    for ckpt in ckpts:
        m = re.match(r"(\d+)_adapters\.safetensors", ckpt.name)
        if not m:
            continue
        iter_n = int(m.group(1))
        path_in_repo = f"checkpoints/iter_{iter_n}"
        print(f"    ↑ iter_{iter_n} ({ckpt.stat().st_size / 1e9:.2f} GB)")
        if dry_run:
            continue
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=f"{path_in_repo}/adapters.safetensors",
            repo_id=repo,
        )
        if cfg.exists():
            api.upload_file(
                path_or_fileobj=str(cfg),
                path_in_repo=f"{path_in_repo}/adapter_config.json",
                repo_id=repo,
            )


def push_dir(api: HfApi, repo: str, local: Path, prefix: str, dry_run: bool) -> None:
    if not local.exists():
        print(f"  ✗ skip (missing): {local}")
        return
    sz = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e9
    print(f"  ↑ {local.name} ({sz:.2f} GB) → {prefix}/")
    if dry_run:
        return
    api.upload_folder(
        folder_path=str(local),
        path_in_repo=prefix,
        repo_id=repo,
        ignore_patterns=["*.partial", ".DS_Store", "__pycache__/*"],
    )


def push_gguf(api: HfApi, repo: str, gguf_path: Path, dry_run: bool) -> None:
    if not gguf_path.exists():
        print(f"  ✗ skip (missing): {gguf_path}")
        return
    # Parse variant suffix: qwen3_4B_v15_Q6_K.gguf → Q6_K
    m = re.match(r".+_v\d+(?:_1)?_(.+)\.gguf$", gguf_path.name)
    variant = m.group(1) if m else gguf_path.stem
    target = f"gguf/{variant}.gguf"
    sz = gguf_path.stat().st_size / 1e9
    print(f"  ↑ {gguf_path.name} ({sz:.2f} GB) → {target}")
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=target,
        repo_id=repo,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", choices=list(SPECS))
    ap.add_argument("--only", choices=["adapters", "merged", "gguf", "mlx"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and os.environ.get("HARNESS_ALLOW_HF_PUSH") != "1":
        print("✗ Safety gate: set HARNESS_ALLOW_HF_PUSH=1 to enable real upload", file=sys.stderr)
        sys.exit(1)

    spec = SPECS[args.version]
    repo = spec["hf_repo"]
    print(f"=== {args.version} → {repo} ===")

    api = HfApi()
    if not args.dry_run:
        create_repo(repo, repo_type="model", private=True, exist_ok=True)

    if args.only in (None, "adapters"):
        print("[adapters]")
        push_adapters(api, repo, spec["adapters_dir"], args.dry_run)

    if args.only in (None, "merged") and spec.get("merged"):
        print("[merged]")
        push_dir(api, repo, spec["merged"], "merged", args.dry_run)

    if args.only in (None, "gguf"):
        print("[gguf]")
        for g in spec.get("gguf", []):
            push_gguf(api, repo, g, args.dry_run)

    if args.only in (None, "mlx"):
        print("[mlx]")
        for m in spec.get("mlx", []):
            # Variant = basename (e.g. qwen3_4B_v15_mlx6bit → mlx6bit, oQ6 → oQ6)
            variant = m.name.replace(f"qwen3_4B_{args.version}_", "").replace(f"qwen3_4B_v15_", "")
            push_dir(api, repo, m, f"mlx/{variant}", args.dry_run)

    print(f"=== done: {args.version} ===")


if __name__ == "__main__":
    main()

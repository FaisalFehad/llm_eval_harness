"""
Lazy HuggingFace download helper.

When a registry path points to an adapter/model that doesn't exist locally,
resolve it via `hf download` on first use, caching to `./models/`.

Gated by HARNESS_ALLOW_DOWNLOAD=1 env var to avoid accidental 10-GB pulls.

Status: stub. Activates once `docs/HF_PUSH_TODO.md` migration complete.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"


def resolve_adapter(
    local_path: str,
    hf_repo: Optional[str] = None,
    hf_revision: Optional[str] = None,
) -> Path:
    """
    Return Path to adapter; download from HF if missing + allowed.

    - local_path: canonical path, e.g. "versions/v15/adapters/0000700_adapters.safetensors"
    - hf_repo:     e.g. "FF-01/qwen3-4b-v15"
    - hf_revision: e.g. "checkpoints/iter_700" (subdir) or git revision

    Behavior:
      1. If local file exists → return Path.
      2. If HARNESS_ALLOW_DOWNLOAD != 1 → raise with hint.
      3. If hf_repo provided → snapshot_download → symlink into local_path → return.
    """
    p = REPO_ROOT / local_path
    if p.exists():
        return p

    if os.environ.get("HARNESS_ALLOW_DOWNLOAD") != "1":
        print(
            f"✗ Adapter missing: {local_path}\n"
            f"  HF source: {hf_repo or '(not set — see docs/HF_PUSH_TODO.md)'}\n"
            f"  Set HARNESS_ALLOW_DOWNLOAD=1 and re-run to auto-download.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hf_repo:
        print(
            f"✗ No hf_repo registered for {local_path}. Add to registry.py "
            f"after HF push (see docs/HF_PUSH_TODO.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("✗ huggingface_hub not installed. `pip install huggingface_hub`", file=sys.stderr)
        sys.exit(1)

    cache = MODELS_DIR / "adapters" / hf_repo.replace("/", "--")
    cache.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if hf_revision and hf_revision.startswith("checkpoints/"):
        allow_patterns = [f"{hf_revision}/*"]

    print(f"↓ Downloading {hf_repo} rev={hf_revision} → {cache}", file=sys.stderr)
    snapshot_download(
        repo_id=hf_repo,
        revision=hf_revision if hf_revision and not hf_revision.startswith("checkpoints/") else None,
        local_dir=str(cache),
        allow_patterns=allow_patterns,
    )

    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        resolved = cache / (hf_revision or "") / p.name
        if resolved.exists():
            p.symlink_to(resolved.resolve())
        else:
            print(f"✗ Download succeeded but {p} still missing — check repo layout", file=sys.stderr)
            sys.exit(1)
    return p

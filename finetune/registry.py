"""
Central pipeline registry — single source of truth for version defaults.

Every CLI verb reads from here instead of hardcoding v15 paths. Adding a new
pipeline means editing ONE file (this one), not eight.

Default resolution:
    1. If --version is passed explicitly, use it.
    2. Otherwise, read $HARNESS_VERSION from the environment.
    3. Otherwise, fall back to DEFAULT_VERSION below.

Variants (e.g. v15-oq6, v15-gguf) share training scripts with their base
via the `base_version` field. Core pipeline config (model/adapter/prompt/
backend) is always per-variant.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_VERSION = "v15"
ENV_VERSION = "HARNESS_VERSION"


@dataclass(frozen=True)
class Pipeline:
    """Canonical config for a trained-model pipeline.

    Attributes set only once you actually have them — `None` means "not
    applicable for this version". For example, v15-oq6 is a merged model
    so it has no adapter and no lora_config.
    """
    # ── Inference (read by `eval`, `promptfoo`) ─────────────────
    model: str                                  # HF alias or local path
    prompt: str                                 # Student prompt .txt path
    adapter: Optional[str] = None               # LoRA adapter (None for merged)
    backend: str = "mlx"                        # mlx | gguf | hf

    # ── Eval defaults ───────────────────────────────────────────
    test_file: str = "data/v12/test_labeled_audited.jsonl"
    regex_version: str = "v13_1"                # For hybrid scoring

    # ── Training / sweep / conversion ───────────────────────────
    base_version: Optional[str] = None          # Script dispatch: use this key. None = self.
    sweep_script: Optional[str] = None          # finetune/sweep_*.py
    lora_config: Optional[str] = None           # finetune/lora_config_*.yaml
    base_model_hf: Optional[str] = None         # HF base ID for `convert from-hf-adapter`

    # ── Display ─────────────────────────────────────────────────
    description: str = ""


PIPELINES: dict[str, Pipeline] = {
    # ── V15 — peak accuracy (bf16 + iter 700 + fix4) ────────────
    "v15": Pipeline(
        model="mlx-community/Qwen3-4B-bf16",
        adapter="finetune/adapters_v15_4B/0000700_adapters.safetensors",
        prompt="prompts/student_v15_fix4.txt",
        sweep_script="finetune/sweep_v15.py",
        lora_config="finetune/lora_config_v15_4B.yaml",
        base_model_hf="mlx-community/Qwen3-4B-bf16",
        description="V15 peak — 91.4% model-only / 99.6% hybrid (bf16 + iter 700 + fix4)",
    ),

    # ── V15 deployment variants (share v15's scripts) ────────────
    "v15-oq6": Pipeline(
        model="~/MLX Models/qwen3_4B_v15_oQ6",
        prompt="prompts/student_v15_fix4.txt",
        base_version="v15",
        description="V15 Mac deployment — MLX 6-bit, merged, ~3.1 GB",
    ),
    "v15-mlx6": Pipeline(
        model="~/qwen3_4B_v15_mlx6bit",
        prompt="prompts/student_v15_fix4.txt",
        base_version="v15",
        description="V15 MLX 6-bit alternate (pre-oQ6 build)",
    ),
    "v15-gguf": Pipeline(
        model="~/qwen3_4B_v15_Q6_K.gguf",
        prompt="prompts/student_v15_fix4.txt",
        backend="gguf",
        base_version="v15",
        description="V15 GGUF Q6_K for llama-cpp-python",
    ),

    # ── V14 — needs rehydration (use `harness convert from-hf-adapter`) ─
    "v14": Pipeline(
        model="~/qwen3_4B_v14_mlx6bit",
        prompt="prompts/student_v14_exp1.txt",
        sweep_script="finetune/sweep_v14.py",
        base_model_hf="Qwen/Qwen3-4B-Instruct-2507",
        description="V14 4B (98.7% hybrid) — model MISSING, rehydrate via `harness convert from-hf-adapter --version v14`",
    ),

    # ── V13.1 — 1.5B reference ──────────────────────────────────
    "v13_1": Pipeline(
        model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter="finetune/adapters_v13_1_1.5B/0001800_adapters.safetensors",
        prompt="prompts/student_v13_1.txt",
        sweep_script="finetune/sweep_v13_1_1.5B.py",
        lora_config="finetune/lora_config_v13_1_1.5B.yaml",
        description="V13.1 1.5B reference — 97.5% hybrid (iter 1800)",
    ),

    # ── V13 — 0.6B production (lowest latency) ──────────────────
    "v13": Pipeline(
        model="mlx-community/Qwen3-0.6B-4bit",
        adapter="finetune/adapters_v13_0.6B/0001500_adapters.safetensors",
        prompt="prompts/student_v13.txt",
        sweep_script="finetune/sweep_v13.py",
        lora_config="finetune/lora_config_v13_0.6B.yaml",
        description="V13 0.6B production — 97.9% hybrid (iter 1500, fastest)",
    ),
}


# ────────────────────────────────────────────────────────────────
# Helpers — every command should use these instead of reading PIPELINES directly
# ────────────────────────────────────────────────────────────────

def default_version() -> str:
    """Return the current default pipeline key ($HARNESS_VERSION or DEFAULT_VERSION)."""
    return os.environ.get(ENV_VERSION, DEFAULT_VERSION)


def get(version: Optional[str] = None) -> Pipeline:
    """Fetch a pipeline. Uses default_version() when None. Raises KeyError with a helpful message."""
    key = version or default_version()
    if key not in PIPELINES:
        raise KeyError(
            f"Unknown version '{key}'. Registered: {sorted(PIPELINES)}. "
            f"Change the default via ${ENV_VERSION} or add an entry to finetune/registry.py."
        )
    return PIPELINES[key]


def base_key(version: Optional[str] = None) -> str:
    """Resolve the key used for script-filename dispatch (handles variants like v15-oq6 → v15)."""
    p = get(version)
    return p.base_version or (version or default_version())


def list_versions() -> list[str]:
    return list(PIPELINES.keys())

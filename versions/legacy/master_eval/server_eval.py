#!/usr/bin/env python3
"""
master_eval/server_eval.py

Evaluates ALL models on the audited V12 test set using single-prompt generation.

Architecture:
  1. Single-prompt mlx_lm.generate() — matches eval_student_v7.py exactly
  2. One model load per adapter (not 16 subprocess loads)
  3. In-process hybrid scoring with 3 cached regex versions (zero repeated work)
  4. GGUF models via subprocess (llama-cpp-python)

Models run sequentially (one at a time, full GPU bandwidth).
SAFE: writes only to master_eval/results/ and master_eval/adapters/.

Usage:
    .venv/bin/python3 master_eval/server_eval.py [OPTIONS]

Options:
    --only NAME ...  Run only these model names
    --skip NAME ...  Skip these model names
    --reuse          Load cached hybrid_summary.json instead of re-running

Examples:
    .venv/bin/python3 master_eval/server_eval.py                    # all 10 models
    .venv/bin/python3 master_eval/server_eval.py --only v12_1_1.5B  # single model
    .venv/bin/python3 master_eval/server_eval.py --reuse            # report from cache
"""

import argparse
import datetime
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Project imports ──────────────────────────────────────────────────────────

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "finetune"))
from semantic_tokens_v7 import validate_prediction, compute_from_tokens

# All 3 regex versions — same interface, different rules
from deterministic_baseline import classify_job as regex_v12_classify
from deterministic_baseline_v13 import classify_job as regex_v13_classify
from deterministic_baseline_v13_1 import classify_job as regex_v13_1_classify

REGEX_VERSIONS = {
    "v12":   regex_v12_classify,
    "v13":   regex_v13_classify,
    "v13.1": regex_v13_1_classify,  # current best
}

# ── Paths ────────────────────────────────────────────────────────────────────

TEST_FILE   = REPO / "versions/v12/data/v12_original/test_labeled_audited.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
ADAPTERS_DIR = Path(__file__).parent / "adapters"


# ── Model registry ──────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    display: str
    model: str          # HF ID or local path (~ supported)
    prompt: str         # path relative to REPO
    size_gb: float
    adapter: str = ""   # path relative to REPO; empty = standalone
    no_think: bool = False
    max_tokens: int = 1500  # safety cap — model stops at EOS well before this
    is_gguf: bool = False
    expected: str = ""
    notes: str = ""


def _res(path: str) -> str:
    """Expand ~, resolve repo-relative paths, or pass through HF model IDs."""
    if not path:
        return ""
    if path.startswith("~"):
        return str(Path(path).expanduser())
    repo_path = REPO / path
    if repo_path.exists():
        return str(repo_path)
    return path  # pass through as HF model ID


MODELS: list[ModelConfig] = [
    # Ordered by base model to maximize HF cache hits between consecutive loads.
    # Same base model = disk cache hit on reload (saves ~3-10s per load).

    # ── Qwen2.5-0.5B (1 adapter) ─────────────────────────────────────────────
    ModelConfig(
        name="v7_0.5B", display="V7 0.5B — Qwen2.5-0.5B",
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        adapter="finetune/adapters_v7",
        prompt="prompts/student_v7.txt", size_gb=0.25,
        notes="Pre-V12 baseline, original Qwen2.5 0.5B",
    ),

    # ── Qwen2.5-1.5B (3 adapters) ────────────────────────────────────────────
    ModelConfig(
        name="v7_1.5B", display="V7 1.5B — Qwen2.5-1.5B",
        model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter="finetune/adapters_v7_1.5B",
        prompt="prompts/student_v7.txt", size_gb=0.84,
        notes="Pre-V12 baseline 1.5B",
    ),
    ModelConfig(
        name="v12_1_1.5B", display="V12.1 1.5B — Qwen2.5-1.5B  iter 2000",
        model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter="finetune/adapters_v12/0002000_adapters.safetensors",
        prompt="prompts/student_v7.txt", size_gb=0.84,
        expected="98.3%",
        notes="V12.1 1.5B best checkpoint iter 2000",
    ),
    ModelConfig(
        name="v13_1_1.5B", display="V13.1 1.5B — Qwen2.5-1.5B  iter 1800",
        model="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter="finetune/adapters_v13_1_1.5B/0001800_adapters.safetensors",
        prompt="prompts/student_v13_1.txt", size_gb=0.84,
        expected="97.5%",
        notes="V13.1 1.5B fresh train, best iter 1800",
    ),

    # ── Qwen3-0.6B (3 adapters) ──────────────────────────────────────────────
    ModelConfig(
        name="v12_0.6B", display="V12 0.6B — Qwen3-0.6B  iter 1400",
        model="mlx-community/Qwen3-0.6B-4bit",
        adapter="finetune/adapters_v12_qwen3_0.6B/0001400_adapters.safetensors",
        prompt="prompts/student_v7.txt", size_gb=0.335,
        no_think=True, expected="96.7%",
        notes="V12 Qwen3 0.6B, best checkpoint iter 1400",
    ),
    ModelConfig(
        name="v13_0.6B", display="V13 0.6B — Qwen3-0.6B  iter 1500",
        model="mlx-community/Qwen3-0.6B-4bit",
        adapter="finetune/adapters_v13_0.6B/0001500_adapters.safetensors",
        prompt="prompts/student_v13.txt", size_gb=0.335,
        no_think=True, expected="97.9%",
        notes="V13 production 0.6B, best iter 1500",
    ),
    ModelConfig(
        name="v13_1_0.6B", display="V13.1 0.6B corrective — Qwen3-0.6B  iter 200",
        model="mlx-community/Qwen3-0.6B-4bit",
        adapter="finetune/adapters_v13_1_0.6B/0000200_adapters.safetensors",
        prompt="prompts/student_v13_1.txt", size_gb=0.335,
        no_think=True, expected="97.5%",
        notes="V13.1 0.6B corrective retrain, best iter 200",
    ),

    # ── Qwen3-4B — MLX variants ────────────────────────────────────────────
    # IMPORTANT: HF bfloat16 was trained with no-think → student_v14.txt
    #            MLX 6-bit uses exp1 (no-think) or exp2_fix3 (thinking ON)
    #            Never mix: thinking prompt + no_think=True or vice versa
    ModelConfig(
        name="v14_4B_hf", display="V14 4B HF bfloat16 — golden reference",
        model="~/merged_v14_4B",
        prompt="prompts/student_v14.txt", size_gb=8.0,
        no_think=True, expected="98.7%",
        notes="V14 4B full precision, step 800, training prompt, no-think",
    ),
    ModelConfig(
        name="v14_4B_mlx6bit", display="V14 4B MLX 6-bit — exp1 fix1  no-think",
        model="~/qwen3_4B_v14_mlx6bit",
        prompt="prompts/student_v14_exp1.txt", size_gb=3.1,
        no_think=True, expected="98.3%",
        notes="V14 4B MLX 6-bit, exp1 fix1, no-think",
    ),
    ModelConfig(
        name="v14_4B_mlx6bit_think", display="V14 4B MLX 6-bit — exp2 fix3  thinking ON",
        model="~/qwen3_4B_v14_mlx6bit",
        prompt="prompts/student_v14_exp2_fix3.txt", size_gb=3.1,
        no_think=False, max_tokens=3000, expected="96.7%",
        notes="V14 4B MLX 6-bit, thinking ON ceiling — WORSE than no-think",
    ),

    # ── Qwen3-4B — GGUF variants ─────────────────────────────────────────
    # All GGUFs use student_v14.txt. Thinking disabled via "/no_think" in the
    # system prompt inside eval_student_v14_gguf.py (NOT via no_think field).
    # We set no_think=True here for audit consistency only — it has no effect
    # on the GGUF subprocess path.
    ModelConfig(
        name="v14_f16_gguf", display="V14 4B F16 GGUF — full precision",
        model="~/qwen3_4B_v14_f16.gguf",
        prompt="prompts/student_v14.txt", size_gb=6.3,
        is_gguf=True, no_think=True, max_tokens=1000, expected="98.7%",
        notes="V14 GGUF F16, should match HF bfloat16 accuracy",
    ),
    ModelConfig(
        name="v14_Q6_K", display="V14 4B Q6_K GGUF",
        model="~/qwen3_4B_v14_Q6_K.gguf",
        prompt="prompts/student_v14.txt", size_gb=3.1,
        is_gguf=True, no_think=True, max_tokens=1000, expected="98.3%",
        notes="V14 GGUF Q6_K, min viable quantisation for fine-tuned JSON schema",
    ),
    ModelConfig(
        name="v14_Q4_K_M", display="V14 4B Q4_K_M GGUF",
        model="~/qwen3_4B_v14_Q4_K_M.gguf",
        prompt="prompts/student_v14.txt", size_gb=2.3,
        is_gguf=True, no_think=True, max_tokens=1000, expected="97.9%",
        notes="V14 GGUF Q4_K_M, schema hallucination at 4-bit (low model-only)",
    ),
    ModelConfig(
        name="v14_Q2_K", display="V14 4B Q2_K GGUF — broken",
        model="~/qwen3_4B_v14_Q2_K.gguf",
        prompt="prompts/student_v14.txt", size_gb=1.6,
        is_gguf=True, no_think=True, max_tokens=1000, expected="broken",
        notes="V14 GGUF Q2_K, fine-tuning lost below ~4-bit — expect garbage",
    ),
    ModelConfig(
        name="v14_IQ2_XXS", display="V14 4B IQ2_XXS GGUF — broken",
        model="~/qwen3_4B_v14_IQ2_XXS.gguf",
        prompt="prompts/student_v14.txt", size_gb=1.2,
        is_gguf=True, no_think=True, max_tokens=1000, expected="broken",
        notes="V14 GGUF IQ2_XXS, extreme quantization — completely broken",
    ),
]


# ── HuggingFace auto-download ────────────────────────────────────────────────
# Adapters: FF-01/eval-harness-adapters/{name}/adapters.safetensors
# V14 GGUFs: FF-01/qwen3-4b-v14/{filename}.gguf
# V14 MLX 6-bit: FF-01/qwen3-4b-v14/mlx_6bit/
# V14 HF bfloat16: FF-01/qwen3-4b-v14/merged_v14_4B/

HF_ADAPTER_REPO = "FF-01/eval-harness-adapters"
HF_V14_REPO = "FF-01/qwen3-4b-v14"

# Map model name → HF download info for adapters
HF_ADAPTER_MAP = {
    "v7_0.5B":     "v7_0.5B",
    "v7_1.5B":     "v7_1.5B",
    "v12_0.6B":    "v12_0.6B",
    "v12_1_1.5B":  "v12_1_1.5B",
    "v13_0.6B":    "v13_0.6B",
    "v13_1_0.6B":  "v13_1_0.6B",
    "v13_1_1.5B":  "v13_1_1.5B",
}

# Map model name → HF download info for local models
HF_MODEL_MAP = {
    "v14_4B_hf":           ("merged_v14_4B", "~/merged_v14_4B"),
    "v14_4B_mlx6bit":      ("mlx_6bit", "~/qwen3_4B_v14_mlx6bit"),
    "v14_4B_mlx6bit_think":("mlx_6bit", "~/qwen3_4B_v14_mlx6bit"),
    "v14_f16_gguf":        ("qwen3_4B_v14_f16.gguf", "~/qwen3_4B_v14_f16.gguf"),
    "v14_Q6_K":            ("qwen3_4B_v14_Q6_K.gguf", "~/qwen3_4B_v14_Q6_K.gguf"),
    "v14_Q4_K_M":          ("qwen3_4B_v14_Q4_K_M.gguf", "~/qwen3_4B_v14_Q4_K_M.gguf"),
    "v14_Q2_K":            ("qwen3_4B_v14_Q2_K.gguf", "~/qwen3_4B_v14_Q2_K.gguf"),
    "v14_IQ2_XXS":         ("qwen3_4B_v14_IQ2_XXS.gguf", "~/qwen3_4B_v14_IQ2_XXS.gguf"),
}


def _download_adapter(m: ModelConfig) -> bool:
    """Download adapter from HF if missing locally. Returns True if available after."""
    hf_name = HF_ADAPTER_MAP.get(m.name)
    if not hf_name:
        return False

    ap = Path(_res(m.adapter))
    # For .safetensors file paths, check the parent dir for adapters.safetensors
    if ap.suffix == ".safetensors":
        target_dir = ap.parent
        target_file = target_dir / "adapters.safetensors"
    else:
        target_dir = ap
        target_file = ap / "adapters.safetensors"

    if target_file.exists():
        return True

    try:
        from huggingface_hub import hf_hub_download
        print(f"  [DOWNLOAD] {m.name} adapter from {HF_ADAPTER_REPO}...", end="", flush=True)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Download adapters.safetensors
        hf_hub_download(HF_ADAPTER_REPO, f"{hf_name}/adapters.safetensors",
                        local_dir=str(target_dir.parent),
                        local_dir_use_symlinks=False)
        # Download adapter_config.json
        try:
            hf_hub_download(HF_ADAPTER_REPO, f"{hf_name}/adapter_config.json",
                            local_dir=str(target_dir.parent),
                            local_dir_use_symlinks=False)
        except Exception:
            pass  # config is optional

        print(f" done")
        return target_file.exists() or (target_dir / hf_name / "adapters.safetensors").exists()
    except Exception as e:
        print(f" failed: {e}")
        return False


def _download_model(m: ModelConfig) -> bool:
    """Download V14 model from HF if missing locally. Returns True if available after."""
    info = HF_MODEL_MAP.get(m.name)
    if not info:
        return False

    hf_path, local_path = info
    local = Path(local_path).expanduser()

    if local.exists():
        return True

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        print(f"  [DOWNLOAD] {m.name} from {HF_V14_REPO}/{hf_path}...", flush=True)

        if hf_path.endswith(".gguf"):
            # Single file download
            hf_hub_download(HF_V14_REPO, hf_path,
                            local_dir=str(local.parent),
                            local_dir_use_symlinks=False)
        else:
            # Directory download (merged_v14_4B or mlx_6bit)
            snapshot_download(HF_V14_REPO, allow_patterns=f"{hf_path}/*",
                              local_dir=str(local.parent),
                              local_dir_use_symlinks=False)
            # snapshot_download puts files in local.parent/hf_path/
            # Rename to expected path if needed
            downloaded = local.parent / hf_path
            if downloaded.exists() and not local.exists():
                downloaded.rename(local)

        print(f"  [DOWNLOAD] {m.name} complete ({local})")
        return local.exists()
    except Exception as e:
        print(f"  [DOWNLOAD] {m.name} failed: {e}")
        return False


def check_available(m: ModelConfig) -> tuple[bool, str]:
    """Check if model files exist locally. Auto-downloads from HF if missing."""
    # Check prompt first (always local, never downloaded)
    if not (REPO / m.prompt).exists():
        return False, f"prompt missing: {m.prompt}"

    # Check adapter — download if missing
    if m.adapter:
        ap = Path(_res(m.adapter))
        if not ap.exists():
            if not _download_adapter(m):
                return False, f"adapter missing: {ap}"

    # Check local model — download if missing
    if m.model.startswith("~"):
        mp = Path(m.model).expanduser()
        if not mp.exists():
            if not _download_model(m):
                return False, f"model missing: {mp}"

    return True, "ok"


# ── JSON parsing (same logic as eval_student_v7.py) ─────────────────────────

def parse_json_output(text: str) -> dict | None:
    """Extract JSON from model output, handling truncation and formatting issues."""
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Fix unquoted keys
    text = re.sub(r'\{(\s*)(\w+)":', r'{"\2":', text)
    text = re.sub(r'\{(\s*)(\w+):', r'{"\2":', text)
    text = re.sub(r',(\s*)(\w+):', r',"\2":', text)

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: regex extraction of complete JSON
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 3: auto-fix truncated JSON (common with small token budgets)
    match = re.search(r'\{[\s\S]*', text)
    if match:
        truncated = match.group()
        if truncated.count('"') % 2 != 0:
            truncated += '"'
        if '"tech"' in truncated and '[' in truncated.split('"tech"')[-1]:
            after_tech = truncated.split('"tech"')[-1]
            if after_tech.count('[') > after_tech.count(']'):
                truncated += ']'
        truncated += '}'
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    return None


# ── Prompt construction ──────────────────────────────────────────────────────

def build_prompt(tokenizer, job: dict, prompt_template: str, model_path: str,
                 no_think: bool, system_msg: str = "Respond with JSON only.") -> str:
    """
    Build a formatted prompt string for one job.

    Replicates eval_student_v7.py lines 339-372 exactly:
      1. Fill template placeholders (title, location, JD)
      2. Wrap as chat messages (system + user)
      3. Apply chat template (with enable_thinking=False for Qwen3 no-think)
      4. Pre-fill opening '{' for JSON guidance

    Returns a string ready for mlx_lm.generate().
    """
    is_qwen3 = "qwen3" in model_path.lower()

    raw_location = job.get("job_location", job.get("location", ""))
    jd_text = job.get("jd_text", job.get("description", ""))[:6000]

    prompt_text = (prompt_template
        .replace("{{job_title}}", job["title"])
        .replace("{{job_location}}", raw_location)
        .replace("{{jd_text}}", jd_text))

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]

    chat_kwargs = {"enable_thinking": False} if (is_qwen3 and no_think) else {}
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs)

    # Pre-fill opening brace ONLY if the prompt doesn't already instruct it.
    # V14 prompts say "Begin your response with {" — pre-filling causes "{{" (invalid).
    # V7/V13 prompts don't — pre-fill is needed to guide JSON output.
    prompt_has_begin_brace = "Begin" in prompt_template and "{" in prompt_template.split("Begin")[-1][:30]
    if not prompt_has_begin_brace and (not is_qwen3 or no_think):
        formatted += "{"

    return formatted, prompt_has_begin_brace


# ── Build prediction record ──────────────────────────────────────────────────

def build_prediction(job_index: int, job: dict, parsed: dict | None,
                     raw_output: str) -> dict:
    """
    Build a prediction dict compatible with compute_hybrid_v13_1.py.

    Required fields: job_index, parse_fail, pred_tokens, label_match.
    Mirrors eval_student_v7.py's scoring logic.
    """
    if parsed is None:
        return {
            "job_index": job_index,
            "title": job.get("title", ""),
            "parse_fail": True,
            "raw_output": raw_output[:500],
        }

    validation = validate_prediction(parsed)
    if not validation["valid"]:
        # Mark as parse_fail=True so compute_hybrid treats it as missing prediction
        # (avoids KeyError on mp["pred_tokens"] when has_model=True)
        return {
            "job_index": job_index,
            "title": job.get("title", ""),
            "parse_fail": True,
            "invalid_token": True,
            "raw_output": raw_output[:500],
            "errors": validation["errors"],
        }

    pred = validation["corrected"]
    pred_computed = compute_from_tokens(pred)
    golden_computed = compute_from_tokens(
        {f: job[f] for f in ("loc", "arr", "sen", "tech", "comp")}
    )

    return {
        "job_index": job_index,
        "title": job.get("title", ""),
        "parse_fail": False,
        "pred_tokens": {f: pred[f] for f in ("loc", "arr", "sen", "tech", "comp")},
        "golden_tokens": {f: job[f] for f in ("loc", "arr", "sen", "tech", "comp")},
        "pred_label": pred_computed["label"],
        "golden_label": golden_computed["label"],
        "pred_score": pred_computed["score"],
        "golden_score": golden_computed["score"],
        "label_match": pred_computed["label"] == golden_computed["label"],
    }


# ── Adapter prep ─────────────────────────────────────────────────────────────

def prep_adapter(m: ModelConfig) -> str | None:
    """
    Return adapter path suitable for mlx_lm.load().

    If the adapter is a .safetensors FILE, copy to an isolated directory
    (master_eval/adapters/{name}/adapters.safetensors) so mlx_lm finds it
    by convention. If it's already a directory, pass as-is.
    """
    if not m.adapter:
        return None

    ap = Path(_res(m.adapter))
    if not ap.exists():
        return None

    if ap.is_dir():
        return str(ap)

    # Copy to isolated directory
    dst_dir = ADAPTERS_DIR / m.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "adapters.safetensors"
    if not dst.exists():
        shutil.copy2(str(ap), str(dst))
    # Also copy adapter_config.json (LoRA rank/alpha settings) if present
    config_src = ap.parent / "adapter_config.json"
    config_dst = dst_dir / "adapter_config.json"
    if config_src.exists() and not config_dst.exists():
        shutil.copy2(str(config_src), str(config_dst))
    return str(dst_dir)


# ── Hybrid scoring (in-process, regex cached) ────────────────────────────────
#
# The regex classification (deterministic_baseline_v13_1.classify_job) produces
# identical results for all 15 models — same 239 jobs, same rules.
# We run it ONCE at startup via compute_regex_cache(), then each model's
# scoring combines cached regex + model predictions. Zero repeated work.

# Score maps (same as compute_hybrid_v13_1.py)
_LOC_MAP  = {"IN_LONDON": 25, "REMOTE": 25, "UK_OTHER": 10, "OUTSIDE_UK": -50, "UNK": 0}
_SEN_MAP  = {"LEVEL_3": 25, "LEVEL_2": 15, "LEVEL_1": 0}
_TECH_MAP = {"OOS": 0, "NODE": 10, "REACT": 5, "JS_TS": 5, "AI_ML": 10}
_COMP_MAP = {"NO_GBP": 0, "UP_TO_ONLY": 0, "BELOW_45K": -30, "RANGE_45_54K": 0,
             "RANGE_55_74K": 5, "RANGE_75_99K": 15, "ABOVE_100K": 25}


def _compute_label(loc: str, sen: str, tech: list, comp: str) -> dict:
    """Score + label from tokens. Same logic as compute_hybrid_v13_1.py."""
    loc_s  = _LOC_MAP.get(loc, 0)
    comp_s = _COMP_MAP.get(comp, 0)
    is_oos = "OOS" in tech or len(tech) == 0
    tech_s = 0 if is_oos else sum(_TECH_MAP.get(t, 0) for t in tech)
    role_s = 0 if is_oos else _SEN_MAP.get(sen, 0)
    score  = max(0, min(100, loc_s + role_s + tech_s + comp_s))
    label  = "good_fit" if score >= 70 else "maybe" if score >= 50 else "bad_fit"
    return {"score": score, "label": label}


def compute_all_regex_caches(test_jobs: list[dict]) -> dict[str, list[dict]]:
    """
    Run ALL regex versions once on all test jobs.
    Returns {"v12": [...], "v13": [...], "v13.1": [...]}.
    Each list has 239 dicts with loc/arr/sen/tech/comp fields.
    """
    caches = {}
    for name, classify_fn in REGEX_VERSIONS.items():
        caches[name] = [classify_fn(job) for job in test_jobs]
    return caches


def score_hybrid(test_jobs: list[dict], regex_cache: list[dict],
                 predictions: list[dict], out_dir: Path) -> tuple[dict | None, float]:
    """
    In-process hybrid scoring — same logic as compute_hybrid_v13_1.py --v12.
    Uses pre-computed regex_cache. Returns (summary_dict, elapsed_seconds).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pred_by_idx = {p["job_index"]: p for p in predictions}
    flds = ["loc", "arr", "sen", "tech", "comp"]

    results = {
        "model_only": {"correct": 0, "total": 0, "parse_fail": 0,
                       "field_correct": {f: 0 for f in flds}},
        "regex_only": {"correct": 0, "total": 0,
                       "field_correct": {f: 0 for f in flds}},
        "hybrid_A":   {"correct": 0, "total": 0, "parse_fail": 0,
                       "field_correct": {f: 0 for f in flds}},
        "v12_hybrid": {"correct": 0, "total": 0, "parse_fail": 0,
                       "field_correct": {f: 0 for f in flds}},
    }
    v12_errors = []
    v12_per_label = {l: {"correct": 0, "total": 0} for l in ["good_fit", "maybe", "bad_fit"]}

    for i, job in enumerate(test_jobs):
        golden_label = job["label"]
        golden = {f: job[f] for f in flds}
        regex = regex_cache[i]
        mp = pred_by_idx.get(i + 1)
        has_model = mp is not None and not mp.get("parse_fail", False)

        # ── Regex only ────────────────────────────────────────────────
        rx_lbl = _compute_label(regex["loc"], regex["sen"], regex["tech"], regex["comp"])["label"]
        results["regex_only"]["total"] += 1
        if rx_lbl == golden_label:
            results["regex_only"]["correct"] += 1
        for f in ["loc", "arr", "sen", "comp"]:
            if regex[f] == golden[f]:
                results["regex_only"]["field_correct"][f] += 1
        if sorted(regex["tech"]) == sorted(golden["tech"]):
            results["regex_only"]["field_correct"]["tech"] += 1

        # ── Model only ────────────────────────────────────────────────
        if has_model:
            mt = mp["pred_tokens"]
            results["model_only"]["total"] += 1
            if mp.get("label_match", False):
                results["model_only"]["correct"] += 1
            for f in ["loc", "arr", "sen", "comp"]:
                if mt[f] == golden[f]:
                    results["model_only"]["field_correct"][f] += 1
            if sorted(mt["tech"]) == sorted(golden["tech"]):
                results["model_only"]["field_correct"]["tech"] += 1
        else:
            results["model_only"]["parse_fail"] += 1

        # ── Hybrid A: model loc/arr/sen + regex tech/comp ─────────────
        if has_model:
            mt = mp["pred_tokens"]
            ha_lbl = _compute_label(mt["loc"], mt["sen"], regex["tech"], regex["comp"])["label"]
            results["hybrid_A"]["total"] += 1
            if ha_lbl == golden_label:
                results["hybrid_A"]["correct"] += 1
            for f in ["loc", "arr", "sen"]:
                if mt[f] == golden[f]:
                    results["hybrid_A"]["field_correct"][f] += 1
            if sorted(regex["tech"]) == sorted(golden["tech"]):
                results["hybrid_A"]["field_correct"]["tech"] += 1
            if regex["comp"] == golden["comp"]:
                results["hybrid_A"]["field_correct"]["comp"] += 1
        else:
            results["hybrid_A"]["parse_fail"] += 1
            fb_lbl = _compute_label(regex["loc"], regex["sen"], regex["tech"], regex["comp"])["label"]
            results["hybrid_A"]["total"] += 1
            if fb_lbl == golden_label:
                results["hybrid_A"]["correct"] += 1

        # ── V12 Hybrid: regex loc/tech/comp + model sen/arr ───────────
        v12_loc, v12_tech, v12_comp = regex["loc"], regex["tech"], regex["comp"]
        if has_model:
            mt = mp["pred_tokens"]
            v12_sen, v12_arr = mt["sen"], mt["arr"]
        else:
            results["v12_hybrid"]["parse_fail"] += 1
            v12_sen, v12_arr = regex["sen"], regex["arr"]

        v12_result = _compute_label(v12_loc, v12_sen, v12_tech, v12_comp)
        v12_label = v12_result["label"]
        results["v12_hybrid"]["total"] += 1
        v12_per_label[golden_label]["total"] += 1

        if v12_label == golden_label:
            results["v12_hybrid"]["correct"] += 1
            v12_per_label[golden_label]["correct"] += 1
        else:
            diffs = []
            if v12_loc != golden["loc"]:
                diffs.append(f"loc:{golden['loc']}->{v12_loc}")
            if v12_sen != golden["sen"]:
                diffs.append(f"sen:{golden['sen']}->{v12_sen}")
            if sorted(v12_tech) != sorted(golden["tech"]):
                diffs.append(f"tech:{golden['tech']}->{v12_tech}")
            if v12_comp != golden["comp"]:
                diffs.append(f"comp:{golden['comp']}->{v12_comp}")
            v12_errors.append({
                "idx": i, "golden_label": golden_label, "v12_label": v12_label,
                "golden_score": job.get("score", "?"), "v12_score": v12_result["score"],
                "title": job.get("title", ""), "diffs": diffs,
            })

        # V12 per-field
        for f, val in [("loc", v12_loc), ("arr", v12_arr), ("sen", v12_sen), ("comp", v12_comp)]:
            if val == golden[f]:
                results["v12_hybrid"]["field_correct"][f] += 1
        if sorted(v12_tech) == sorted(golden["tech"]):
            results["v12_hybrid"]["field_correct"]["tech"] += 1

    # ── Build summary (same JSON structure as compute_hybrid_v13_1.py) ────
    summary = {}
    for name, r in results.items():
        total = r["total"]
        summary[name] = {
            "correct": r["correct"], "total": total,
            "accuracy_pct": round(100 * r["correct"] / total, 1) if total > 0 else 0,
            "parse_fail": r.get("parse_fail", 0),
            "field_accuracy": {
                f: round(100 * r["field_correct"][f] / total, 1) if total > 0 else 0
                for f in flds
            },
        }
    summary["v12_errors"] = [
        {"job_index": e["idx"] + 1, "golden": e["golden_label"],
         "predicted": e["v12_label"], "diffs": e["diffs"]}
        for e in v12_errors
    ]
    summary["v12_per_label"] = {
        lbl: {"correct": v12_per_label[lbl]["correct"],
              "total": v12_per_label[lbl]["total"]}
        for lbl in ["good_fit", "maybe", "bad_fit"]
    }

    hybrid_json = out_dir / "hybrid_summary.json"
    with open(hybrid_json, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, time.time() - t0



# ── MLX single-prompt eval ────────────────────────────────────────────────────

def _compute_batch_size(model, prompts: list[list[int]], m: ModelConfig) -> int:
    """
    Compute safe batch size from model architecture + available RAM.

    Uses the model config to calculate KV cache per prompt, then fits
    as many prompts as possible into 40% of physical RAM (proven safe —
    65% caused OOM on 4B bfloat16).

    Returns batch size (clamped to [16, len(prompts)]).
    """
    cfg = getattr(model, "args", getattr(model, "config", None))
    num_layers   = getattr(cfg, "num_hidden_layers", 32)
    num_heads    = getattr(cfg, "num_attention_heads", 32)
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim     = getattr(cfg, "head_dim",
                           getattr(cfg, "hidden_size", 2048) // num_heads)

    # KV cache per prompt at max sequence length
    kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
    avg_input = sum(len(p) for p in prompts) / max(len(prompts), 1)
    max_seq = int(avg_input + m.max_tokens)
    kv_per_prompt_gb = kv_bytes_per_token * max_seq / 1e9

    # Available RAM: 40% of physical — leaves headroom for OS, Metal, activations
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / 1e9
    except ImportError:
        import subprocess as sp
        out = sp.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        total_ram = int(out) / 1e9

    available = total_ram * 0.40 - m.size_gb
    batch = max(16, min(len(prompts), int(available / max(kv_per_prompt_gb, 0.0001))))

    print(f"  [BATCH]  {m.name}  "
          f"kv/prompt={kv_per_prompt_gb:.3f}G  avail={available:.0f}G  "
          f"-> batch={batch}")
    return batch


def _clear_gpu():
    """Force GPU memory cleanup between models to prevent state corruption."""
    gc.collect()
    try:
        import mlx.core as mx
        mx.synchronize()
    except Exception:
        pass


def eval_mlx(m: ModelConfig, test_jobs: list, regex_caches: dict[str, list[dict]]) -> dict:
    """
    Evaluate one MLX model using batch_generate with architecture-adaptive batch size.

    Batch size is computed from the model's KV cache requirements and available RAM.
    GPU memory is cleaned between models to prevent state corruption after OOM.
    """
    import mlx.core as mx
    from mlx_lm import load, batch_generate

    out_dir = RESULTS_DIR / m.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "name": m.name, "display": m.display, "size_gb": m.size_gb,
        "expected": m.expected, "notes": m.notes,
        "error": None, "t_eval": 0.0, "t_hybrid": 0.0,
        "hybrid": None, "reused": False,
    }

    # ── Load model ──────────────────────────────────────────────────────────
    adapter_path = prep_adapter(m)
    t0 = time.time()
    print(f"  [LOAD]   {m.name}  ({m.model})", end="", flush=True)
    model, tokenizer = load(_res(m.model), adapter_path=adapter_path)
    t_load = time.time() - t0
    print(f"  {t_load:.1f}s")

    with open(REPO / m.prompt) as f:
        prompt_template = f.read()

    # ── Pre-tokenize all jobs ───────────────────────────────────────────────
    is_qwen3 = "qwen3" in m.model.lower()

    print(f"  [PREP]   {m.name}  building prompts...", end="", flush=True)
    t1 = time.time()
    all_prompts = []  # tokenized: List[List[int]]
    prompt_has_begin = False  # set by build_prompt based on template content
    for job in test_jobs:
        prompt_str, prompt_has_begin = build_prompt(
            tokenizer, job, prompt_template, m.model, m.no_think)
        all_prompts.append(tokenizer.encode(prompt_str))
    use_brace_prefix = not prompt_has_begin and (not is_qwen3 or m.no_think)
    t_tok = time.time() - t1
    avg_len = sum(len(p) for p in all_prompts) / max(len(all_prompts), 1)
    print(f"  {t_tok:.1f}s  (avg {avg_len:.0f} tokens/job)")

    # ── Batch generate in chunks ──────────────────────────────────────────
    n = len(test_jobs)
    batch_size = _compute_batch_size(model, all_prompts, m)
    num_batches = (n + batch_size - 1) // batch_size
    print(f"  [GEN]    {m.name}  {n} jobs in {num_batches} batches of {batch_size}  "
          f"(max_tokens={m.max_tokens})", flush=True)

    predictions = []
    parse_fails = 0
    invalid_tokens = 0
    t_gen_start = time.time()

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_prompts = all_prompts[start:end]
        batch_jobs = test_jobs[start:end]

        t_batch = time.time()
        response = batch_generate(
            model, tokenizer,
            prompts=batch_prompts,
            max_tokens=m.max_tokens,
            verbose=False,
        )
        t_batch_elapsed = time.time() - t_batch

        # Parse this batch's outputs
        for j, (text, job) in enumerate(zip(response.texts, batch_jobs)):
            job_idx = start + j + 1  # 1-indexed
            if use_brace_prefix:
                full_text = "{" + text
            else:
                full_text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

            parsed = parse_json_output(full_text)
            pred = build_prediction(job_idx, job, parsed, full_text)
            predictions.append(pred)

            if pred.get("parse_fail"):
                parse_fails += 1
            elif pred.get("invalid_token"):
                invalid_tokens += 1

        # Progress per batch
        done = end
        elapsed = time.time() - t_gen_start
        avg_s = elapsed / done
        eta = avg_s * (n - done)
        valid = len(predictions) - parse_fails - invalid_tokens
        print(f"  [{done:>3}/{n}]  batch {batch_idx+1}/{num_batches}  "
              f"{t_batch_elapsed:.1f}s  "
              f"{valid} ok / {parse_fails} fail  "
              f"ETA {_fmt_time(eta)}", flush=True)

    t_gen = time.time() - t_gen_start
    result["t_eval"] = t_load + t_tok + t_gen

    valid_count = len(predictions) - parse_fails - invalid_tokens
    avg_per_job = t_gen / max(n, 1)
    print(f"  [PARSE]  {m.name}  "
          f"{valid_count} valid / {parse_fails} parse fails / "
          f"{invalid_tokens} invalid  "
          f"({avg_per_job:.1f}s/job, {_fmt_time(t_gen)} total)")

    # ── Save predictions ────────────────────────────────────────────────────
    pred_file = out_dir / "all.predictions.jsonl"
    with open(pred_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # ── Hybrid scoring (in-process, all 3 regex versions) ───────────────────
    t_h0 = time.time()
    all_hybrid = {}
    for rx_name, rx_cache in regex_caches.items():
        rx_dir = out_dir / f"hybrid_{rx_name.replace('.', '_')}"
        rx_dir.mkdir(exist_ok=True)
        summary, _ = score_hybrid(test_jobs, rx_cache, predictions, rx_dir)
        all_hybrid[rx_name] = summary

    result["t_hybrid"] = time.time() - t_h0
    result["hybrid"] = all_hybrid.get("v13.1")
    result["hybrid_all_regex"] = all_hybrid

    if result["hybrid"]:
        pct = (result["hybrid"].get("v12_hybrid") or {}).get("accuracy_pct", 0)
        rx_scores = "  ".join(
            f"{rx}={((all_hybrid[rx].get('v12_hybrid') or {}).get('accuracy_pct', 0)):.1f}%"
            for rx in ["v12", "v13", "v13.1"]
        )
        total_t = result["t_eval"] + result["t_hybrid"]
        print(f"  [DONE]   {m.name}  ->  {pct:.1f}% hybrid  ({rx_scores})  "
              f"({_fmt_time(total_t)})")
    else:
        result["error"] = "Hybrid scoring failed"
        print(f"  [FAIL]   {m.name}  hybrid scoring produced no output")

    # ── Cleanup GPU to prevent state corruption for next model ────────────
    del model, tokenizer
    _clear_gpu()

    return result

# ── GGUF eval via llama-server ────────────────────────────────────────────────
# Tested: llama-server at 0.8s/job beats Ollama at 3-5s/job for long prompts.
# Speed is limited by Homebrew's Metal shaders (31 tok/s vs optimal ~200 tok/s).
# Fix: install Xcode then `brew reinstall llama.cpp`.

LLAMA_SERVER = shutil.which("llama-server")
LLAMA_SERVER_PORT = 18080
GGUF_PARALLEL = 8


def eval_gguf(m: ModelConfig, test_jobs: list[dict], regex_caches: dict[str, list[dict]]) -> dict:
    """Evaluate a GGUF model via llama-server with concurrent HTTP requests."""
    import urllib.request
    import urllib.error

    out_dir = RESULTS_DIR / m.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "name": m.name, "display": m.display, "size_gb": m.size_gb,
        "expected": m.expected, "notes": m.notes,
        "error": None, "t_eval": 0.0, "t_hybrid": 0.0,
        "hybrid": None, "reused": False,
    }

    if not LLAMA_SERVER:
        result["error"] = "llama-server not found (brew install llama.cpp)"
        print(f"  [FAIL]   {m.name}  llama-server not installed")
        return result

    with open(REPO / m.prompt) as f:
        prompt_template = f.read()

    # ── Start llama-server ──────────────────────────────────────────────────
    model_path = _res(m.model)
    server_log = out_dir / "server.log"
    server_cmd = [
        LLAMA_SERVER, "-m", model_path,
        "--port", str(LLAMA_SERVER_PORT),
        "--ctx-size", str(GGUF_PARALLEL * 2560),
        "--n-gpu-layers", "99",
        "--flash-attn", "on",
        "--parallel", str(GGUF_PARALLEL),
    ]

    print(f"  [SERVER] {m.name}  starting llama-server...", end="", flush=True)
    t0 = time.time()
    srv_log_fh = open(server_log, "w")
    server_proc = subprocess.Popen(
        server_cmd, stdout=srv_log_fh, stderr=subprocess.STDOUT)

    api_base = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
    ready = False
    for _ in range(120):
        time.sleep(0.5)
        try:
            req = urllib.request.Request(f"{api_base}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            if server_proc.poll() is not None:
                break

    t_load = time.time() - t0
    if not ready:
        server_proc.kill()
        srv_log_fh.close()
        result["error"] = "llama-server failed to start"
        print(f"  FAILED ({t_load:.0f}s)")
        return result
    print(f"  ready ({t_load:.1f}s)")

    # ── Concurrent inference ────────────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(test_jobs)
    predictions = []
    parse_fails = 0
    invalid_tokens = 0
    print(f"  [GEN]    {m.name}  {n} jobs, {GGUF_PARALLEL} concurrent slots", flush=True)

    def _infer_one(idx_job):
        i, job = idx_job
        raw_location = job.get("job_location", job.get("location", ""))
        jd_text = job.get("jd_text", job.get("description", ""))[:6000]
        prompt_text = (prompt_template
            .replace("{{job_title}}", job["title"])
            .replace("{{job_location}}", raw_location)
            .replace("{{jd_text}}", jd_text))
        req_body = json.dumps({
            "model": "local",
            "messages": [
                {"role": "system", "content": "Respond with JSON only. /no_think"},
                {"role": "user", "content": prompt_text},
            ],
            "max_tokens": m.max_tokens,
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            f"{api_base}/v1/chat/completions",
            data=req_body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_data = json.loads(resp.read())
            return i, resp_data["choices"][0]["message"]["content"]
        except Exception:
            return i, ""

    raw_responses = [""] * n
    t_gen_start = time.time()
    done_count = 0

    try:
        with ThreadPoolExecutor(max_workers=GGUF_PARALLEL) as pool:
            futures = {pool.submit(_infer_one, (i, job)): i
                       for i, job in enumerate(test_jobs)}
            for fut in as_completed(futures):
                idx, text = fut.result()
                raw_responses[idx] = text
                done_count += 1
                if done_count % 20 == 0 or done_count == n:
                    elapsed = time.time() - t_gen_start
                    avg_s = elapsed / done_count
                    eta = avg_s * (n - done_count)
                    print(f"  [{done_count:>3}/{n}]  "
                          f"avg {avg_s:.1f}s/job  ETA {_fmt_time(eta)}", flush=True)

        for i, (text, job) in enumerate(zip(raw_responses, test_jobs)):
            parsed = parse_json_output(text)
            pred = build_prediction(i + 1, job, parsed, text)
            predictions.append(pred)
            if pred.get("parse_fail"):
                parse_fails += 1
            elif pred.get("invalid_token"):
                invalid_tokens += 1
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        srv_log_fh.close()

    t_gen = time.time() - t_gen_start
    result["t_eval"] = t_load + t_gen

    valid_count = len(predictions) - parse_fails - invalid_tokens
    print(f"  [PARSE]  {m.name}  "
          f"{valid_count} valid / {parse_fails} parse fails / "
          f"{invalid_tokens} invalid  "
          f"({t_gen/max(n,1):.1f}s/job, {_fmt_time(t_gen)} total)")

    # ── Save predictions ────────────────────────────────────────────────────
    pred_file = out_dir / "all.predictions.jsonl"
    with open(pred_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # ── Hybrid scoring ──────────────────────────────────────────────────────
    t_h0 = time.time()
    all_hybrid = {}
    for rx_name, rx_cache in regex_caches.items():
        rx_dir = out_dir / f"hybrid_{rx_name.replace('.', '_')}"
        rx_dir.mkdir(exist_ok=True)
        summary, _ = score_hybrid(test_jobs, rx_cache, predictions, rx_dir)
        all_hybrid[rx_name] = summary

    result["t_hybrid"] = time.time() - t_h0
    result["hybrid"] = all_hybrid.get("v13.1")
    result["hybrid_all_regex"] = all_hybrid

    if result["hybrid"]:
        pct = (result["hybrid"].get("v12_hybrid") or {}).get("accuracy_pct", 0)
        rx_scores = "  ".join(
            f"{rx}={((all_hybrid[rx].get('v12_hybrid') or {}).get('accuracy_pct', 0)):.1f}%"
            for rx in ["v12", "v13", "v13.1"]
        )
        print(f"  [DONE]   {m.name}  ->  {pct:.1f}% hybrid  ({rx_scores})  "
              f"({_fmt_time(result['t_eval'])})")
    else:
        result["error"] = "Hybrid scoring failed"
        print(f"  [FAIL]   {m.name}  hybrid scoring failed")

    return result


# ── Progress tracker ──────────────────────────────────────────────────────────

class Progress:
    """Track model eval progress with ETA and elapsed time."""

    def __init__(self, total: int, start_time: float):
        self.total = total
        self.done = 0
        self.t_start = start_time
        self.model_times: list[float] = []  # elapsed per completed model

    def start_model(self, idx: int, name: str, display: str, size_gb: float,
                    is_gguf: bool) -> None:
        elapsed = time.time() - self.t_start
        eta_str = self._eta()
        kind = "GGUF" if is_gguf else f"MLX {size_gb:.1f}G"
        bar = self._bar()
        print(f"\n  {'─' * 80}")
        print(f"  [{idx}/{self.total}] {bar}  {display}")
        print(f"  Type: {kind}  |  Elapsed: {_fmt_time(elapsed)}  |  ETA: {eta_str}")
        print(f"  {'─' * 80}")

    def finish_model(self, t_model: float) -> None:
        self.done += 1
        self.model_times.append(t_model)

    def _eta(self) -> str:
        if not self.model_times:
            return "estimating..."
        avg = sum(self.model_times) / len(self.model_times)
        remaining = self.total - self.done
        eta_secs = avg * remaining
        if eta_secs < 60:
            return f"~{eta_secs:.0f}s"
        return f"~{_fmt_time(eta_secs)}"

    def _bar(self) -> str:
        pct = self.done / max(self.total, 1)
        filled = int(pct * 20)
        return f"[{'█' * filled}{'░' * (20 - filled)}] {self.done}/{self.total}"


def _fmt_time(secs: float) -> str:
    """Format seconds as human-readable."""
    if secs < 60:
        return f"{secs:.0f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Import report printer from run_all.py (same module, shared logic)
    sys.path.insert(0, str(Path(__file__).parent))
    from run_all import print_report

    parser = argparse.ArgumentParser(
        description="Native MLX batch eval — all models, maximum throughput")
    parser.add_argument("--only",  nargs="+", metavar="NAME",
                        help="Run only these model names")
    parser.add_argument("--skip",  nargs="+", metavar="NAME",
                        help="Skip these model names")
    parser.add_argument("--reuse", action="store_true",
                        help="Load cached hybrid_summary.json instead of re-running")
    args = parser.parse_args()

    # Filter models
    models: list[ModelConfig] = list(MODELS)
    if args.only:
        models = [m for m in models if m.name in args.only]
        unknown = set(args.only) - {m.name for m in MODELS}
        if unknown:
            print(f"WARNING: unknown model names: {unknown}")
    if args.skip:
        models = [m for m in models if m.name not in args.skip]

    # Check availability
    runnable: list[ModelConfig] = []
    skipped: list[tuple[ModelConfig, str]] = []
    for m in models:
        ok, reason = check_available(m)
        if ok:
            runnable.append(m)
        else:
            skipped.append((m, reason))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Separate MLX from GGUF
    mlx_models  = [m for m in runnable if not m.is_gguf]
    gguf_models = [m for m in runnable if m.is_gguf]

    # Load test data once (shared across all models)
    test_jobs: list[dict] = []
    with open(TEST_FILE) as f:
        for line in f:
            if line.strip():
                test_jobs.append(json.loads(line))

    # Compute ALL regex versions once (each model scored against all 3 for free)
    print("  Computing regex caches (v12, v13, v13.1)...", end="", flush=True)
    t_regex = time.time()
    regex_caches = compute_all_regex_caches(test_jobs)
    t_regex = time.time() - t_regex
    print(f" {t_regex:.2f}s ({len(test_jobs)} jobs × {len(regex_caches)} regex versions)")

    now = datetime.datetime.now()
    print(f"\n{'═' * 80}")
    print(f"  MASTER EVAL — Native MLX Batch")
    print(f"  Started: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:  {len(runnable)} to run ({len(mlx_models)} MLX + {len(gguf_models)} GGUF)"
          f" + {len(skipped)} skipped")
    print(f"  Jobs:    {len(test_jobs)} (audited test set)")
    print(f"  Regex:   {len(regex_caches)} versions (v12, v13, v13.1) — computed once")
    print(f"  Reuse:   {args.reuse}")
    print(f"{'═' * 80}")

    if skipped:
        print("\n  Skipped (files missing):")
        for m, reason in skipped:
            print(f"    {m.name}: {reason}")

    if runnable:
        print(f"\n  Eval queue:")
        for i, m in enumerate(runnable, 1):
            kind = "GGUF" if m.is_gguf else "MLX"
            think = " thinking" if not m.no_think and "qwen3" in m.model.lower() else ""
            print(f"    {i:>2}. {m.name:<28} {kind}{think:<12} {m.size_gb:.1f} GB  "
                  f"max_tok={m.max_tokens}")

    print()

    t_wall = time.time()
    results: list[dict] = []
    progress = Progress(total=len(runnable), start_time=t_wall)

    # ── Unified model loop (MLX then GGUF, same progress tracking) ─────────
    all_models = list(mlx_models) + list(gguf_models)
    for model_idx, m in enumerate(all_models, 1):
        # Check reuse first
        if args.reuse:
            # Multi-regex scoring saves to hybrid_v13_1/ subdirectory
            cached = RESULTS_DIR / m.name / "hybrid_v13_1" / "hybrid_summary.json"
            if cached.exists():
                # Load primary (v13.1) result
                with open(cached) as f:
                    hybrid = json.load(f)
                # Also load other regex versions if available
                all_rx = {"v13.1": hybrid}
                for rx in ["v12", "v13"]:
                    rx_file = RESULTS_DIR / m.name / f"hybrid_{rx.replace('.', '_')}" / "hybrid_summary.json"
                    if rx_file.exists():
                        with open(rx_file) as f:
                            all_rx[rx] = json.load(f)
                pct = (hybrid.get("v12_hybrid") or {}).get("accuracy_pct", 0)
                progress.start_model(model_idx, m.name, m.display, m.size_gb, m.is_gguf)
                print(f"  [REUSE]  {m.name}  ->  {pct:.1f}% (cached)")
                results.append({
                    "name": m.name, "display": m.display, "size_gb": m.size_gb,
                    "expected": m.expected, "notes": m.notes,
                    "error": None, "t_eval": 0.0, "t_hybrid": 0.0,
                    "hybrid": hybrid, "reused": True,
                    "hybrid_all_regex": all_rx,
                })
                progress.finish_model(0.1)
                continue

        progress.start_model(model_idx, m.name, m.display, m.size_gb, m.is_gguf)
        t_model = time.time()

        try:
            if m.is_gguf:
                r = eval_gguf(m, test_jobs, regex_caches)
            else:
                r = eval_mlx(m, test_jobs, regex_caches)
            results.append(r)
        except Exception as e:
            print(f"  [FAIL]   {m.name}  {type(e).__name__}: {e}")
            results.append({
                "name": m.name, "display": m.display, "size_gb": m.size_gb,
                "expected": m.expected, "notes": m.notes,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
                "t_eval": 0.0, "t_hybrid": 0.0, "hybrid": None, "reused": False,
            })

        progress.finish_model(time.time() - t_model)

    # ── Skipped models ──────────────────────────────────────────────────────
    for m, reason in skipped:
        results.append({
            "name": m.name, "display": m.display, "size_gb": m.size_gb,
            "expected": m.expected, "notes": m.notes,
            "error": f"SKIPPED — {reason}",
            "t_eval": 0.0, "t_hybrid": 0.0, "hybrid": None, "reused": False,
        })

    elapsed = time.time() - t_wall
    finished = datetime.datetime.now()
    print(f"\n{'═' * 80}")
    print(f"  All {len(runnable)} evals finished in {_fmt_time(elapsed)}")
    print(f"  Started:  {now.strftime('%H:%M:%S')}  |  Finished: {finished.strftime('%H:%M:%S')}")
    if progress.model_times:
        avg_t = sum(progress.model_times) / len(progress.model_times)
        print(f"  Avg per model: {_fmt_time(avg_t)}  |  "
              f"Fastest: {_fmt_time(min(progress.model_times))}  |  "
              f"Slowest: {_fmt_time(max(progress.model_times))}")
    print(f"{'═' * 80}\n")

    print_report(results)
    print_conclusions(results)


# ── Conclusions report ───────────────────────────────────────────────────────

def print_conclusions(results: list[dict]) -> None:
    """
    Print a human-readable conclusions section comparing all models.
    Finds winners across multiple metrics and highlights key findings.
    """
    valid = [r for r in results if r.get("hybrid") and not r.get("error")]
    if not valid:
        return

    W = 120
    sep = "═" * W

    # Extract metrics into a flat list for easy comparison
    rows = []
    for r in valid:
        h = r["hybrid"]
        v12 = h.get("v12_hybrid") or {}
        mo  = h.get("model_only") or {}
        fa  = v12.get("field_accuracy") or {}
        mo_fa = mo.get("field_accuracy") or {}

        rows.append({
            "name":       r["name"],
            "display":    r["display"],
            "size_gb":    r["size_gb"],
            "hybrid_pct": v12.get("accuracy_pct", 0),
            "hybrid_n":   v12.get("correct", 0),
            "mo_pct":     mo.get("accuracy_pct", 0),
            "mo_n":       mo.get("correct", 0),
            "parse_fail": mo.get("parse_fail", 0),
            "h_loc":      fa.get("loc", 0),
            "h_arr":      fa.get("arr", 0),
            "h_sen":      fa.get("sen", 0),
            "h_tech":     fa.get("tech", 0),
            "h_comp":     fa.get("comp", 0),
            "mo_sen":     mo_fa.get("sen", 0),
            "mo_arr":     mo_fa.get("arr", 0),
            "expected":   r.get("expected", ""),
            "t_total":    r.get("t_eval", 0) + r.get("t_hybrid", 0),
        })

    if not rows:
        return

    print(f"\n{sep}")
    print(f"  CONCLUSIONS")
    print(f"{sep}")

    # ── Winners ─────────────────────────────────────────────────────────────
    def best(key, label, unit="%"):
        top = max(rows, key=lambda r: r[key])
        ties = [r for r in rows if r[key] == top[key]]
        names = ", ".join(r["name"] for r in ties)
        return f"  {label:<36} {top[key]:>6.1f}{unit}  ← {names}"

    def smallest_winner(key, label):
        """Among models tied for the best score on 'key', pick the smallest."""
        top_val = max(r[key] for r in rows)
        tied = [r for r in rows if r[key] == top_val]
        winner = min(tied, key=lambda r: r["size_gb"])
        return (f"  {label:<36} {winner['name']} "
                f"({top_val:.1f}% at {winner['size_gb']:.2f} GB)")

    print()
    print("  CATEGORY WINNERS:")
    print(f"  {'─' * 60}")
    print(best("hybrid_pct", "Best hybrid accuracy"))
    print(best("mo_pct", "Best model-only accuracy"))
    print(best("mo_sen", "Best seniority (model-only)"))
    print(best("mo_arr", "Best arrangement (model-only)"))
    print(best("h_sen", "Best seniority (hybrid)"))

    # Best efficiency: highest hybrid accuracy per GB
    for r in rows:
        r["eff"] = r["hybrid_pct"] / max(r["size_gb"], 0.01)
    eff_top = max(rows, key=lambda r: r["eff"])
    print(f"  {'Best efficiency (hybrid/GB)':<36} {eff_top['hybrid_pct']:.1f}% "
          f"in {eff_top['size_gb']:.2f} GB  ← {eff_top['name']}")

    # Fewest parse failures (among models with >0)
    parseable = [r for r in rows if r["parse_fail"] > 0]
    if parseable:
        worst_parse = max(parseable, key=lambda r: r["parse_fail"])
        print(f"  {'Most parse failures':<36} {worst_parse['parse_fail']:>6}    ← {worst_parse['name']}")
    zero_parse = [r for r in rows if r["parse_fail"] == 0]
    if zero_parse:
        names = ", ".join(r["name"] for r in zero_parse)
        print(f"  {'Zero parse failures':<36}          ← {names}")

    print()
    print(smallest_winner("hybrid_pct", "Best hybrid (smallest model):"))
    print(smallest_winner("mo_pct", "Best model-only (smallest model):"))

    # ── Expected vs actual ──────────────────────────────────────────────────
    mismatches = []
    for r in rows:
        if r["expected"] and r["expected"] != "broken":
            exp_val = float(r["expected"].replace("%", ""))
            if abs(r["hybrid_pct"] - exp_val) > 0.1:
                mismatches.append(r)

    if mismatches:
        print()
        print("  EXPECTED vs ACTUAL (mismatches > 0.1pp):")
        print(f"  {'─' * 60}")
        for r in mismatches:
            exp_str = r["expected"]
            print(f"  {r['name']:<28}  expected {exp_str:>7}  actual {r['hybrid_pct']:>5.1f}%  "
                  f"Δ {r['hybrid_pct'] - float(exp_str.replace('%','')):>+5.1f}pp")

    # ── Quantization ladder (V14 models) ────────────────────────────────────
    v14 = [r for r in rows if "v14" in r["name"]]
    if len(v14) > 1:
        v14.sort(key=lambda r: r["size_gb"], reverse=True)
        print()
        print("  V14 QUANTIZATION LADDER (same weights, different precision):")
        print(f"  {'─' * 60}")
        print(f"  {'Format':<28}  {'Size':>6}  {'Hybrid':>7}  {'MO':>7}  {'Parse':>5}")
        print(f"  {'─'*28}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*5}")
        for r in v14:
            print(f"  {r['name']:<28}  {r['size_gb']:>5.1f}G  "
                  f"{r['hybrid_pct']:>6.1f}%  {r['mo_pct']:>6.1f}%  {r['parse_fail']:>5}")

    # ── Multi-regex comparison ─────────────────────────────────────────────
    multi_rx = [r for r in results if r.get("hybrid_all_regex") and not r.get("error")]
    if multi_rx:
        print()
        print("  REGEX VERSION COMPARISON (same model predictions, 3 regex classifiers):")
        print(f"  {'─' * 75}")
        print(f"  {'Model':<28}  {'v12 regex':>9}  {'v13 regex':>9}  {'v13.1 regex':>11}  {'Best'}")
        print(f"  {'─'*28}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*6}")
        for r in multi_rx:
            arx = r["hybrid_all_regex"]
            scores = {}
            for rx in ["v12", "v13", "v13.1"]:
                s = arx.get(rx, {})
                scores[rx] = (s.get("v12_hybrid") or {}).get("accuracy_pct", 0)
            best_rx = max(scores, key=scores.get)
            print(f"  {r['name']:<28}  {scores['v12']:>8.1f}%  {scores['v13']:>8.1f}%  "
                  f"{scores['v13.1']:>10.1f}%  {best_rx}")

        # Summary: which regex wins most often?
        rx_wins = {rx: 0 for rx in ["v12", "v13", "v13.1"]}
        for r in multi_rx:
            arx = r["hybrid_all_regex"]
            scores = {rx: (arx.get(rx, {}).get("v12_hybrid") or {}).get("accuracy_pct", 0)
                      for rx in ["v12", "v13", "v13.1"]}
            best = max(scores, key=scores.get)
            rx_wins[best] += 1
        print(f"\n  Regex wins: " + "  ".join(f"{rx}={n}" for rx, n in rx_wins.items()))

    # ── Generation progression (V7 → V14) ──────────────────────────────────
    gens = [r for r in rows if r["name"] in (
        "v7_0.5B", "v7_1.5B", "v12_0.6B", "v12_1_1.5B",
        "v13_0.6B", "v13_1_1.5B", "v14_4B_mlx6bit")]
    if len(gens) > 1:
        print()
        print("  VERSION PROGRESSION (best checkpoint per generation):")
        print(f"  {'─' * 60}")
        print(f"  {'Model':<28}  {'Size':>6}  {'Hybrid':>7}  {'MO':>7}")
        print(f"  {'─'*28}  {'─'*6}  {'─'*7}  {'─'*7}")
        for r in gens:
            print(f"  {r['name']:<28}  {r['size_gb']:>5.1f}G  "
                  f"{r['hybrid_pct']:>6.1f}%  {r['mo_pct']:>6.1f}%")

    # ── Key findings ────────────────────────────────────────────────────────
    print()
    print("  KEY FINDINGS:")
    print(f"  {'─' * 60}")

    # Find if small model matches large
    best_hybrid = max(r["hybrid_pct"] for r in rows)
    best_large = [r for r in rows if r["hybrid_pct"] == best_hybrid]
    smallest_at_best = min(best_large, key=lambda r: r["size_gb"])
    largest_at_best = max(best_large, key=lambda r: r["size_gb"])
    if smallest_at_best["size_gb"] < largest_at_best["size_gb"] * 0.8:
        ratio = largest_at_best["size_gb"] / smallest_at_best["size_gb"]
        print(f"  • {smallest_at_best['name']} matches {largest_at_best['name']} "
              f"at {ratio:.1f}× smaller")

    # Check if thinking mode helps or hurts
    think_on  = next((r for r in rows if r["name"] == "v14_4B_mlx6bit_think"), None)
    think_off = next((r for r in rows if r["name"] == "v14_4B_mlx6bit"), None)
    if think_on and think_off:
        diff = think_on["hybrid_pct"] - think_off["hybrid_pct"]
        verdict = "HELPS" if diff > 0 else "HURTS" if diff < 0 else "NO EFFECT"
        print(f"  • Thinking mode {verdict}: "
              f"{think_off['hybrid_pct']:.1f}% (off) vs {think_on['hybrid_pct']:.1f}% (on) "
              f"= {diff:+.1f}pp")

    # Model-only vs hybrid gap
    if rows:
        avg_mo = sum(r["mo_pct"] for r in rows) / len(rows)
        avg_hy = sum(r["hybrid_pct"] for r in rows) / len(rows)
        print(f"  • Regex hybrid boost: avg {avg_hy - avg_mo:+.1f}pp "
              f"(model-only {avg_mo:.1f}% → hybrid {avg_hy:.1f}%)")

    # Broken models
    broken = [r for r in rows if r["expected"] == "broken"]
    if broken:
        for r in broken:
            print(f"  • {r['name']}: {r['hybrid_pct']:.1f}% hybrid "
                  f"({r['parse_fail']} parse fails) — "
                  f"{'confirmed broken' if r['hybrid_pct'] < 90 else 'surprisingly functional'}")

    print(f"\n{sep}\n")

    # ── Save conclusions JSON ───────────────────────────────────────────────
    conclusions = {
        "best_hybrid": max(rows, key=lambda r: r["hybrid_pct"])["name"],
        "best_model_only": max(rows, key=lambda r: r["mo_pct"])["name"],
        "best_efficiency": eff_top["name"],
        "total_models": len(rows),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "conclusions.json"
    with open(out_json, "w") as f:
        json.dump(conclusions, f, indent=2)
    print(f"  Conclusions saved → {out_json}")


if __name__ == "__main__":
    main()

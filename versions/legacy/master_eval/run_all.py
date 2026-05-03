#!/usr/bin/env python3
"""
master_eval/run_all.py

Evaluates ALL models/adapters on the audited V12 test set (239 jobs).
Each model runs with the exact prompt and settings it was trained on.
MLX models use parallel chunked inference for maximum throughput on Apple Silicon.

SAFE: reads only from the repo, writes only to master_eval/results/ and master_eval/adapters/.
      No files outside master_eval/ are created or modified.

Usage:
    cd /path/to/ai_eval_harness
    .venv/bin/python3 master_eval/run_all.py [OPTIONS]

Options:
    --workers N      Max concurrent models (default: 4)
    --only NAME ...  Run only these model names
    --skip NAME ...  Skip these model names
    --reuse          Skip models that already have a hybrid_summary.json

Examples:
    .venv/bin/python3 master_eval/run_all.py                      # run all 10 models
    .venv/bin/python3 master_eval/run_all.py --workers 8          # 8 models in parallel
    .venv/bin/python3 master_eval/run_all.py --only v12_1_1.5B    # single model
    .venv/bin/python3 master_eval/run_all.py --skip v14_Q4_K_M    # skip one
    .venv/bin/python3 master_eval/run_all.py --reuse              # re-report without re-running
"""

import argparse
import json
import math
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# ── Paths (all relative — writes only to master_eval/) ────────────────────────

REPO         = Path(__file__).parent.parent
PYTHON       = str(REPO / ".venv/bin/python3")
TEST_FILE    = str(REPO / "versions/v12/data/v12_original/test_labeled_audited.jsonl")
EVAL_MLX     = str(REPO / "finetune/eval_student_v7.py")
EVAL_GGUF    = str(REPO / "finetune/eval_student_v14_gguf.py")
HYBRID       = str(REPO / "finetune/compute_hybrid_v13_1.py")
RESULTS_DIR  = Path(__file__).parent / "results"
ADAPTERS_DIR = Path(__file__).parent / "adapters"  # pre-copied adapters live here

# ── Adaptive RAM scheduler ────────────────────────────────────────────────────
# Tracks how much RAM is currently "in flight" across all running models.
# Before starting a model, a thread waits until budget allows it.
# When a model finishes it releases its reservation and wakes waiting threads.

_RAM_BUDGET_GB: float = 0.0   # set by main() at startup
_in_flight_gb:  float = 0.0
_ram_cond = threading.Condition()


def _total_ram_gb() -> float:
    """Physical RAM in GB (macOS via sysctl; psutil fallback)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / 1e9


def peak_ram_gb(m) -> float:
    """
    Estimated peak RAM when all subprocesses for this model are running.

    Each MLX subprocess loads the full model weights independently —
    so peak = size_gb × num_chunks + Python overhead per process.
    GGUF runs a single process with slightly higher overhead (llama-cpp-python).
    """
    if m.is_gguf:
        return m.size_gb * 1.2 + 0.3
    return m.size_gb * m.chunks + 0.12 * m.chunks


def _acquire_ram(needed_gb: float) -> bool:
    """Block until RAM budget allows this model, then reserve. Returns False if impossible."""
    global _in_flight_gb
    if needed_gb > _RAM_BUDGET_GB:
        # Model can never fit — even with nothing else running
        return False
    with _ram_cond:
        _ram_cond.wait_for(lambda: _in_flight_gb + needed_gb <= _RAM_BUDGET_GB)
        _in_flight_gb += needed_gb
    return True


def _release_ram(needed_gb: float) -> None:
    """Release reserved RAM and wake any threads waiting for budget."""
    global _in_flight_gb
    with _ram_cond:
        _in_flight_gb = max(0.0, _in_flight_gb - needed_gb)
        _ram_cond.notify_all()


# ── Model Registry ────────────────────────────────────────────────────────────

@dataclass
class Model:
    name: str            # unique key — used as result subdirectory name
    display: str         # human-readable label for the report
    model: str           # HF model ID or local path (~ supported)
    prompt: str          # path relative to REPO root
    size_gb: float       # approx weight size for the report
    adapter: str = ""    # path relative to REPO root; empty = standalone model
    no_think: bool = False
    max_tokens: int = 1000
    is_gguf: bool = False
    chunks: int = 16     # MLX only: parallel chunks (ignored for GGUF)
    expected: str = ""   # expected hybrid % for sanity check
    notes: str = ""


def _res(path: str) -> str:
    """Expand ~, resolve repo-relative paths, or pass through HF model IDs."""
    if not path:
        return ""
    if path.startswith("~"):
        return str(Path(path).expanduser())
    # Could be repo-relative (finetune/adapters_v7) or HF ID (mlx-community/Qwen3-0.6B-4bit)
    repo_path = REPO / path
    if repo_path.exists():
        return str(repo_path)
    return path  # pass through as HF model ID


MODELS: list[Model] = [

    # ── V7 era (pre-V12, Qwen2.5, student_v7.txt) ────────────────────────────
    Model(
        name     = "v7_0.5B",
        display  = "V7 0.5B — Qwen2.5-0.5B",
        model    = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        adapter  = "finetune/adapters_v7",
        prompt   = "prompts/student_v7.txt",
        size_gb  = 0.25,
        chunks   = 16,
        notes    = "Pre-V12 baseline, original Qwen2.5 0.5B",
    ),
    Model(
        name     = "v7_1.5B",
        display  = "V7 1.5B — Qwen2.5-1.5B",
        model    = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter  = "finetune/adapters_v7_1.5B",
        prompt   = "prompts/student_v7.txt",
        size_gb  = 0.84,
        chunks   = 16,
        notes    = "Pre-V12 baseline 1.5B",
    ),

    # ── V12 era (Qwen3 0.6B + Qwen2.5 1.5B, student_v7.txt) ─────────────────
    Model(
        name     = "v12_0.6B",
        display  = "V12 0.6B — Qwen3-0.6B  iter 1400",
        model    = "mlx-community/Qwen3-0.6B-4bit",
        adapter  = "finetune/adapters_v12_qwen3_0.6B/0001400_adapters.safetensors",
        prompt   = "prompts/student_v7.txt",
        size_gb  = 0.335,
        no_think = True,
        chunks   = 16,
        expected = "96.7%",
        notes    = "V12 Qwen3 0.6B, best checkpoint iter 1400",
    ),
    Model(
        name     = "v12_1_1.5B",
        display  = "V12.1 1.5B — Qwen2.5-1.5B  iter 2000",
        model    = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter  = "finetune/adapters_v12/0002000_adapters.safetensors",
        prompt   = "prompts/student_v7.txt",
        size_gb  = 0.84,
        chunks   = 16,
        expected = "98.3%",
        notes    = "V12.1 1.5B best checkpoint iter 2000",
    ),

    # ── V13 era (Qwen3 + corrective Qwen2.5, student_v13.txt/v13_1.txt) ──────
    Model(
        name     = "v13_0.6B",
        display  = "V13 0.6B — Qwen3-0.6B  iter 1500",
        model    = "mlx-community/Qwen3-0.6B-4bit",
        adapter  = "finetune/adapters_v13_0.6B/0001500_adapters.safetensors",
        prompt   = "prompts/student_v13.txt",
        size_gb  = 0.335,
        no_think = True,
        chunks   = 16,
        expected = "97.9%",
        notes    = "V13 production 0.6B, best iter 1500",
    ),
    Model(
        name     = "v13_1_0.6B",
        display  = "V13.1 0.6B corrective — Qwen3-0.6B  iter 200",
        model    = "mlx-community/Qwen3-0.6B-4bit",
        adapter  = "finetune/adapters_v13_1_0.6B/0000200_adapters.safetensors",
        prompt   = "prompts/student_v13_1.txt",
        size_gb  = 0.335,
        no_think = True,
        chunks   = 16,
        expected = "97.5%",
        notes    = "V13.1 0.6B corrective retrain, best iter 200",
    ),
    Model(
        name     = "v13_1_1.5B",
        display  = "V13.1 1.5B — Qwen2.5-1.5B  iter 1800",
        model    = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter  = "finetune/adapters_v13_1_1.5B/0001800_adapters.safetensors",
        prompt   = "prompts/student_v13_1.txt",
        size_gb  = 0.84,
        chunks   = 16,
        expected = "97.5%",
        notes    = "V13.1 1.5B fresh train, best iter 1800",
    ),

    # ── V14 era (Qwen3 4B — MLX + GGUF) ─────────────────────────────────────
    Model(
        name       = "v14_4B_mlx6bit",
        display    = "V14 4B MLX 6-bit — exp1 fix1  no-think",
        model      = "~/qwen3_4B_v14_mlx6bit",
        prompt     = "prompts/student_v14_exp1.txt",
        size_gb    = 3.1,
        no_think   = True,
        max_tokens = 1200,
        chunks     = 8,   # larger model — 8 chunks keeps peak RAM sane
        expected   = "98.3%",
        notes      = "V14 4B MLX 6-bit, exp1 fix1, no-think, 1200 tok budget",
    ),
    Model(
        name       = "v14_Q6_K",
        display    = "V14 4B Q6_K GGUF",
        model      = "~/qwen3_4B_v14_Q6_K.gguf",
        prompt     = "prompts/student_v14.txt",
        size_gb    = 3.1,
        is_gguf    = True,
        max_tokens = 600,
        chunks     = 1,   # GGUF: single process
        expected   = "98.3%",
        notes      = "V14 GGUF Q6_K, min viable quantisation for fine-tuned JSON schema",
    ),
    Model(
        name       = "v14_Q4_K_M",
        display    = "V14 4B Q4_K_M GGUF",
        model      = "~/qwen3_4B_v14_Q4_K_M.gguf",
        prompt     = "prompts/student_v14.txt",
        size_gb    = 2.3,
        is_gguf    = True,
        max_tokens = 600,
        chunks     = 1,   # GGUF: single process
        expected   = "97.9%",
        notes      = "V14 GGUF Q4_K_M, schema hallucination at 4-bit (low model-only)",
    ),
]


# ── Availability check ────────────────────────────────────────────────────────

def check_available(m: Model) -> tuple[bool, str]:
    """Return (available, reason). Skips models with missing files."""
    if m.adapter:
        ap = Path(_res(m.adapter))
        if not ap.exists():
            return False, f"adapter missing: {ap}"
    if m.model.startswith("~"):
        mp = Path(m.model).expanduser()
        if not mp.exists():
            return False, f"model missing: {mp}"
    if not (REPO / m.prompt).exists():
        return False, f"prompt missing: {m.prompt}"
    return True, "ok"


# ── Adapter pre-copy ──────────────────────────────────────────────────────────

def prep_adapter(m: Model) -> str:
    """
    Return the adapter path to pass to eval_student_v7.py.

    The race condition: when eval_student_v7.py receives a .safetensors FILE
    (not named 'adapters.safetensors'), it copies it to parent/adapters.safetensors.
    With 16 concurrent chunks all doing this simultaneously, writes collide.

    Fix: pre-copy to master_eval/adapters/{name}/adapters.safetensors ONCE before
    any chunks start. All chunks receive the directory and read the pre-copied file
    with zero write activity — no race.

    If the adapter is already a directory (e.g. adapters_v7/), pass it as-is;
    eval_student_v7.py reads adapters.safetensors from it without any copy.
    """
    if not m.adapter:
        return ""

    src = Path(_res(m.adapter))
    if not src.exists():
        return str(src)

    if src.is_dir():
        return str(src)   # already a dir with adapters.safetensors inside

    # It's a specific .safetensors file — copy to an isolated directory
    dst_dir = ADAPTERS_DIR / m.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "adapters.safetensors"
    if not dst.exists():
        shutil.copy2(str(src), str(dst))
    # Also copy adapter_config.json (LoRA rank/alpha) if present
    config_src = src.parent / "adapter_config.json"
    config_dst = dst_dir / "adapter_config.json"
    if config_src.exists() and not config_dst.exists():
        shutil.copy2(str(config_src), str(config_dst))
    return str(dst_dir)


# ── Single-chunk subprocess ───────────────────────────────────────────────────

def run_chunk(m: Model, chunk: int, adapter_path: str, out_dir: Path) -> tuple[int, int]:
    """
    Run one eval chunk as a subprocess. Returns (returncode, chunk_number).
    Each chunk processes ~(239 / m.chunks) jobs and writes its own predictions file.
    """
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    cmd = [
        PYTHON, EVAL_MLX,
        "--model",      _res(m.model),
        "--test-file",  TEST_FILE,
        "--prompt",     _res(m.prompt),
        "--output-dir", str(out_dir),
        "--save-predictions",
        "--max-tokens", str(m.max_tokens),
        "--chunk",      str(chunk),
        "--num-chunks", str(m.chunks),
    ]
    if adapter_path:
        cmd += ["--adapter", adapter_path]
    if m.no_think:
        cmd.append("--no-think")

    log_file = logs_dir / f"chunk_{chunk:02d}.log"
    with open(log_file, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)

    return proc.returncode, chunk


# ── Merge chunk predictions ───────────────────────────────────────────────────

def merge_predictions(out_dir: Path, num_chunks: int) -> Path | None:
    """
    Collect all *_chunk*of{N}.predictions.jsonl files (anywhere under out_dir),
    merge them, sort by job_index, and write to out_dir/all.predictions.jsonl.
    Returns the merged file path, or None if no chunk files found.
    """
    chunk_files = list(out_dir.rglob(f"*_chunk*of{num_chunks}.predictions.jsonl"))
    if not chunk_files:
        return None

    all_preds = []
    for cf in sorted(chunk_files):
        with open(cf) as f:
            for line in f:
                if line.strip():
                    all_preds.append(json.loads(line))

    all_preds.sort(key=lambda x: x.get("job_index", 0))

    merged = out_dir / "all.predictions.jsonl"
    with open(merged, "w") as f:
        for pred in all_preds:
            f.write(json.dumps(pred) + "\n")

    return merged


# ── Hybrid scoring ────────────────────────────────────────────────────────────

def run_hybrid(pred_file: Path, out_dir: Path) -> tuple[dict | None, float]:
    """
    Run compute_hybrid_v13_1.py on a predictions file.
    Returns (hybrid_dict, elapsed_seconds). dict is None if scoring failed.
    """
    t0 = time.time()
    hybrid_json = out_dir / "hybrid_summary.json"
    hybrid_log  = out_dir / "hybrid.log"

    cmd = [
        PYTHON, HYBRID,
        "--test-file",   TEST_FILE,
        "--predictions", str(pred_file),
        "--v12",
        "--output",      str(hybrid_json),
    ]
    with open(hybrid_log, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        subprocess.run(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0

    if hybrid_json.exists():
        with open(hybrid_json) as f:
            return json.load(f), elapsed
    return None, elapsed


# ── Model runners ─────────────────────────────────────────────────────────────

def run_model(m: Model, reuse: bool = False) -> dict:
    """
    Top-level runner for one model. Acquires RAM budget before starting,
    releases it when done. Thread-safe: all writes go to out_dir = results/{name}/.
    Dispatches to GGUF or MLX path.
    """
    needed = peak_ram_gb(m)
    if not _acquire_ram(needed):
        print(f"  [SKIP]   {m.name}  peak RAM {needed:.1f} GB exceeds budget {_RAM_BUDGET_GB:.0f} GB")
        return {
            "name": m.name, "display": m.display, "size_gb": m.size_gb,
            "expected": m.expected, "notes": m.notes,
            "error": f"Peak RAM {needed:.1f} GB exceeds budget {_RAM_BUDGET_GB:.0f} GB",
            "t_eval": 0.0, "t_hybrid": 0.0, "hybrid": None, "reused": False,
        }
    try:
        return _run_model_inner(m, reuse)
    finally:
        _release_ram(needed)


def _run_model_inner(m: Model, reuse: bool = False) -> dict:
    out_dir = RESULTS_DIR / m.name
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "name":    m.name,
        "display": m.display,
        "size_gb": m.size_gb,
        "expected": m.expected,
        "notes":   m.notes,
        "error":   None,
        "t_eval":  0.0,
        "t_hybrid": 0.0,
        "hybrid":  None,
        "reused":  False,
    }

    # Short-circuit if results already exist
    if reuse:
        cached = out_dir / "hybrid_summary.json"
        if cached.exists():
            with open(cached) as f:
                result["hybrid"] = json.load(f)
            result["reused"] = True
            pct = (result["hybrid"].get("v12_hybrid") or {}).get("accuracy_pct", 0)
            print(f"  [REUSE]  {m.name}  →  {pct:.1f}% (cached)")
            return result

    if m.is_gguf:
        return _run_gguf(m, out_dir, result)
    else:
        return _run_mlx(m, out_dir, result)


def _run_gguf(m: Model, out_dir: Path, result: dict) -> dict:
    """Single-process GGUF inference + hybrid scoring."""
    t0 = time.time()
    eval_log = out_dir / "eval.log"

    cmd = [
        PYTHON, EVAL_GGUF,
        "--model",      _res(m.model),
        "--test-file",  TEST_FILE,
        "--prompt",     _res(m.prompt),
        "--output-dir", str(out_dir),
        "--max-tokens", str(m.max_tokens),
    ]

    print(f"  [START]  {m.name}  (GGUF, single process)")
    with open(eval_log, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT)

    result["t_eval"] = time.time() - t0

    if proc.returncode != 0:
        result["error"] = f"GGUF inference failed (rc={proc.returncode})"
        print(f"  [FAIL]   {m.name}  rc={proc.returncode}")
        return result

    # GGUF eval always writes a predictions file directly to out_dir
    pred_files = sorted(out_dir.glob("*.predictions.jsonl"), key=lambda f: f.stat().st_mtime)
    if not pred_files:
        result["error"] = "No predictions file found after GGUF inference"
        print(f"  [FAIL]   {m.name}  no predictions")
        return result

    hybrid, t_hybrid = run_hybrid(pred_files[-1], out_dir)
    result["t_hybrid"] = t_hybrid
    _finalise(m, result, hybrid)
    return result


def _run_mlx(m: Model, out_dir: Path, result: dict) -> dict:
    """
    Parallel-chunked MLX inference + merge + hybrid scoring.

    Steps:
      1. Pre-copy adapter to master_eval/adapters/{name}/ (once, avoids write races)
      2. Spawn m.chunks subprocesses, each handling ~(239 / m.chunks) jobs
      3. Merge all chunk prediction files into out_dir/all.predictions.jsonl
      4. Run hybrid scoring on merged file
    """
    adapter_path = prep_adapter(m)

    # Clean stale chunk files from prior runs to prevent corrupt merge
    for old in out_dir.rglob(f"*_chunk*of{m.chunks}.predictions.jsonl"):
        old.unlink()

    t0 = time.time()
    print(f"  [START]  {m.name}  ({m.chunks} chunks in parallel)")

    failed_chunks: list[int] = []
    with ThreadPoolExecutor(max_workers=m.chunks) as pool:
        futures = {
            pool.submit(run_chunk, m, i, adapter_path, out_dir): i
            for i in range(1, m.chunks + 1)
        }
        for fut in as_completed(futures):
            rc, chunk_num = fut.result()
            if rc != 0:
                failed_chunks.append(chunk_num)

    result["t_eval"] = time.time() - t0

    if failed_chunks:
        result["error"] = f"Chunk(s) failed: {sorted(failed_chunks)}"
        print(f"  [FAIL]   {m.name}  chunk(s) {sorted(failed_chunks)} non-zero exit")
        return result

    pred_file = merge_predictions(out_dir, m.chunks)
    if not pred_file:
        result["error"] = "Merge failed: no chunk prediction files found"
        print(f"  [FAIL]   {m.name}  merge step found no chunk outputs")
        return result

    hybrid, t_hybrid = run_hybrid(pred_file, out_dir)
    result["t_hybrid"] = t_hybrid
    _finalise(m, result, hybrid)
    return result


def _finalise(m: Model, result: dict, hybrid: dict | None) -> None:
    """Attach hybrid results and print status line."""
    if hybrid:
        result["hybrid"] = hybrid
        pct = (hybrid.get("v12_hybrid") or {}).get("accuracy_pct", 0)
        t = result["t_eval"] + result["t_hybrid"]
        print(f"  [DONE]   {m.name}  →  {pct:.1f}% hybrid  "
              f"({result['t_eval']:.0f}s eval + {result['t_hybrid']:.0f}s hybrid = {t:.0f}s)")
    else:
        result["error"] = "Hybrid scoring failed — no output JSON"
        print(f"  [FAIL]   {m.name}  hybrid scoring step produced no output")


# ── Wilson score confidence interval ─────────────────────────────────────────

def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score CI for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    s = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / d
    return max(0.0, c - s), min(1.0, c + s)


# ── Report helpers ────────────────────────────────────────────────────────────

def fmt_pct(v) -> str:
    return "  —   " if v is None else f"{v:5.1f}%"

def fmt_time(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


# ── Full report ───────────────────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    W = 120
    sep = "═" * W

    def sort_key(r):
        if r["error"]:
            return -1, 0
        pct = ((r.get("hybrid") or {}).get("v12_hybrid") or {}).get("accuracy_pct", 0)
        return 1, pct

    results = sorted(results, key=sort_key, reverse=True)

    print(f"\n{sep}")
    print(f"  MASTER EVAL — {len(results)} models — {Path(TEST_FILE).name}")
    print(f"  Hybrid strategy: V13.1 regex (loc/tech/comp) + model (sen/arr)")
    print(sep)

    # ── Summary table ─────────────────────────────────────────────────────────
    hdr = (f"  {'Rank':<4}  {'Model':<46}  {'Size':>5}  "
           f"{'Hybrid':>7}  {'Exp':>7}  {'Model-only':>10}  "
           f"{'Parse':>5}  {'Time':>9}  Status")
    print(hdr)
    print("  " + "─" * (W - 2))

    for rank, r in enumerate(results, 1):
        h   = r.get("hybrid") or {}
        v12 = h.get("v12_hybrid") or {}
        mo  = h.get("model_only") or {}
        hyb_pct = fmt_pct(v12.get("accuracy_pct"))
        mo_pct  = fmt_pct(mo.get("accuracy_pct"))
        parse   = mo.get("parse_fail", "?") if not r["error"] else "?"
        exp     = r.get("expected") or "—"
        t_total = fmt_time(r["t_eval"] + r["t_hybrid"]) if not r["error"] else "—"
        flag    = " (cached)" if r.get("reused") else ""
        status  = f"✓{flag}" if not r["error"] else f"✗ {r['error'][:38]}"

        print(f"  {rank:<4}  {r['display'][:46]:<46}  {r['size_gb']:>4.2f}G  "
              f"{hyb_pct}  {exp:>7}  {mo_pct:>10}  "
              f"{str(parse):>5}  {t_total:>9}  {status}")

    print(sep)

    # ── Per-model detail ───────────────────────────────────────────────────────
    for r in results:
        if r["error"] or not r.get("hybrid"):
            continue

        h   = r["hybrid"]
        v12 = h.get("v12_hybrid") or {}
        mo  = h.get("model_only") or {}
        re_ = h.get("regex_only") or {}
        hA  = h.get("hybrid_A") or {}

        print(f"\n  ┌─ {r['display']}")
        print(f"  │  {r['notes']}")
        print(f"  │")

        # Accuracy headline with 95% CI
        print(f"  │  {'Metric':<24}  {'Correct':>8}  {'Total':>6}  {'Acc':>7}  95% CI")
        print(f"  │  {'─'*24}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*18}")
        for label, data in [
            ("V12 Hybrid (prod)",     v12),
            ("Hybrid A (model+regex)", hA),
            ("Model only",            mo),
            ("Regex only",            re_),
        ]:
            correct = data.get("correct", 0)
            total   = data.get("total", 0)
            pct     = data.get("accuracy_pct", 0)
            lo, hi  = wilson_ci(correct, total)
            print(f"  │  {label:<24}  {correct:>8}  {total:>6}  {pct:>6.1f}%  "
                  f"[{lo*100:.1f}%, {hi*100:.1f}%]")

        # Per-field accuracy
        print(f"  │")
        print(f"  │  Field accuracy (V12 hybrid):")
        fa    = v12.get("field_accuracy") or {}
        mo_fa = mo.get("field_accuracy") or {}
        re_fa = re_.get("field_accuracy") or {}
        print(f"  │  {'Field':<8}  {'Hybrid':>7}  {'Model':>7}  {'Regex':>7}  Source")
        print(f"  │  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*12}")
        field_src = {"loc": "regex", "arr": "model", "sen": "model",
                     "tech": "regex", "comp": "regex"}
        for fld in ["loc", "arr", "sen", "tech", "comp"]:
            print(f"  │  {fld:<8}  {fa.get(fld,0):>6.1f}%  "
                  f"{mo_fa.get(fld,0):>6.1f}%  "
                  f"{re_fa.get(fld,0):>6.1f}%  ← {field_src[fld]}")

        # Per-label accuracy
        per_label = h.get("v12_per_label") or {}
        if per_label:
            print(f"  │")
            print(f"  │  Per-label (V12 hybrid):")
            print(f"  │  {'Label':<12}  {'Correct':>8}  {'Total':>6}  {'Acc':>7}")
            print(f"  │  {'─'*12}  {'─'*8}  {'─'*6}  {'─'*7}")
            for lbl in ["good_fit", "maybe", "bad_fit"]:
                d = per_label.get(lbl) or {}
                c = d.get("correct", 0)
                t = d.get("total", 0)
                a = c / t * 100 if t else 0
                print(f"  │  {lbl:<12}  {c:>8}  {t:>6}  {a:>6.1f}%")

        # Errors — JSON keys (from compute_hybrid_v13_1.py line 493):
        #   job_index (1-indexed), golden, predicted, diffs (list of strings)
        errors = h.get("v12_errors") or []
        if errors:
            print(f"  │")
            print(f"  │  V12 Hybrid errors ({len(errors)}):")
            for e in errors:
                diffs  = "  ".join(e.get("diffs") or [])
                idx    = e.get("job_index", "?")
                g_lbl  = e.get("golden", "?")
                p_lbl  = e.get("predicted", "?")
                print(f"  │    Job {idx:>3}: {g_lbl} → {p_lbl}  [{diffs}]")

        # Timing
        print(f"  │")
        print(f"  │  Timing: eval {fmt_time(r['t_eval'])} + "
              f"hybrid {fmt_time(r['t_hybrid'])} = "
              f"{fmt_time(r['t_eval'] + r['t_hybrid'])} total")
        print(f"  └" + "─" * (W - 4))

    # ── Cross-model field comparison ───────────────────────────────────────────
    valid = [r for r in results if r.get("hybrid") and not r["error"]]
    if len(valid) > 1:
        print(f"\n{sep}")
        print(f"  FIELD ACCURACY COMPARISON — V12 HYBRID  (regex: loc/tech/comp  model: arr/sen)")
        print(sep)
        print(f"  {'Model':<46}  {'loc':>6}  {'arr':>6}  {'sen':>6}  "
              f"{'tech':>6}  {'comp':>6}  {'hybrid':>7}  {'MO':>6}")
        print(f"  {'─'*46}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")
        for r in valid:
            v12 = r["hybrid"].get("v12_hybrid") or {}
            mo  = r["hybrid"].get("model_only") or {}
            fa  = v12.get("field_accuracy") or {}
            pct = v12.get("accuracy_pct", 0)
            mo_pct = mo.get("accuracy_pct", 0)
            print(f"  {r['display'][:46]:<46}  "
                  f"{fa.get('loc',0):>5.1f}%  {fa.get('arr',0):>5.1f}%  "
                  f"{fa.get('sen',0):>5.1f}%  {fa.get('tech',0):>5.1f}%  "
                  f"{fa.get('comp',0):>5.1f}%  {pct:>6.1f}%  {mo_pct:>5.1f}%")
        print(sep)

    # ── Save full results to JSON ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "all_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results → {out_json}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Master eval — all models in parallel")
    parser.add_argument(
        "--workers", type=int, default=len(MODELS),
        help="Hard cap on concurrent models (default: all; RAM budget is the real limit)",
    )
    parser.add_argument(
        "--ram-reserve", type=float, default=0, metavar="GB",
        help="GB to keep free for OS (default: 15%% of total RAM, min 4 GB)",
    )
    parser.add_argument("--only",  nargs="+", metavar="NAME",
                        help="Run only these model names")
    parser.add_argument("--skip",  nargs="+", metavar="NAME",
                        help="Skip these model names")
    parser.add_argument("--reuse", action="store_true",
                        help="Load cached hybrid_summary.json instead of re-running")
    args = parser.parse_args()

    models: list[Model] = list(MODELS)
    if args.only:
        models = [m for m in models if m.name in args.only]
        unknown = set(args.only) - {m.name for m in MODELS}
        if unknown:
            print(f"WARNING: unknown model names: {unknown}")
    if args.skip:
        models = [m for m in models if m.name not in args.skip]

    runnable: list[Model] = []
    skipped:  list[tuple[Model, str]] = []
    for m in models:
        ok, reason = check_available(m)
        (runnable if ok else skipped).append(m if ok else (m, reason))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── RAM budget ────────────────────────────────────────────────────────────
    global _RAM_BUDGET_GB
    total_ram = _total_ram_gb()
    ram_reserve = args.ram_reserve if args.ram_reserve > 0 else max(4.0, total_ram * 0.15)
    _RAM_BUDGET_GB = max(1.0, total_ram - ram_reserve)

    w = min(args.workers, len(runnable)) if runnable else 1
    print(f"\nMaster Eval — {len(runnable)} models to run, {len(skipped)} skipped")
    print(f"RAM: {_RAM_BUDGET_GB:.0f} GB budget "
          f"({total_ram:.0f} GB total − {ram_reserve:.0f} GB reserved)  "
          f"|  Workers cap: {w}  |  Reuse: {args.reuse}")
    print(f"Test set: {Path(TEST_FILE).name}")

    if runnable:
        print(f"\n  {'Model':<32}  {'Peak RAM':>9}  {'Chunks':>6}")
        print(f"  {'─'*32}  {'─'*9}  {'─'*6}")
        for m in runnable:
            pr = peak_ram_gb(m)
            label = "1 (GGUF)" if m.is_gguf else str(m.chunks)
            fits  = "✓" if pr <= _RAM_BUDGET_GB else "⚠ exceeds budget"
            print(f"  {m.name:<32}  {pr:>7.1f} GB  {label:>6}  {fits}")
        print()

    if skipped:
        print("Skipped (files missing):")
        for m, reason in skipped:
            print(f"  {m.name}: {reason}")
    print()

    t_wall = time.time()
    results: list[dict] = []

    # Thread pool size = workers cap; actual concurrency is RAM-gated inside run_model()
    with ThreadPoolExecutor(max_workers=w) as pool:
        futures = {pool.submit(run_model, m, args.reuse): m for m in runnable}
        for fut in as_completed(futures):
            results.append(fut.result())

    for m, reason in skipped:
        results.append({
            "name":    m.name,   "display": m.display,
            "size_gb": m.size_gb, "expected": m.expected, "notes": m.notes,
            "error":   f"SKIPPED — {reason}",
            "t_eval":  0.0, "t_hybrid": 0.0, "hybrid": None, "reused": False,
        })

    elapsed = time.time() - t_wall
    print(f"\nAll evals finished in {fmt_time(elapsed)} wall time\n")

    print_report(results)


if __name__ == "__main__":
    main()

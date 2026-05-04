#!/usr/bin/env python3
"""
V16 Re-label using local MLX Gemma 4 model — FIXED VERSION.
Manually formats prompts to avoid Gemma 4 thinking mode issues.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.models.cache import make_prompt_cache


def greedy_sampler(logits: mx.array):
    return mx.argmax(logits, axis=-1)


def start_progress_monitor(remaining: int):
    shared = {"done": 0, "running": True}
    start = time.time()

    def monitor():
        while shared["running"]:
            time.sleep(30)
            done = shared["done"]
            elapsed = (time.time() - start) / 60
            if elapsed == 0:
                elapsed = 0.001
            speed = done / elapsed
            pct = (done / remaining * 100) if remaining > 0 else 0
            header = (
                f"{'Worker':<12} {'Done':>6} {'Total':>6} {'%':>6} "
                f"{'Jobs/min':>10} {'Status':>10}"
            )
            print(f"\n{header}", flush=True)
            print("-" * len(header), flush=True)
            print(
                f"{'mlx_main':<12} {done:>6} {remaining:>6} {pct:>6.1f} "
                f"{speed:>10.2f} {'RUNNING':>10}",
                flush=True,
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return shared, thread


def parse_response(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    text = text.strip()
    if text.count("{") == 0:
        return None
    if text.count("{") > text.count("}"):
        text += "}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def deep_copy_cache(cache):
    new_cache = []
    for c in cache:
        cls = type(c)
        new_c = cls.__new__(cls)
        state = c.state
        if isinstance(state, tuple):
            new_state = tuple(mx.array(s) for s in state)
        elif isinstance(state, list):
            new_state = [mx.array(s) for s in state]
        else:
            new_state = mx.array(state)
        new_c.state = new_state
        if hasattr(c, "meta_state"):
            new_c.meta_state = c.meta_state
        new_cache.append(new_c)
    return new_cache


def find_common_prefix_length(tokenizer, prompts: list[str]) -> int:
    token_lists = [tokenizer.encode(p) for p in prompts]
    if not token_lists:
        return 0
    prefix_len = 0
    min_len = min(len(t) for t in token_lists)
    while prefix_len < min_len:
        if all(t[prefix_len] == token_lists[0][prefix_len] for t in token_lists):
            prefix_len += 1
        else:
            break
    return prefix_len


def build_static_cache(model, tokenizer, prefix_tokens: list[int]):
    cache = make_prompt_cache(model)
    y = mx.array(prefix_tokens)
    model(y[None], cache=cache)
    mx.eval([c.state for c in cache])
    return cache


def clean_jd_for_teacher(jd_text: str) -> str:
    for suffix in [
        "\n\nBegin your response with {",
        "\n\nBegin response with {",
        "\nBegin your response with {",
        "\nBegin response with {",
        "Respond with JSON only.",
        "Classify this job.",
    ]:
        idx = jd_text.find(suffix)
        if idx != -1:
            jd_text = jd_text[:idx].strip()
    return jd_text


def build_prompt_manual(prompt_template: str, job: dict) -> str:
    """Manual prompt formatting — bypass Gemma 4 chat template thinking issues."""
    content = prompt_template
    content = content.replace("{{job_title}}", job.get("title", ""))
    content = content.replace("{{job_location}}", job.get("job_location", ""))
    content = content.replace("{{jd_text}}", clean_jd_for_teacher(job.get("jd_text", "")))

    # Build a simple instruction format instead of chat template
    # This avoids the <|channel>thought mechanism entirely
    system_msg = "You are a detail-oriented job classifier. Respond with JSON only."
    user_msg = content

    # For Gemma 4, use a clean instruction format without thinking tokens
    prompt = f"{system_msg}\n\n{user_msg}\n\nRespond with JSON only. Begin with '{{' and end with '}}'."
    return prompt + "\n{"  # Add { prefill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/Users/faisal/MLX Models/gemma-4-31B-it-oQ4",
        help="Local MLX model directory (or HF hub id)",
    )
    parser.add_argument("--input", required=True, help="Input JSONL with job fields")
    parser.add_argument(
        "--output", required=True, help="Output JSONL with v16_teacher labels"
    )
    parser.add_argument(
        "--prompt",
        default="versions/v16/prompts/teacher.txt",
        help="Teacher prompt template",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max generation tokens")
    parser.add_argument(
        "--max-total", type=int, default=0, help="Stop after N total jobs (0=unlimited)"
    )
    args = parser.parse_args()

    prompt_template = Path(args.prompt).read_text()
    out_path = Path(args.output)

    print(f"Loading MLX model: {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    with open(args.input) as f:
        all_jobs = [json.loads(l) for l in f]

    # ── Resume: skip jobs already done by ANY worker ─────────────
    done_indices = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done_indices.add(d.get("index", d.get("job_id")))
    for existing_batch in Path("versions/v16/data/full_relabel").glob("batch_*.jsonl"):
        if existing_batch.name == out_path.name:
            continue
        try:
            with open(existing_batch) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        done_indices.add(d.get("index", d.get("job_id")))
        except Exception:
            pass
    for existing_batch in Path("versions/v16/data/ollama_100").glob("batch_*.jsonl"):
        try:
            with open(existing_batch) as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        done_indices.add(d.get("index", d.get("job_id")))
        except Exception:
            pass

    total_input = len(all_jobs)
    all_jobs = [j for j in all_jobs if j.get("index") not in done_indices]
    already_done = total_input - len(all_jobs)

    if not all_jobs:
        print(f"Nothing left to do. ({already_done}/{total_input} already done)")
        return

    if args.max_total > 0:
        all_jobs = all_jobs[: args.max_total]

    remaining = len(all_jobs)

    if out_path.exists():
        print(f"Removing stale output file: {out_path}")
        out_path.unlink()

    print(f"Already done:     {already_done}/{total_input}")
    print(f"Remaining now:    {remaining}")
    print(f"Output file:      {out_path}")
    print(f"Batch size:       {args.batch_size}")
    print("-" * 60)

    shared_progress, monitor_thread = start_progress_monitor(remaining)
    parse_fails = 0
    start_time = time.time()
    header = (
        f"{'Worker':<12} {'Done':>6} {'Total':>6} {'%':>6} "
        f"{'Jobs/min':>10} {'Status':>10}"
    )
    print(f"\n{header}", flush=True)
    print("-" * len(header), flush=True)

    for batch_start in range(0, remaining, args.batch_size):
        batch = all_jobs[batch_start : batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1

        # Use manual prompt format (no chat template thinking issues)
        full_prompts = [build_prompt_manual(prompt_template, job) for job in batch]
        full_tokens_list = [tokenizer.encode(p) for p in full_prompts]

        prefix_len = find_common_prefix_length(tokenizer, full_prompts)
        if prefix_len > 0:
            prefix_tokens = full_tokens_list[0][:prefix_len]
            static_cache = build_static_cache(model, tokenizer, prefix_tokens)
            prompt_tokens_list = [tokens[prefix_len:] for tokens in full_tokens_list]
            prompt_caches = [deep_copy_cache(static_cache) for _ in batch]
        else:
            prompt_tokens_list = full_tokens_list
            prompt_caches = None

        t_gen = time.time()
        response = batch_generate(
            model,
            tokenizer,
            prompts=prompt_tokens_list,
            prompt_caches=prompt_caches,
            max_tokens=args.max_tokens,
            sampler=greedy_sampler,
            prefill_step_size=4096,
            verbose=False,
        )
        gen_time = time.time() - t_gen

        with open(out_path, "a") as out_f:
            for i, job in enumerate(batch):
                raw_text = response.texts[i]
                # Strip potential leading { if it was part of prefill
                if raw_text.startswith("{"):
                    raw_text = raw_text[1:]
                parsed = parse_response(raw_text)

                out = dict(job)
                if parsed is None:
                    parse_fails += 1
                    for field in [
                        "loc_raw", "loc", "arr_raw", "arr",
                        "sen_raw", "sen", "tech_raw", "tech",
                        "comp_raw", "comp",
                    ]:
                        out[f"v16_{field}"] = None
                    out["_v16_raw_response"] = raw_text[:500]
                    out["_v16_parse_error"] = True
                else:
                    for field in [
                        "loc_raw", "loc", "arr_raw", "arr",
                        "sen_raw", "sen", "tech_raw", "tech",
                        "comp_raw", "comp",
                    ]:
                        out[f"v16_{field}"] = parsed.get(field)
                    out["_v16_raw_response"] = raw_text[:500]
                    out["_v16_parse_error"] = False
                out_f.write(json.dumps(out) + "\n")
                out_f.flush()
                shared_progress["done"] += 1

        elapsed = time.time() - start_time
        done = shared_progress["done"]
        speed = done / elapsed if elapsed else 0
        eta = (remaining - done) / speed if speed > 0 else 0
        status = "RUNNING"
        print(
            f"Batch {batch_num:>3}: +{len(batch)} in {gen_time:.1f}s | "
            f"Done: {done}/{remaining} | "
            f"Speed: {speed:.2f} jobs/s ({60*speed:.1f}/min) | "
            f"ETA: {eta/60:.1f}min | "
            f"ParseFails: {parse_fails}",
            flush=True,
        )

    shared_progress["running"] = False
    monitor_thread.join(timeout=1)

    print("-" * 60, flush=True)
    print(
        f"Done. Processed {shared_progress['done']} jobs ({parse_fails} parse fails)",
        flush=True,
    )
    print(f"Output: {out_path}", flush=True)
    print(f"Total time: {time.time()-start_time:.1f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Partial results are safe in output file.")
        sys.exit(1)

#!/usr/bin/env python3
"""
Workaround for omlx oQ quantization bug on Qwen3 models.

The bug: mlx_lm's quantize_model() adds a set to the config dict,
which json.dump() can't serialize. This script patches the config
to convert any sets to lists before saving.

Usage:
    .venv/bin/python3 finetune/quantize_oq_fix.py \
        --model ~/merged_v15_4B \
        --output ~/qwen3_4B_v15_oQ4 \
        --bits 4
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load, save, quantize_model


def fix_sets_in_dict(d):
    """Recursively convert sets to sorted lists in a dict."""
    if isinstance(d, dict):
        return {k: fix_sets_in_dict(v) for k, v in d.items()}
    elif isinstance(d, set):
        return sorted(list(d))
    elif isinstance(d, list):
        return [fix_sets_in_dict(v) for v in d]
    return d


def main():
    parser = argparse.ArgumentParser(description="Quantize with set serialization fix")
    parser.add_argument("--model", required=True, help="Path to merged model")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    # Read config
    config_path = Path(args.model) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    print(f"Quantizing to {args.bits}-bit (group_size={args.group_size})...")
    model, config = quantize_model(
        model, config, group_size=args.group_size, bits=args.bits
    )

    # FIX: convert any sets to lists before saving
    config = fix_sets_in_dict(config)

    print(f"Saving to {args.output}...")
    save(args.output, args.model, model, tokenizer, config)

    # Verify the output
    out_path = Path(args.output)
    total_size = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    print(f"\nDone! Output: {args.output} ({total_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()

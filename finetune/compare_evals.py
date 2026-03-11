#!/usr/bin/env python3
"""
Compare multiple V7 eval summary JSONs side by side.

Usage:
  python3 finetune/compare_evals.py eval_results/adapters_v7/*.summary.json
  python3 finetune/compare_evals.py eval_results/adapters_v7/summary1.json eval_results/adapters_v7_1.5B/summary2.json
"""

import json
import sys
from pathlib import Path


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 finetune/compare_evals.py <summary1.json> [summary2.json] ...")
        sys.exit(1)

    summaries = []
    for path in sys.argv[1:]:
        s = load_summary(path)
        s["_file"] = Path(path).name
        summaries.append(s)

    # Sort by label accuracy descending
    summaries.sort(key=lambda s: s.get("label_accuracy", 0), reverse=True)

    # Header
    W = 90
    print("=" * W)
    print("  V7 EVAL COMPARISON")
    print("=" * W)

    # Table header
    cols = ["Run", "Label%", "loc", "arr", "sen", "tech", "comp", "gf", "maybe", "bad", "Parse", "Invalid"]
    widths = [35, 7, 6, 6, 6, 6, 6, 5, 6, 5, 6, 8]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * W)

    for s in summaries:
        fa = s.get("field_accuracy", {})
        pl = s.get("per_label", {})
        name = f"{s.get('checkpoint', '?')} ({s.get('model', '?').split('/')[-1][:15]})"

        row = [
            f"{name[:35]:<35}",
            f"{s.get('label_accuracy', 0):>5.1f}%",
            f"{fa.get('loc', 0):>4.1f}%",
            f"{fa.get('arr', 0):>4.1f}%",
            f"{fa.get('sen', 0):>4.1f}%",
            f"{fa.get('tech', 0):>4.1f}%",
            f"{fa.get('comp', 0):>4.1f}%",
            f"{pl.get('good_fit', 0):>3.0f}%",
            f"{pl.get('maybe', 0):>4.0f}%",
            f"{pl.get('bad_fit', 0):>3.0f}%",
            f"{s.get('parse_failures', 0):>4}",
            f"{s.get('invalid_tokens', 0):>6}",
        ]
        print("  ".join(row))

    print("=" * W)

    # Best model callout
    best = summaries[0]
    print(f"\n  Best: {best.get('checkpoint', '?')} ({best.get('model', '?').split('/')[-1]})")
    print(f"  Label accuracy: {best.get('label_accuracy', 0):.1f}%")
    print(f"  Fields: loc={best.get('field_accuracy', {}).get('loc', 0):.1f}%"
          f"  tech={best.get('field_accuracy', {}).get('tech', 0):.1f}%"
          f"  comp={best.get('field_accuracy', {}).get('comp', 0):.1f}%")


if __name__ == "__main__":
    main()

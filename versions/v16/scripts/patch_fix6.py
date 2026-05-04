#!/usr/bin/env python3
"""Replace fix4 sen rule text with fix6 text in MLX chat-format JSONL files."""

import sys
import json
from pathlib import Path

OLD_RULE = "  Decide using the job TITLE only. Ignore experience language in the description."
NEW_RULE = "  Decide using the job TITLE primarily. If the title is ambiguous (e.g. \"Engineer\" with no explicit L1/L3 keywords), use the Description as fallback for years of experience, leadership scope, or responsibility language."

def fix_file(path: str):
    updated = 0
    lines = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages", [])
            for m in msgs:
                if m["role"] == "user":
                    content = m["content"]
                    if OLD_RULE in content:
                        m["content"] = content.replace(OLD_RULE, NEW_RULE, 1)
                        updated += 1
            lines.append(json.dumps(d, ensure_ascii=False))
    
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"Updated {updated} records in {path}")
    return updated


def main():
    files = sys.argv[1:] or ["versions/v16/data/mlx/train.jsonl", "versions/v16/data/mlx/valid.jsonl"]
    total = 0
    for f in files:
        if Path(f).exists():
            total += fix_file(f)
        else:
            print(f"Skipping {f} — does not exist. Run copy-v15-data first.")
    print(f"\nTotal updated: {total}")

if __name__ == "__main__":
    main()

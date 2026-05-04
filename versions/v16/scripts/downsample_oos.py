#!/usr/bin/env python3
"""
Downsample OOS examples in V14 training data for V15.

Selects the most diverse OOS examples by comparing JD text similarity,
keeping edge cases (jobs that mention tracked tech keywords but are correctly OOS)
and removing redundant examples with similar JD patterns.

Usage:
    python3 finetune/downsample_oos_v15.py \
        --input data/v14/train.jsonl \
        --keep 150 \
        --output data/v15/train_rebalanced.jsonl \
        --report data/v15/oos_downsample_report.json
"""

import json
import argparse
import random
import re
from collections import Counter
from pathlib import Path


def extract_labels(messages):
    """Extract the assistant's token predictions from a training example."""
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
    return None


def extract_jd_text(messages):
    """Extract JD text from the user message."""
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            # JD is after "Description:" in the user message
            if "Description:" in content:
                return content.split("Description:")[-1].strip()
    return ""


def extract_title(messages):
    """Extract job title from the user message."""
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if "Title:" in content:
                after_title = content.split("Title:")[-1]
                # Title is between "Title:" and "Location:"
                if "Location:" in after_title:
                    return after_title.split("Location:")[0].strip()
                return after_title[:100].strip()
    return ""


# Regex patterns for tracked tech — whole word matches only
# These detect OOS jobs that are near the boundary (mention tracked tech but correctly OOS)
TRACKED_TECH_PATTERNS = [
    r'\bnode\.?js\b', r'\bnodejs\b', r'\bnode\.js\b',
    r'\breact\b', r'\breact\.js\b', r'\breactjs\b', r'\breact native\b',
    r'\bjavascript\b', r'\btypescript\b',
    r'\bmachine learning\b', r'\bdeep learning\b',
    r'\bllm\b', r'\bllms\b', r'\bpytorch\b', r'\btensorflow\b',
    r'\bgenerative ai\b', r'\bnlp\b', r'\bneural network\b',
    r'\bartificial intelligence\b',
]

# Compile for speed
_TECH_REGEXES = [re.compile(p, re.IGNORECASE) for p in TRACKED_TECH_PATTERNS]


def has_tracked_tech_mention(jd_text):
    """Check if JD mentions tracked tech keywords (edge case OOS)."""
    return any(rx.search(jd_text) for rx in _TECH_REGEXES)


def simple_text_hash(text, n=3):
    """Create a set of character n-grams for rough similarity."""
    text = re.sub(r'\s+', ' ', text.lower().strip())[:500]  # First 500 chars
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Downsample OOS examples for V15")
    parser.add_argument("--input", required=True, help="Input V14 train.jsonl")
    parser.add_argument("--keep", type=int, default=150, help="Number of OOS examples to keep")
    parser.add_argument("--output", required=True, help="Output rebalanced train.jsonl")
    parser.add_argument("--report", default=None, help="Output JSON report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all training examples
    examples = []
    with open(args.input) as f:
        for line in f:
            examples.append(json.loads(line))

    # Separate OOS from non-OOS
    oos_examples = []
    non_oos_examples = []

    for i, ex in enumerate(examples):
        labels = extract_labels(ex["messages"])
        if labels is None:
            non_oos_examples.append((i, ex))  # Can't parse → keep
            continue

        tech = labels.get("tech", [])
        if isinstance(tech, list) and tech == ["OOS"]:
            jd = extract_jd_text(ex["messages"])
            title = extract_title(ex["messages"])
            oos_examples.append({
                "index": i,
                "example": ex,
                "jd_text": jd,
                "title": title,
                "labels": labels,
                "is_edge_case": has_tracked_tech_mention(jd),
                "text_hash": simple_text_hash(jd),
            })
        else:
            non_oos_examples.append((i, ex))

    print(f"Total training examples: {len(examples)}")
    print(f"OOS examples: {len(oos_examples)}")
    print(f"Non-OOS examples: {len(non_oos_examples)}")
    print(f"Target OOS count: {args.keep}")

    if len(oos_examples) <= args.keep:
        print(f"Already at or below target — no downsampling needed.")
        # Just copy input to output
        with open(args.output, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return

    # Priority 1: Keep ALL edge cases (OOS jobs that mention tracked tech)
    edge_cases = [o for o in oos_examples if o["is_edge_case"]]
    non_edge_oos = [o for o in oos_examples if not o["is_edge_case"]]

    print(f"Edge cases (mention tracked tech, correctly OOS): {len(edge_cases)}")
    print(f"Non-edge OOS: {len(non_edge_oos)}")

    # Start with all edge cases
    kept = list(edge_cases)
    remaining_budget = args.keep - len(kept)

    if remaining_budget <= 0:
        print(f"WARNING: More edge cases ({len(edge_cases)}) than budget ({args.keep})")
        # Keep all edge cases even if over budget
        remaining_budget = 0
    else:
        # Priority 2: From non-edge OOS, select diverse examples using greedy diversity
        # Compute pairwise similarity and greedily pick the most different ones
        candidates = list(non_edge_oos)
        random.shuffle(candidates)  # Randomize initial order

        # Greedy diversity selection
        selected = []
        selected_hashes = []

        for candidate in candidates:
            if len(selected) >= remaining_budget:
                break

            # Check similarity to already selected examples
            max_sim = 0.0
            for sel_hash in selected_hashes:
                sim = jaccard_similarity(candidate["text_hash"], sel_hash)
                max_sim = max(max_sim, sim)

            # Keep if sufficiently different from all selected
            if max_sim < 0.7 or len(selected) < 10:  # Always keep first 10
                selected.append(candidate)
                selected_hashes.append(candidate["text_hash"])

        # If we still need more (diversity filter was too strict), fill randomly
        if len(selected) < remaining_budget:
            unselected = [c for c in candidates if c not in selected]
            random.shuffle(unselected)
            selected.extend(unselected[:remaining_budget - len(selected)])

        kept.extend(selected)

    removed = [o for o in oos_examples if o not in kept]

    print(f"\nFinal OOS kept: {len(kept)} (edge: {len(edge_cases)}, diverse: {len(kept) - len(edge_cases)})")
    print(f"OOS removed: {len(removed)}")

    # Combine non-OOS + kept OOS, preserving original order
    kept_indices = set(o["index"] for o in kept)
    removed_indices = set(o["index"] for o in removed)

    output_examples = []
    for i, ex in enumerate(examples):
        if i not in removed_indices:
            output_examples.append(ex)

    print(f"Output training examples: {len(output_examples)}")

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in output_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Distribution check
    print("\n--- Distribution Check ---")
    output_tech = Counter()
    for ex in output_examples:
        labels = extract_labels(ex["messages"])
        if labels:
            tech = labels.get("tech", [])
            if isinstance(tech, list):
                key = tuple(sorted(tech))
                output_tech[key] += 1

    total = len(output_examples)
    oos_count = output_tech.get(("OOS",), 0)
    print(f"OOS: {oos_count}/{total} ({oos_count/total*100:.1f}%)")
    print(f"Non-OOS: {total - oos_count}/{total} ({(total-oos_count)/total*100:.1f}%)")

    # Write report
    if args.report:
        report = {
            "input_total": len(examples),
            "input_oos": len(oos_examples),
            "input_non_oos": len(non_oos_examples),
            "edge_cases_kept": len(edge_cases),
            "diverse_kept": len(kept) - len(edge_cases),
            "total_oos_kept": len(kept),
            "total_oos_removed": len(removed),
            "output_total": len(output_examples),
            "output_oos_pct": round(oos_count / total * 100, 1) if total > 0 else 0,
            "removed_titles": [o["title"][:80] for o in removed[:50]],
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.report}")

    print("\nDone!")


if __name__ == "__main__":
    main()

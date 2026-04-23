"""
Results viewer — show + compare master eval results.

Usage:
    harness results show                       # Latest leaderboard
    harness results show --model v14-4B        # Filter by model
    harness results compare v14-4B v14-0.6B    # Side-by-side
"""
import json
from typing import List, Optional
import typer

from finetune.constants import REPO

app = typer.Typer(no_args_is_help=True, help="View + compare eval results.")

RESULTS_DIR = REPO / "versions/legacy/master_eval/results"


@app.command()
def show(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model name"),
):
    """Show leaderboard from master eval."""
    results_file = RESULTS_DIR / "all_results.json"
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        typer.echo("✗ No results found. Run `harness master-eval run` first.", err=True)
        raise typer.Exit(1)

    models = data.get("models", [])
    if model:
        models = [m for m in models if model.lower() in m.get("name", "").lower()]

    if not models:
        typer.echo(f"✗ No models matching '{model}'", err=True)
        raise typer.Exit(1)

    # Sort by hybrid desc (skip if single model)
    if len(models) > 1:
        models.sort(key=lambda m: m.get("hybrid", 0), reverse=True)

    print("\n" + "=" * 110)
    print("  EVAL RESULTS" + (f" — filtered: {model}" if model else ""))
    print("=" * 110)

    # Header
    cols = ["Model", "Size", "Hybrid", "Model-only", "Parse", "gf", "maybe", "bad", "Notes"]
    widths = [30, 8, 9, 12, 7, 6, 7, 6, 20]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * 110)

    for m in models:
        per_label = m.get("per_label", {})
        row = [
            m.get("name", "?")[:30],
            str(m.get("size_gb", "?")),
            f"{m.get('hybrid', 0):.1f}%",
            f"{m.get('model_only', 0):.1f}%",
            str(m.get("parse_failures", 0)),
            f"{per_label.get('good_fit', 0):.0f}%",
            f"{per_label.get('maybe', 0):.0f}%",
            f"{per_label.get('bad_fit', 0):.0f}%",
            m.get("notes", "")[:20],
        ]
        print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)))

    print("=" * 110)
    print(f"\n  Source: {results_file}")


@app.command()
def compare(
    models: List[str] = typer.Argument(..., help="Model names to compare (e.g. v14-4B v14-0.6B)"),
):
    """Compare specific models side-by-side."""
    results_file = RESULTS_DIR / "all_results.json"
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        typer.echo("✗ No results found. Run `harness master-eval run` first.", err=True)
        raise typer.Exit(1)

    all_models = {m["name"]: m for m in data.get("models", [])}
    selected = []
    for name in models:
        if name in all_models:
            selected.append(all_models[name])
        else:
            # Try case-insensitive match
            match = next((m for m in data["models"] if name.lower() in m["name"].lower()), None)
            if match:
                selected.append(match)
            else:
                typer.echo(f"✗ Model '{name}' not found", err=True)
                raise typer.Exit(1)

    print("\n" + "=" * 120)
    print("  MODEL COMPARISON")
    print("=" * 120)

    # Side-by-side metrics
    print("\n  Key Metrics:")
    print("  " + "-" * 50)
    metrics = [
        ("Hybrid acc", "hybrid"),
        ("Model-only", "model_only"),
        ("Parse fails", "parse_failures"),
        ("Errors", "errors"),
        ("Size (GB)", "size_gb"),
    ]
    for label, key in metrics:
        row = [label] + [str(m.get(key, "?")) for m in selected]
        print("    " + "  vs  ".join(f"{v:>12}" for v in row))

    # Per-label breakdown
    print("\n  Per-Label Accuracy:")
    print("  " + "-" * 50)
    for label in ["good_fit", "maybe", "bad_fit"]:
        row = [label] + [f"{m.get('per_label', {}).get(label, 0):.0f}%" for m in selected]
        print("    " + "  vs  ".join(f"{v:>12}" for v in row))

    print("\n  " + "=" * 120)

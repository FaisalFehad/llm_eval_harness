"""
Multi-model eval — run ALL models, produce leaderboard.

Usage:
    harness master-eval run                    # Eval all models
    harness master-eval run --workers 8        # 8 concurrent
    harness master-eval run --only v14-4B      # Single model
    harness master-eval results                # Show leaderboard
"""
import json
import subprocess
from typing import List, Optional
import typer

from finetune.constants import REPO, PYTHON

app = typer.Typer(no_args_is_help=True, help="Multi-model eval + leaderboard.")

SCRIPT = REPO / "versions/legacy/master_eval/run_all.py"
RESULTS_DIR = REPO / "versions/legacy/master_eval/results"


@app.command()
def run(
    workers: int = typer.Option(4, "--workers", "-w", help="Max concurrent models"),
    only: Optional[List[str]] = typer.Option(None, "--only", help="Run only these models"),
    skip: Optional[List[str]] = typer.Option(None, "--skip", help="Skip these models"),
    reuse: bool = typer.Option(False, "--reuse", help="Skip models with existing results"),
):
    """Eval all models, produce leaderboard."""
    cmd = [PYTHON, str(SCRIPT), "--workers", str(workers)]
    if only:
        cmd += ["--only"] + only
    if skip:
        cmd += ["--skip"] + skip
    if reuse:
        cmd.append("--reuse")

    typer.echo(f"→ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.command()
def results():
    """Show leaderboard from last master eval."""
    results_file = RESULTS_DIR / "all_results.json"
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        typer.echo("✗ No results found. Run `harness master-eval run` first.", err=True)
        raise typer.Exit(1)

    # Print leaderboard table
    print("\n" + "=" * 100)
    print("  MASTER EVAL LEADERBOARD")
    print("=" * 100)

    # Header
    cols = ["Model", "Size", "Hybrid", "Model-only", "Parse", "Errors"]
    widths = [35, 10, 10, 12, 8, 8]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header)
    print("-" * 100)

    # Sort by hybrid desc
    models = data.get("models", [])
    models.sort(key=lambda m: m.get("hybrid", 0), reverse=True)

    for m in models:
        row = [
            m.get("name", "?")[:35],
            m.get("size_gb", "?"),
            f"{m.get('hybrid', 0):.1f}%",
            f"{m.get('model_only', 0):.1f}%",
            str(m.get("parse_failures", 0)),
            str(m.get("errors", 0)),
        ]
        print("  ".join(f"{c:<{w}}" for c, w in zip(row, widths)))

    print("=" * 100)
    print(f"\n  Results: {results_file}")

"""Regex + model hybrid scoring."""
import subprocess
from pathlib import Path
from typing import Optional
import typer

from finetune.registry import SHARED_TEST_FILE, default_version, get as get_pipeline

app = typer.Typer(no_args_is_help=True, help="Hybrid (regex + model) scoring.")

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")


def _default_regex_version(version: Optional[str]) -> str:
    """Fall back to the pipeline's regex_version if no explicit --regex-version is given."""
    try:
        return get_pipeline(version).regex_version
    except KeyError:
        return "v13_1"


@app.command()
def score(
    predictions: Path = typer.Option(..., "--predictions", "-p"),
    test_file: Path = typer.Option(SHARED_TEST_FILE, "--test-file", "-t"),
    regex_version: str = typer.Option("v13_1", "--regex-version", "-r", help="v13 or v13_1 (V13.1 regex is stricter)"),
    output: Path = typer.Option(None, "--output", "-o"),
    v12_compat: bool = typer.Option(True, "--v12/--no-v12", help="Use V12 hybrid evaluator semantics"),
):
    """Compute hybrid (regex + model) accuracy from a predictions JSONL."""
    script = REPO / f"finetune/compute_hybrid_{regex_version}.py"
    if not script.exists():
        typer.echo(f"\u2717 Hybrid script not found: {script}", err=True)
        raise typer.Exit(1)

    cmd = [PY, str(script),
           "--test-file", str(test_file),
           "--predictions", str(predictions)]
    if v12_compat:
        cmd.append("--v12")
    if output:
        cmd += ["--output", str(output)]
    subprocess.run(cmd, check=True)


@app.command()
def regex_baseline(
    test_file: Path = typer.Option(SHARED_TEST_FILE, "--test-file", "-t"),
    regex_version: str = typer.Option("v13_1", "--regex-version", "-r"),
):
    """Run regex-only baseline (no model) to measure ceiling of deterministic rules."""
    script = REPO / f"finetune/deterministic_baseline_{regex_version}.py"
    subprocess.run([PY, str(script), "--test-file", str(test_file)], check=True)

"""Data pipeline: generate, label, audit, merge, downsample."""
import subprocess
from pathlib import Path
import typer

from finetune.registry import base_key, default_version

app = typer.Typer(no_args_is_help=True, help="Data pipeline operations.")

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")
NPX = "npx"


@app.command()
def generate(
    version: str = typer.Option(None, "--version", "-v",
        help=f"Training version. Default: ${{HARNESS_VERSION:-{default_version()}}}"),
    output: Path = typer.Option(..., "--output", "-o", help="Output JSONL path."),
    count: int = typer.Option(100, "--count", "-n", help="Number of synthetic JDs."),
):
    """Generate synthetic job descriptions (v15: generate_v15_data.py)."""
    key = base_key(version)
    script = REPO / f"finetune/generate_{key}_data.py"
    if not script.exists():
        typer.echo(f"\u2717 No generator for version={version} ({script})", err=True)
        raise typer.Exit(1)
    subprocess.run([PY, str(script), "--output", str(output), "--count", str(count)], check=True)


@app.command()
def label(
    input: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
):
    """Label JDs with the teacher model (gpt-4.1-mini, temp=0)."""
    subprocess.run([NPX, "tsx", "src/cli/label-jobs-v7.ts", "--input", str(input), "--output", str(output)],
                   check=True, cwd=REPO)


@app.command()
def audit(
    input: Path = typer.Option(..., "--input", "-i"),
    pre_label: bool = typer.Option(False, "--pre-label", help="Audit BEFORE labeling."),
    eval_set: Path = typer.Option(None, "--eval-set", help="Eval set to compare against."),
):
    """Run the audit script over a labeled JSONL file."""
    args = [NPX, "tsx", "src/cli/audit-training-data-v7.ts", "--input", str(input)]
    if pre_label:
        args.append("--pre-label")
    if eval_set:
        args += ["--eval-set", str(eval_set)]
    subprocess.run(args, check=True, cwd=REPO)


@app.command()
def merge(version: str = typer.Option(None, "--version", "-v",
    help=f"Training version. Default: ${{HARNESS_VERSION:-{default_version()}}}")):
    """Merge data sources (v15: merge_v15_data.py)."""
    script = REPO / f"finetune/merge_{base_key(version)}_data.py"
    subprocess.run([PY, str(script)], check=True)


@app.command()
def downsample(version: str = typer.Option(None, "--version", "-v",
    help=f"Training version. Default: ${{HARNESS_VERSION:-{default_version()}}}")):
    """Downsample OOS examples to balance the dataset (v15: downsample_oos_v15.py)."""
    script = REPO / f"finetune/downsample_oos_{base_key(version)}.py"
    subprocess.run([PY, str(script)], check=True)

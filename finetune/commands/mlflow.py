"""MLflow tracking + model registry."""
import subprocess
from pathlib import Path
from typing import Optional
import typer

from finetune.registry import base_key, default_version

app = typer.Typer(no_args_is_help=True, help="MLflow operations.")

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")


@app.command()
def log(version: Optional[str] = typer.Option(None, "--version", "-v",
    help=f"Training version. Default: ${{HARNESS_VERSION:-{default_version()}}}")):
    """Log an eval run to MLflow."""
    script = REPO / f"versions/{base_key(version)}/scripts/mlflow_eval.py"
    if not script.exists():
        typer.echo(f"\u2717 MLflow log script not found: {script}", err=True)
        raise typer.Exit(1)
    subprocess.run([PY, str(script)], check=True)


@app.command()
def model(version: Optional[str] = typer.Option(None, "--version", "-v",
    help=f"Training version. Default: ${{HARNESS_VERSION:-{default_version()}}}")):
    """Register a model in the MLflow registry."""
    script = REPO / f"versions/{base_key(version)}/scripts/mlflow_model.py"
    if not script.exists():
        typer.echo(f"\u2717 MLflow model script not found: {script}", err=True)
        raise typer.Exit(1)
    subprocess.run([PY, str(script)], check=True)


@app.command()
def ui(port: int = typer.Option(5000, "--port", "-p")):
    """Open the MLflow tracking UI (default port 5000)."""
    subprocess.run([PY, "-m", "mlflow", "ui", "--port", str(port)], check=True)

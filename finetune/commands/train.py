"""LoRA fine-tuning — reads LoRA config from the registry."""
import subprocess
from pathlib import Path
from typing import Optional
import typer

from finetune.registry import PIPELINES, default_version, get as get_pipeline

app = typer.Typer(no_args_is_help=True, help="LoRA training.")

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")


@app.command()
def lora(
    version: Optional[str] = typer.Option(
        None, "--version", "-v",
        help=f"Registered pipeline. Default: ${{HARNESS_VERSION:-{default_version()}}}. "
             f"Available: {sorted(PIPELINES)}"),
    config: Optional[Path] = typer.Option(None, "--config", "-c",
        help="Override LoRA config YAML (otherwise looked up from registry)."),
    log_file: Optional[Path] = typer.Option(None, "--log",
        help="Tee training output to a log file."),
):
    """Run `mlx_lm.lora` with the pipeline's LoRA config (or an override)."""
    if config is None:
        try:
            pipe = get_pipeline(version)
        except KeyError as e:
            typer.echo(f"\u2717 {e}", err=True)
            raise typer.Exit(1)
        if pipe.lora_config is None:
            typer.echo(f"\u2717 No lora_config in registry for '{version or default_version()}'. "
                       f"Pass --config <path>.", err=True)
            raise typer.Exit(1)
        config = REPO / pipe.lora_config

    if not config.exists():
        typer.echo(f"\u2717 Config not found: {config}", err=True)
        raise typer.Exit(1)

    cmd = [PY, "-m", "mlx_lm.lora", "--config", str(config)]
    typer.echo(f"\u2192 {' '.join(cmd)}")
    if log_file:
        subprocess.run(" ".join(cmd) + f" 2>&1 | tee {log_file}", shell=True, check=True)
    else:
        subprocess.run(cmd, check=True)

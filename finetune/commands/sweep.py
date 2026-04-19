"""Multi-checkpoint eval across iterations — reads sweep script from the registry."""
import subprocess
from pathlib import Path
from typing import List, Optional
import typer

from finetune.registry import PIPELINES, default_version, get as get_pipeline, base_key

app = typer.Typer(no_args_is_help=True, help="Sweep eval across checkpoints.")

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")


@app.command()
def run(
    version: Optional[str] = typer.Option(
        None, "--version", "-v",
        help=f"Registered pipeline. Default: ${{HARNESS_VERSION:-{default_version()}}}. "
             f"Available: {sorted(PIPELINES)}"),
    iters: Optional[List[int]] = typer.Option(None, "--iters", "-i",
        help="Specific iteration numbers to sweep (default: all checkpoints)."),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing"),
    script: Optional[Path] = typer.Option(None, "--script",
        help="Override sweep script path (otherwise looked up from registry)."),
):
    """Sweep all (or selected) checkpoints for a registered version."""
    if script is None:
        try:
            pipe = get_pipeline(version)
        except KeyError as e:
            typer.echo(f"\u2717 {e}", err=True)
            raise typer.Exit(1)
        # Variants (v15-oq6, etc.) share scripts with their base_version
        if pipe.sweep_script is None:
            resolved_key = base_key(version)
            typer.echo(f"\u2717 No sweep_script in registry for '{version or default_version()}' "
                       f"(base={resolved_key}). Pass --script <path>.", err=True)
            raise typer.Exit(1)
        script = REPO / pipe.sweep_script

    if not script.exists():
        typer.echo(f"\u2717 Sweep script not found: {script}", err=True)
        raise typer.Exit(1)

    cmd = [PY, str(script)]
    if iters:
        cmd += ["--iters", *[str(i) for i in iters]]
    if skip_existing:
        cmd.append("--skip-existing")
    typer.echo(f"\u2192 {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

"""Shared utilities for the finetune package."""
import os
import subprocess
from pathlib import Path
from typing import Optional

import typer
from finetune.registry import Pipeline, get as get_pipeline


def expand_path(p: str | Path | None) -> str | None:
    """Expand ~ in a path or return None if input is None."""
    return os.path.expanduser(str(p)) if p is not None else None


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command with check=True."""
    subprocess.run(cmd, check=True, cwd=cwd)


def get_pipeline_or_exit(version: Optional[str]) -> Pipeline:
    """Fetch pipeline or exit with error message."""
    try:
        return get_pipeline(version)
    except KeyError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)


def require_script(script: Path) -> None:
    """Exit with error if script doesn't exist."""
    if not script.exists():
        typer.echo(f"✗ Script not found: {script}", err=True)
        raise typer.Exit(1)


def extract_base_key(version_key: str, pipe: Pipeline) -> str:
    """Extract base version key for output paths.

    Handles multi-model variants: v14-4B → v14, v15-oq6 → v15
    """
    base_key = pipe.base_version or version_key
    if base_key and '-' in base_key:
        parts = base_key.split('-')
        if parts[0].startswith('v') and parts[0][1:].split('_')[0].isdigit():
            base_key = parts[0]
    return base_key

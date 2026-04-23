"""
Unified CLI for the student model harness — adaptive to any pipeline.

Usage:
    harness                                # Show all verbs
    harness versions                       # List registered pipelines
    harness eval run --version v15 -o ...  # Run with a pipeline's defaults
    harness eval run -m <path> -p <path>   # Fully manual — any model + prompt

Switch the default pipeline for a session:
    HARNESS_VERSION=v13 harness eval run -o foo     # Uses v13 defaults

Each top-level verb lives in finetune/commands/<verb>.py as a Typer sub-app.
Pipeline config is centralized in finetune/registry.py.
"""
from pathlib import Path
import subprocess
from typing import Optional
import typer

from finetune.commands import (
    convert,
    data,
    eval as eval_cmd,  # `eval` is a Python builtin
    hybrid,
    master_eval,
    mlflow,
    promptfoo,
    results,
    sweep,
    train,
)
from finetune.registry import PIPELINES, default_version

app = typer.Typer(
    name="harness",
    help="Student model training + evaluation harness.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def versions():
    """List all registered pipelines and show the current default."""
    default = default_version()
    typer.echo(f"Default: {default}  (set HARNESS_VERSION env var to change)\n")
    for key, p in PIPELINES.items():
        marker = "\u2605" if key == default else " "
        typer.echo(f"  {marker} {key:10s}  {p.description}")
        typer.echo(f"      model:   {p.model}")
        if p.adapter:
            typer.echo(f"      adapter: {p.adapter}")
        typer.echo(f"      prompt:  {p.prompt}  (backend={p.backend})")
        typer.echo()

app.add_typer(data.app, name="data", help="Data pipeline: generate, label, audit, merge.")
app.add_typer(train.app, name="train", help="LoRA fine-tuning (v14, v15).")
app.add_typer(eval_cmd.app, name="eval", help="Model-only inference + scoring.")
app.add_typer(sweep.app, name="sweep", help="Multi-checkpoint eval across iterations.")
app.add_typer(hybrid.app, name="hybrid", help="Regex + model hybrid scoring.")
app.add_typer(master_eval.app, name="master-eval", help="Multi-model eval + leaderboard.")
app.add_typer(results.app, name="results", help="View + compare eval results.")
app.add_typer(promptfoo.app, name="promptfoo", help="Promptfoo eval workflow (run/compare/view).")
app.add_typer(mlflow.app, name="mlflow", help="MLflow tracking + model registry.")
app.add_typer(convert.app, name="convert", help="HF \u2194 MLX \u2194 GGUF model conversion + quantization.")


# \u2500\u2500 Version shortcuts (version-first pattern) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# harness v15 eval \u2192 harness eval run --version v15
# harness v14 models \u2192 list models for v14

REPO = Path(__file__).resolve().parents[2]
PYTHON = str(REPO / ".venv/bin/python3")


def _make_version_app(version_key: str, display: str, model_list: list[str]):
    """Create a Typer sub-app for a version with eval + models subcommands."""
    version_app = typer.Typer(
        name=version_key.replace("-", "_"),
        help=display,
        no_args_is_help=True,
    )

    @version_app.callback()
    def version_callback(ctx: typer.Context):
        if ctx.invoked_subcommand is None:
            typer.echo(f"\n{display}")
            typer.echo(f"\nModels: {', '.join(model_list)}")
            typer.echo("\nCommands:")
            typer.echo("  eval    Run evaluation (auto-writes to versions/vN/eval_results/)")
            typer.echo("  models  List available models")
            raise typer.Exit()

    @version_app.command("models")
    def list_models():
        """List available models for this version."""
        typer.echo(f"\n{display} models:")
        for m in model_list:
            typer.echo(f"  - {m}")

    @version_app.command("eval")
    def run_eval(
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Model override"),
        prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt override"),
    ):
        """Run evaluation \u2014 auto-writes to versions/vN/eval_results/."""
        cmd = [PYTHON, "-m", "finetune.cli", "eval", "run", "--version", version_key]
        if model:
            cmd += ["--model", model]
        if prompt:
            cmd += ["--prompt", prompt]
        subprocess.run(cmd, check=True)

    return version_app


# Register version shortcuts with subcommands
app.add_typer(
    _make_version_app("v13", "V13 0.6B production (97.9% hybrid)", ["0.6B"]),
    name="v13",
)
app.add_typer(
    _make_version_app("v13_1", "V13.1 1.5B reference (97.5% hybrid)", ["1.5B", "0.6B_corrective"]),
    name="v13_1",
)
app.add_typer(
    _make_version_app("v14", "V14 multi-model", ["0.6B", "1.5B", "4B"]),
    name="v14",
)
app.add_typer(
    _make_version_app("v15", "V15 champion (99.6% hybrid)", ["4B", "4B-oQ6", "4B-mlx6", "4B-GGUF"]),
    name="v15",
)


if __name__ == "__main__":
    app()

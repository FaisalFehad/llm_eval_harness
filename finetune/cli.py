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
import typer

from finetune.commands import (
    convert,
    data,
    eval as eval_cmd,  # `eval` is a Python builtin
    hybrid,
    mlflow,
    promptfoo,
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
app.add_typer(promptfoo.app, name="promptfoo", help="Promptfoo eval workflow (run/compare/view).")
app.add_typer(mlflow.app, name="mlflow", help="MLflow tracking + model registry.")
app.add_typer(convert.app, name="convert", help="HF \u2194 MLX \u2194 GGUF model conversion + quantization.")


if __name__ == "__main__":
    app()

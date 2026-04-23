"""
Model-only inference + scoring — adaptive to any model/prompt combo.

Usage patterns:
    harness eval run                                   # Default pipeline (HARNESS_VERSION or v15)
    harness eval run --version v13                     # Use a registered pipeline's defaults
    harness eval run --model ~/custom --prompt p.txt   # Full manual override
    harness eval run --version v15 --prompt p.txt      # Mix: v15 defaults + custom prompt

All paths in --model / --adapter / --prompt / --test-file support ~ expansion.
"""
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional
import typer

from finetune.constants import REPO, PYTHON
from finetune.registry import PIPELINES, default_version, get as get_pipeline
from finetune.utils import expand_path, extract_base_key

app = typer.Typer(no_args_is_help=True, help="Model-only inference + scoring.")


@app.command()
def run(
    version: Optional[str] = typer.Option(
        None, "--version", "-v",
        help=f"Registered pipeline. Default: ${{HARNESS_VERSION:-{default_version()}}}. "
             f"Available: {sorted(PIPELINES)}"),
    model: Optional[str] = typer.Option(None, "--model", "-m",
        help="Override model HF alias or local path."),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a",
        help="Override LoRA adapter path."),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p",
        help="Override student prompt .txt path."),
    test_file: Optional[str] = typer.Option(None, "--test-file", "-t",
        help="Override test set (default from pipeline)."),
    backend: Optional[str] = typer.Option(None, "--backend", "-b",
        help="mlx | gguf | hf (default from pipeline)."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o",
        help="Where to write predictions + summary. Default: versions/vN/eval_results/<timestamp>_model/"),
    save_predictions: bool = typer.Option(True, "--save-predictions/--no-save-predictions"),
):
    """Run an eval. Flags layer on top of --version defaults."""
    pipe = get_pipeline(version)

    final_model = expand_path(model or pipe.model)
    final_adapter = expand_path(adapter or pipe.adapter)
    final_prompt = expand_path(prompt or pipe.prompt)
    final_test = expand_path(test_file or pipe.test_file)
    final_backend = backend or pipe.backend

    if not final_model or not final_prompt:
        typer.echo("\u2717 --model and --prompt are required (directly or via --version).", err=True)
        raise typer.Exit(1)

    # Auto output-dir: versions/vN/eval_results/<timestamp>_model/
    if output_dir is None:
        version_key = version or default_version()
        base_key = extract_base_key(version_key, pipe)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = final_model.split("/")[-1].replace(".gguf", "").replace("-4bit", "").replace("-bf16", "")[:20]
        output_dir = REPO / f"versions/{base_key}/eval_results/{timestamp}_{model_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

    script = REPO / ("finetune/eval_student_v14_gguf.py" if final_backend == "gguf"
                     else "finetune/eval_student_v7.py")

    cmd = [PYTHON, str(script),
           "--model", final_model,
           "--prompt", final_prompt,
           "--test-file", final_test,
           "--output-dir", str(output_dir)]
    if final_adapter:
        cmd += ["--adapter", final_adapter]
    if save_predictions:
        cmd.append("--save-predictions")

    typer.echo(f"\u2192 {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

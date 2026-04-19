"""HF \u2194 MLX \u2194 GGUF model conversion + quantization."""
import subprocess
from pathlib import Path
from typing import Optional
import typer

from finetune.registry import PIPELINES, default_version, get as get_pipeline

app = typer.Typer(no_args_is_help=True, help="Model format conversion.")

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "bin" / "python3")


@app.command("to-mlx")
def to_mlx(
    hf_path: Path = typer.Option(..., "--hf-path", "-i", help="Source HuggingFace model dir."),
    mlx_path: Path = typer.Option(..., "--mlx-path", "-o", help="Destination MLX model dir."),
    bits: int = typer.Option(6, "--bits", "-q", help="Quantization bits (4, 6, 8)"),
):
    """Convert HF model \u2192 MLX with quantization (uses mlx_lm convert)."""
    cmd = [PY, "-m", "mlx_lm", "convert",
           "--hf-path", str(hf_path),
           "--mlx-path", str(mlx_path),
           "-q", "--q-bits", str(bits)]
    typer.echo(f"\u2192 {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


@app.command("oq-fix")
def oq_fix():
    """Run quantize_oq_fix.py (V15 OQ quantization fix script)."""
    subprocess.run([PY, str(REPO / "versions/v15/scripts/quantize_oq.py")], check=True)


@app.command("from-hf-adapter")
def from_hf_adapter(
    adapter_dir: Path = typer.Option(..., "--adapter-dir", "-a",
        help="Directory containing adapter_config.json + adapter_model.safetensors (HF/PEFT format)."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o",
        help="Where to write the merged HF model."),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b",
        help="HF model ID (overrides registry)."),
    version: Optional[str] = typer.Option(None, "--version", "-v",
        help=f"Registered pipeline to source base_model_hf from. Default: "
             f"${{HARNESS_VERSION:-{default_version()}}}."),
    also_convert_mlx: Optional[Path] = typer.Option(None, "--mlx-out",
        help="If set, also convert merged model to MLX 6-bit at this path."),
):
    """
    Merge an HF/PEFT LoRA adapter into its base model and save the merged HF model.

    This is the "rehydration" path for V14-style HF adapters. After running, you can
    further convert to MLX or GGUF for deployment.

    Requires: pip install -e ".[hf-merge]"  (installs peft + transformers).
    """
    # Resolve base_model from registry if not passed explicitly
    if not base_model:
        try:
            pipe = get_pipeline(version)
        except KeyError as e:
            typer.echo(f"\u2717 {e}", err=True)
            raise typer.Exit(1)
        base_model = pipe.base_model_hf
        if not base_model:
            typer.echo(f"\u2717 No base_model_hf in registry for '{version or default_version()}'. "
                       f"Pass --base-model <hf_id>.", err=True)
            raise typer.Exit(1)

    # Lazy import — peft/transformers are heavy and only needed here
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        typer.echo(f"\u2717 Missing dependency: {e}", err=True)
        typer.echo("   Install with: pip install -e \".[hf-merge]\"", err=True)
        raise typer.Exit(1)

    typer.echo(f"\u2192 Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    tok = AutoTokenizer.from_pretrained(base_model)

    typer.echo(f"\u2192 Loading LoRA adapter from: {adapter_dir}")
    merged = PeftModel.from_pretrained(base, str(adapter_dir))

    typer.echo("\u2192 Merging adapter into base weights...")
    merged = merged.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"\u2192 Saving merged HF model to: {output_dir}")
    merged.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)

    if also_convert_mlx:
        typer.echo(f"\u2192 Converting to MLX 6-bit at: {also_convert_mlx}")
        subprocess.run([PY, "-m", "mlx_lm", "convert",
                        "--hf-path", str(output_dir),
                        "--mlx-path", str(also_convert_mlx),
                        "-q", "--q-bits", "6"], check=True)
        typer.echo(f"\u2713 MLX 6-bit model ready at: {also_convert_mlx}")

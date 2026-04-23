"""Promptfoo eval workflow \u2014 thin wrapper around eval.sh."""
import subprocess
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True, help="Promptfoo workflow.")

REPO = Path(__file__).resolve().parents[2]
EVAL_SH = str(REPO / "eval.sh")


def _eval_sh(*args):
    subprocess.run([EVAL_SH, *args], check=True, cwd=REPO)


@app.command()
def run(prompt: str = typer.Argument("production", help="Prompt name: production, fix1..fix5")):
    """Run an eval against a single prompt (delegates to ./eval.sh run)."""
    _eval_sh("run", prompt)


@app.command()
def compare(prompt_a: str, prompt_b: str):
    """A/B compare two prompts (delegates to ./eval.sh compare)."""
    _eval_sh("compare", prompt_a, prompt_b)


@app.command()
def smoke(n: int = typer.Argument(10, help="Number of test cases.")):
    """Smoke test with N cases (delegates to ./eval.sh smoke)."""
    _eval_sh("smoke", str(n))


@app.command()
def view():
    """Open promptfoo results UI in browser (port 15500)."""
    _eval_sh("view")


@app.command("eval-setup")
def eval_setup():
    """Open the Promptfoo Create Evaluation wizard."""
    subprocess.run(["npx", "promptfoo", "eval", "setup", "configs/"], check=True, cwd=REPO)


@app.command()
def server():
    """Start the OMLX model server on port 8000."""
    _eval_sh("server")


@app.command()
def langfuse():
    """Open Langfuse traces UI (port 3000)."""
    _eval_sh("langfuse")


@app.command("langfuse-up")
def langfuse_up():
    """Start the Langfuse Docker stack."""
    _eval_sh("langfuse-up")


@app.command("langfuse-down")
def langfuse_down():
    """Stop the Langfuse Docker stack."""
    _eval_sh("langfuse-down")

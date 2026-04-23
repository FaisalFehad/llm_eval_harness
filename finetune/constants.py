"""Shared constants for the finetune package."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = str(REPO / ".venv" / "bin" / "python3")

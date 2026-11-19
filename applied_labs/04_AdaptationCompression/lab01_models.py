"""Import Applied-Lab-01 model definitions without duplicating architecture code."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

LAB01_DIR = (
    Path(__file__).resolve().parents[1]
    / "01_CIFAR10Backbones"
)
if str(LAB01_DIR) not in sys.path:
    sys.path.insert(0, str(LAB01_DIR))

from models import build_model, parameter_count  # noqa: E402

__all__ = ["build_model", "parameter_count"]


def checkpoint_model_name(checkpoint: dict[str, Any]) -> str:
    name = checkpoint.get("model_name")
    if not isinstance(name, str):
        raise ValueError("checkpoint missing string model_name")
    return name

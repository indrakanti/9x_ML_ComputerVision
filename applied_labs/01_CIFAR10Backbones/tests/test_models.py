from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from models import build_model, parameter_count  # noqa: E402


def test_all_backbones_produce_ten_logits() -> None:
    sample = torch.randn(2, 3, 32, 32)

    for name in ("tinycnn", "resnet18", "tinyvit"):
        model = build_model(name)
        output = model(sample)

        assert output.shape == (2, 10)
        assert parameter_count(model) > 0


def test_unknown_model_is_rejected() -> None:
    try:
        build_model("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model should raise ValueError")

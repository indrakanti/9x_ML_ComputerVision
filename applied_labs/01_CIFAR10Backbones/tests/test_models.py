from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_models() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cv9x_lab01_models",
        ROOT / "models.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Lab 01 models")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_backbones_produce_ten_logits() -> None:
    models = load_models()
    sample = torch.randn(2, 3, 32, 32)

    for name in ("tinycnn", "resnet18", "tinyvit"):
        model = models.build_model(name)
        output = model(sample)

        assert output.shape == (2, 10)
        assert models.parameter_count(model) > 0


def test_unknown_model_is_rejected() -> None:
    models = load_models()
    try:
        models.build_model("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown model should raise ValueError")

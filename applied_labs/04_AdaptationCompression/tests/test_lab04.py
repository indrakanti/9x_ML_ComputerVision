from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lora_preserves_initial_output_and_freezes_base() -> None:
    lora = load_module("cv9x_lab04_lora", "lora.py")
    base = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    sample = torch.randn(5, 4)
    expected = base(sample).detach()

    replaced = lora.inject_lora(base, rank=1, alpha=2.0)
    actual = base(sample).detach()
    lora.freeze_except_lora(base)

    assert replaced == 2
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert lora.trainable_parameter_count(base) > 0
    assert all(
        parameter.requires_grad
        == (name.endswith("lora_a") or name.endswith("lora_b"))
        for name, parameter in base.named_parameters()
    )


def test_manual_pruning_policy_produces_zeros() -> None:
    layer = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.01, 0.20, -0.80, 0.03],
                    [0.90, -0.02, 0.40, 0.10],
                ]
            )
        )

    threshold = 0.15
    with torch.no_grad():
        layer.weight[torch.abs(layer.weight) < threshold] = 0.0

    assert int((layer.weight == 0).sum().item()) > 0

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from mlp import forward_logit, he_initialize, load_model, normalize, relu


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "mlp_binary_model.txt"


def test_relu() -> None:
    assert relu(-2.0) == 0.0
    assert relu(3.0) == 3.0


def test_normalize() -> None:
    assert normalize([2.0], (1.0,), (0.5,)) == pytest.approx([2.0])


def test_he_initialization_is_deterministic() -> None:
    assert he_initialize(4, 3, seed=7) == he_initialize(4, 3, seed=7)


def test_reference_matches_pytorch() -> None:
    model = load_model(artifact_path())
    sample = [0.6, 0.2, 0.8, 0.3]

    network = torch.nn.Sequential(
        torch.nn.Linear(4, 3, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1, dtype=torch.float64),
    )
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor(model.w1, dtype=torch.float64))
        network[0].bias.copy_(torch.tensor(model.b1, dtype=torch.float64))
        network[2].weight.copy_(torch.tensor([model.w2], dtype=torch.float64))
        network[2].bias.copy_(torch.tensor([model.b2], dtype=torch.float64))

    normalized = torch.tensor(
        [[(x - m) / s for x, m, s in zip(sample, model.mean, model.std, strict=True)]],
        dtype=torch.float64,
    )
    assert network(normalized).item() == pytest.approx(forward_logit(model, sample))


def test_bad_input_shape_rejected() -> None:
    model = load_model(artifact_path())
    with pytest.raises(ValueError):
        forward_logit(model, [1.0])


def test_he_rejects_bad_fan_in() -> None:
    with pytest.raises(ValueError):
        he_initialize(0, 3)

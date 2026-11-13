from __future__ import annotations

from pathlib import Path

import pytest
import torch
from linear_classifier import (
    LinearBinaryModel,
    binary_cross_entropy_with_logits,
    linear_logit,
    load_model,
    mean_loss,
    predict,
    predict_probability,
    save_model,
    sigmoid,
    train_logistic_regression,
)


def test_linear_logit() -> None:
    value = linear_logit([0.5, 0.2, 0.7, 0.1], [0.8, -0.4, 1.1, 0.6], -0.25)
    assert value == pytest.approx(0.9)


def test_sigmoid_is_stable_for_large_magnitude_logits() -> None:
    assert sigmoid(1000.0) == pytest.approx(1.0)
    assert sigmoid(-1000.0) == pytest.approx(0.0)


def test_bce_with_logits_is_finite() -> None:
    assert binary_cross_entropy_with_logits(1000.0, 1) < 1e-6
    assert binary_cross_entropy_with_logits(-1000.0, 0) < 1e-6


def test_prediction_uses_threshold() -> None:
    model = LinearBinaryModel(weights=(1.0,), bias=0.0, threshold=0.7)

    assert predict_probability(model, [0.0]) == pytest.approx(0.5)
    assert predict(model, [0.0]) == 0


def test_training_reduces_loss_and_separates_tiny_dataset() -> None:
    features = [
        [0.10, 0.20],
        [0.20, 0.10],
        [0.80, 0.70],
        [0.90, 0.80],
    ]
    targets = [0, 0, 1, 1]
    initial = LinearBinaryModel(weights=(0.0, 0.0), bias=0.0)
    trained = train_logistic_regression(features, targets, learning_rate=0.3, steps=500)

    assert mean_loss(trained, features, targets) < mean_loss(initial, features, targets)
    assert [predict(trained, sample) for sample in features] == targets


def test_manual_logit_matches_pytorch_linear_layer() -> None:
    weights = (0.8, -0.4, 1.1, 0.6)
    bias = -0.25
    sample = [0.5, 0.2, 0.7, 0.1]

    layer = torch.nn.Linear(4, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([weights], dtype=torch.float64))
        layer.bias.copy_(torch.tensor([bias], dtype=torch.float64))

    actual = layer(torch.tensor([sample], dtype=torch.float64)).item()
    expected = linear_logit(sample, weights, bias)

    assert actual == pytest.approx(expected)


def test_model_artifact_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "model.txt"
    model = LinearBinaryModel(weights=(0.1, -0.2), bias=0.3, threshold=0.6)

    save_model(model, path)
    loaded = load_model(path)

    assert loaded == model


def test_model_artifact_rejects_bad_weight_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(
        "version=1\ninput_size=2\nweights=0.1\nbias=0.0\nthreshold=0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_model(path)


def test_invalid_shapes_and_targets_are_rejected() -> None:
    with pytest.raises(ValueError):
        linear_logit([1.0], [1.0, 2.0], 0.0)

    with pytest.raises(ValueError):
        binary_cross_entropy_with_logits(0.0, 2)

    with pytest.raises(ValueError):
        train_logistic_regression([[1.0]], [0, 1])

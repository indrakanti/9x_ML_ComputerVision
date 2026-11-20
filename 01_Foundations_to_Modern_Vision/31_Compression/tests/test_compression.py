from __future__ import annotations

from pathlib import Path

import pytest
from compression import (
    distillation_loss,
    load_artifact,
    prune_weights,
    pytorch_distillation_loss,
    quantize_symmetric_int8,
    sparsity,
)


def artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "compression.txt"
    )


def test_distillation_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())

    assert distillation_loss(artifact) == pytest.approx(
        float(pytorch_distillation_loss(artifact).item()),
        rel=1e-12,
        abs=1e-12,
    )


def test_pruning_creates_sparsity() -> None:
    artifact = load_artifact(artifact_path())
    pruned = prune_weights(
        artifact.weights,
        artifact.prune_threshold,
    )

    assert sparsity(pruned) > 0.0


def test_quantization_round_trip_error_is_bounded() -> None:
    artifact = load_artifact(artifact_path())
    scale, quantized, dequantized, max_error = quantize_symmetric_int8(
        artifact.quant_values
    )

    assert scale > 0.0
    assert all(-127 <= value <= 127 for value in quantized)
    assert len(dequantized) == len(artifact.quant_values)
    assert max_error <= scale / 2.0 + 1e-12

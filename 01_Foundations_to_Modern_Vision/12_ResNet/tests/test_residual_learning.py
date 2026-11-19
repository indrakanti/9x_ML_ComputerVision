from __future__ import annotations

from pathlib import Path

import pytest
import torch
from residual_learning import (
    ResidualArtifact,
    TinyResidualBlock,
    apply_artifact,
    example_input,
    load_artifact,
    manual_residual_forward,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "residual_block.txt"


def test_manual_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    model = TinyResidualBlock(artifact.channels)
    apply_artifact(model, artifact)

    values = example_input()
    tensor = torch.tensor(values, dtype=torch.float64).reshape(
        1,
        artifact.channels,
        artifact.height,
        artifact.width,
    )
    expected = manual_residual_forward(artifact, values)
    with torch.no_grad():
        actual = model(tensor).reshape(-1).tolist()

    assert actual == pytest.approx(expected)


def test_zero_residual_branch_reduces_to_relu_identity() -> None:
    artifact = load_artifact(artifact_path())
    zero = ResidualArtifact(
        channels=artifact.channels,
        height=artifact.height,
        width=artifact.width,
        weights=(0.0,) * len(artifact.weights),
        bias=(0.0,) * len(artifact.bias),
    )
    values = example_input()
    output = manual_residual_forward(zero, values)
    assert output == pytest.approx([max(0.0, value) for value in values])

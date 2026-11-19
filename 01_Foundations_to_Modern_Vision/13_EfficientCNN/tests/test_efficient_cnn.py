from __future__ import annotations

from pathlib import Path

import pytest
from efficient_cnn import (
    example_input,
    load_artifact,
    mac_counts,
    manual_forward,
    pytorch_forward,
)


def artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "depthwise_separable.txt"
    )


def test_manual_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    expected = manual_forward(artifact, example_input())
    actual = pytorch_forward(artifact, example_input()).reshape(-1).tolist()
    assert actual == pytest.approx(expected)


def test_separable_uses_fewer_macs_than_standard() -> None:
    artifact = load_artifact(artifact_path())
    standard, separable = mac_counts(artifact)
    assert separable < standard

from __future__ import annotations

from pathlib import Path

import pytest
from hybrid_backbone import (
    example_image,
    load_artifact,
    manual_forward,
    pytorch_forward,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "hybrid.txt"


def test_manual_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    expected = manual_forward(artifact, example_image())
    actual = pytorch_forward(
        artifact,
        example_image(),
    ).reshape(-1).tolist()

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

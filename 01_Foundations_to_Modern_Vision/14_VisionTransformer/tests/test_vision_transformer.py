from __future__ import annotations

from pathlib import Path

import pytest
from vision_transformer import (
    example_image,
    load_artifact,
    manual_attention,
    patchify,
    pytorch_attention,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "tiny_vit.txt"


def test_patchify_order() -> None:
    artifact = load_artifact(artifact_path())
    patches = patchify(artifact, list(range(16)))
    assert patches == [
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]


def test_manual_attention_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    expected = manual_attention(artifact, example_image())
    actual = pytorch_attention(artifact, example_image()).reshape(-1).tolist()
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

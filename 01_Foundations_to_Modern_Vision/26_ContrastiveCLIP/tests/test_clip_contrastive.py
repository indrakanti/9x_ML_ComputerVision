from __future__ import annotations

from pathlib import Path

import pytest
from clip_contrastive import (
    load_artifact,
    pytorch_contrastive_loss,
    similarity_matrix,
    symmetric_contrastive_loss,
    top_text_matches,
)


def artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "embeddings.txt"
    )


def test_similarity_matrix_shape_and_ranking() -> None:
    artifact = load_artifact(artifact_path())
    matrix = similarity_matrix(artifact)

    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert top_text_matches(artifact) == [0, 1, 2]


def test_manual_loss_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    manual = symmetric_contrastive_loss(artifact)
    torch_loss = float(pytorch_contrastive_loss(artifact).item())

    assert manual == pytest.approx(torch_loss, rel=1e-12, abs=1e-12)

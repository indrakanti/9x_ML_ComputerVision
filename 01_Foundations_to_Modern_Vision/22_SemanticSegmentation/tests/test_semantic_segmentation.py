from __future__ import annotations

from pathlib import Path

import pytest
from semantic_segmentation import (
    confusion_matrix,
    load_artifact,
    mean_iou,
    predict_labels,
)


def artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "segmentation.txt"
    )


def test_prediction_shape() -> None:
    artifact = load_artifact(artifact_path())
    predictions = predict_labels(artifact)

    assert len(predictions) == artifact.height * artifact.width


def test_metric_values_are_valid() -> None:
    artifact = load_artifact(artifact_path())
    matrix = confusion_matrix(artifact, predict_labels(artifact))
    miou = mean_iou(matrix)

    assert 0.0 <= miou <= 1.0
    assert miou == pytest.approx(1.0)

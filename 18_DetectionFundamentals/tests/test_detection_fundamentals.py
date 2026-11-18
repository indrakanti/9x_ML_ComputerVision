from __future__ import annotations

from pathlib import Path

import pytest
from detection_fundamentals import (
    Box,
    average_precision,
    class_aware_nms,
    iou,
    load_artifact,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "detections.txt"


def test_iou_known_value() -> None:
    assert iou(Box(0, 0, 2, 2), Box(1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_class_aware_nms() -> None:
    artifact = load_artifact(artifact_path())
    keep = class_aware_nms(
        artifact.boxes,
        artifact.scores,
        artifact.classes,
        artifact.nms_iou,
    )
    assert keep == [0, 2]


def test_average_precision_perfect_ranking() -> None:
    ap = average_precision(
        scores=[0.9, 0.8, 0.2],
        true_positive=[True, True, False],
        ground_truth_count=2,
    )
    assert ap == pytest.approx(1.0)

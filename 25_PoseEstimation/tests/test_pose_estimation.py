from __future__ import annotations

from pathlib import Path

import pytest
from pose_estimation import decode_keypoints, load_artifact, pck


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "pose.txt"


def test_decode_expected_locations() -> None:
    artifact = load_artifact(artifact_path())
    points = decode_keypoints(artifact)

    assert points[0].x == pytest.approx(2.5)
    assert points[0].y == pytest.approx(2.5)
    assert points[2].x == pytest.approx(4.5)
    assert points[2].y == pytest.approx(4.5)


def test_pck_is_perfect_for_reference() -> None:
    artifact = load_artifact(artifact_path())
    points = decode_keypoints(artifact)

    assert pck(artifact, points) == pytest.approx(1.0)

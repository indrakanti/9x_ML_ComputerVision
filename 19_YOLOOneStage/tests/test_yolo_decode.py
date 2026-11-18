from __future__ import annotations

from pathlib import Path

from yolo_decode import decode, final_detections, load_artifact


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "yolo_head.txt"


def test_decode_produces_candidates() -> None:
    artifact = load_artifact(artifact_path())
    detections = decode(artifact)

    assert len(detections) >= 3
    assert all(
        detection.score >= artifact.score_threshold
        for detection in detections
    )


def test_final_detections_are_score_sorted() -> None:
    artifact = load_artifact(artifact_path())
    detections = final_detections(artifact)
    scores = [detection.score for detection in detections]

    assert scores == sorted(scores, reverse=True)

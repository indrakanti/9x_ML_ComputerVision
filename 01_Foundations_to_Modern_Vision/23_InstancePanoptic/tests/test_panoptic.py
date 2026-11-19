from __future__ import annotations

from pathlib import Path

import pytest
from panoptic import compose_panoptic, load_artifact, panoptic_quality


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "panoptic.txt"


def test_panoptic_map_size() -> None:
    artifact = load_artifact(artifact_path())
    output = compose_panoptic(artifact)

    assert len(output) == artifact.height * artifact.width


def test_higher_score_instance_claims_overlap_first() -> None:
    artifact = load_artifact(artifact_path())
    output = compose_panoptic(artifact)

    assert output[1] == 1


def test_panoptic_quality_formula() -> None:
    value = panoptic_quality(
        matched_ious=[0.8, 0.7],
        false_positives=1,
        false_negatives=0,
    )
    assert value == pytest.approx(1.5 / 2.5)

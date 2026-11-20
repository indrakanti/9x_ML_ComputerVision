from __future__ import annotations

from pathlib import Path

from foundation_backbone import (
    adapter_forward,
    adapter_overhead_ratio,
    load_artifact,
    total_adapter_parameters,
)


def artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "foundation.txt"
    )


def test_two_adapters_share_same_features() -> None:
    artifact = load_artifact(artifact_path())
    classification = adapter_forward(
        artifact.features,
        artifact.classification,
    )
    geometry = adapter_forward(
        artifact.features,
        artifact.geometry,
    )

    assert len(classification) == 3
    assert len(geometry) == 2


def test_adapter_overhead_is_small() -> None:
    artifact = load_artifact(artifact_path())

    assert total_adapter_parameters(artifact) == 25
    assert adapter_overhead_ratio(artifact) < 0.001

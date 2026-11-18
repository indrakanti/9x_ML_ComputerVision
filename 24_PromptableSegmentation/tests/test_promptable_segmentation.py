from __future__ import annotations

from pathlib import Path

from promptable_segmentation import (
    load_artifact,
    prompt_consistent,
    select_mask,
)


def artifact_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "promptable.txt"
    )


def test_selects_highest_scoring_prompt_consistent_mask() -> None:
    artifact = load_artifact(artifact_path())
    index, mask = select_mask(artifact)

    assert index == 0
    assert prompt_consistent(artifact, mask)


def test_selected_mask_has_expected_size() -> None:
    artifact = load_artifact(artifact_path())
    _, mask = select_mask(artifact)

    assert len(mask) == artifact.height * artifact.width

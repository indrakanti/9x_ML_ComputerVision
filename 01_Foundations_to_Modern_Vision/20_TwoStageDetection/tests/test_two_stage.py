from __future__ import annotations

from pathlib import Path

from two_stage import Box, load_artifact, proposals, roi_pool_2x2


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "proposals.txt"


def test_proposal_policy_respects_post_nms_topk() -> None:
    artifact = load_artifact(artifact_path())
    output = proposals(artifact)

    assert len(output) <= artifact.post_nms_topk
    assert output[0][1] >= output[-1][1]


def test_roi_pool_returns_four_bins() -> None:
    feature = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]
    pooled = roi_pool_2x2(feature, Box(0.0, 0.0, 4.0, 4.0))

    assert pooled == (6.0, 8.0, 14.0, 16.0)

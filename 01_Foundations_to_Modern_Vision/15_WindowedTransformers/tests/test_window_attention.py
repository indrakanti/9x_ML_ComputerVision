from __future__ import annotations

from pathlib import Path

import pytest
from window_attention import (
    example_tokens,
    load_artifact,
    manual_window_attention,
    partition_windows,
    pytorch_window_attention,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "window_attention.txt"


def test_partition_order() -> None:
    artifact = load_artifact(artifact_path())
    windows = partition_windows(artifact, example_tokens())
    assert len(windows) == 4
    assert windows[0][0] == [0.0, 0.1]
    assert windows[0][1] == [0.2, 0.3]


def test_manual_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())
    expected = manual_window_attention(artifact, example_tokens())
    actual = pytorch_window_attention(artifact, example_tokens()).reshape(-1).tolist()
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

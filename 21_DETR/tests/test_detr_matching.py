from __future__ import annotations

from pathlib import Path

from detr_matching import cost_matrix, load_artifact, optimal_assignment


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "matching.txt"


def test_expected_assignment() -> None:
    artifact = load_artifact(artifact_path())
    assignment = optimal_assignment(cost_matrix(artifact))

    assert assignment == [(0, 0), (1, 1)]


def test_cost_matrix_shape() -> None:
    artifact = load_artifact(artifact_path())
    matrix = cost_matrix(artifact)

    assert len(matrix) == 3
    assert all(len(row) == 2 for row in matrix)

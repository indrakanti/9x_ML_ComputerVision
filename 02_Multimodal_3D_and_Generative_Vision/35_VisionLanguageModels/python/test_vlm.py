"""Offline tests for visual projection and placeholder replacement."""
import pytest
from vlm import fuse, project, synthetic


def test_project() -> None:
    assert project([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]]) == [[1.0, 2.0]]


def test_fusion() -> None:
    assert synthetic() == [
        [9.0, 9.0, 9.0],
        [1.0, 1.5, 2.0],
        [3.0, 3.5, 4.0],
        [8.0, 8.0, 8.0],
    ]


@pytest.mark.parametrize("patches,weights", [
    ([], [[1.0]]),
    ([[1.0]], []),
    ([[1.0, 2.0]], [[1.0]]),
    ([[float("nan")]], [[1.0]]),
    ([[1.0]], [[float("inf")]]),
])
def test_bad_projection(patches: list[list[float]], weights: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        project(patches, weights)


def test_bad_fusion() -> None:
    with pytest.raises(ValueError):
        fuse([[1.0]], [[1.0, 2.0]], 0)
    with pytest.raises(ValueError):
        fuse([[1.0]], [[1.0]], 2)

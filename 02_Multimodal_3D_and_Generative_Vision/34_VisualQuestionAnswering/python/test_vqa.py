"""Offline deterministic tests for VQA grounding."""
import math

import pytest

from vqa import ground, softmax, synthetic


def test_grounding() -> None:
    answer, attention = synthetic()
    assert answer == "red car"
    assert attention[0] > attention[1]
    assert sum(attention) == pytest.approx(1.0)


def test_stable_softmax() -> None:
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("question,patches,labels", [
    ([], [[1.0]], ["x"]),
    ([1.0], [], []),
    ([1.0], [[1.0]], []),
    ([1.0, 0.0], [[1.0]], ["x"]),
    ([float("nan")], [[1.0]], ["x"]),
    ([1.0], [[float("inf")]], ["x"]),
    ([1.0], [[1.0]], [""]),
])
def test_reject_invalid_inputs(
    question: list[float], patches: list[list[float]], labels: list[str]
) -> None:
    with pytest.raises(ValueError):
        ground(question, patches, labels)


def test_finite_attention() -> None:
    _, weights = synthetic()
    assert all(math.isfinite(x) for x in weights)

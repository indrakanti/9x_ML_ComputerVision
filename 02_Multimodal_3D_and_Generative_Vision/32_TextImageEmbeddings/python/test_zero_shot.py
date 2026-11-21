"""Deterministic CPU-only tests for Module 32."""
import math

import pytest
from zero_shot import normalize, scores


def test_scores() -> None:
    assert scores([3, 4], [[3, 4], [4, -3], [-3, -4]]) == pytest.approx([1, 0, -1])


def test_temperature() -> None:
    assert scores([1, 0], [[1, 0]], temperature=0.5) == pytest.approx([2])


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        scores([1, 0], [[1, 0]], temperature=temperature)


def test_zero_embedding() -> None:
    with pytest.raises(ValueError):
        normalize([0, 0])


def test_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        scores([1, 0], [[1, 0, 0]])


def test_unit_length() -> None:
    vector = normalize([3, 4])
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1)

"""Module 33 deterministic CPU tests (no downloads)."""
import math

import pytest
from captioning import attend, softmax, synthetic


def test_softmax_stable() -> None:
    assert softmax([1000.0, 1000.0]) == pytest.approx([0.5, 0.5])


def test_attention_weights_and_context() -> None:
    weights, context = attend([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    assert sum(weights) == pytest.approx(1.0)
    assert context == pytest.approx(weights)
    assert weights[0] > weights[1]


def test_synthetic_contract() -> None:
    assert len(synthetic()) == 4
    assert all(math.isfinite(x) for x in synthetic())


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")]])
def test_invalid_logits(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        softmax(bad)


def test_invalid_patches() -> None:
    with pytest.raises(ValueError):
        attend([1.0, 0.0], [[1.0]])
    with pytest.raises(ValueError):
        attend([1.0, 0.0], [])

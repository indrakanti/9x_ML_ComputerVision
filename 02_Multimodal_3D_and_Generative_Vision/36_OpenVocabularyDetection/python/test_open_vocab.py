"""Deterministic CPU tests for Module 36."""
import pytest
from open_vocab import cosine, detect, iou, synthetic


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def test_iou() -> None:
    assert iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def test_nms_and_open_label() -> None:
    assert synthetic() == pytest.approx([(0, 0, 1.0), (2, 1, 1.0)])


def test_different_classes_not_suppressed() -> None:
    result = detect(
        [[0, 0, 2, 2], [0, 0, 2, 2]],
        [[1, 0], [0, 1]],
        ["car", "person"],
        [[1, 0], [0, 1]],
    )
    assert len(result) == 2


@pytest.mark.parametrize("a,b", [
    ([], [1.0]),
    ([0.0], [1.0]),
    ([float("nan")], [1.0]),
])
def test_bad_embeddings(a: list[float], b: list[float]) -> None:
    with pytest.raises(ValueError):
        cosine(a, b)


def test_bad_box() -> None:
    with pytest.raises(ValueError):
        iou([0, 0, 0, 2], [0, 0, 1, 1])

from __future__ import annotations

import pytest
from data_discipline import assign_split, binary_metrics, find_split_leakage, stable_hash64


def test_stable_hash_is_deterministic() -> None:
    assert stable_hash64("frame_0001") == stable_hash64("frame_0001")


def test_split_assignment_is_order_independent() -> None:
    sample_ids = ["a", "b", "c", "d", "e"]
    forward = {sample_id: assign_split(sample_id) for sample_id in sample_ids}
    reverse = {sample_id: assign_split(sample_id) for sample_id in reversed(sample_ids)}

    assert forward == reverse


def test_existing_assignment_does_not_change_when_new_sample_is_added() -> None:
    original = {sample_id: assign_split(sample_id) for sample_id in ["a", "b", "c"]}
    extended = {sample_id: assign_split(sample_id) for sample_id in ["a", "b", "c", "new"]}

    assert {key: extended[key] for key in original} == original


def test_find_split_leakage_detects_cross_split_overlap() -> None:
    leakage = find_split_leakage(
        {
            "train": ["a", "b"],
            "validation": ["c"],
            "test": ["b", "d"],
        }
    )

    assert leakage == {"b": ["train", "test"]}


def test_find_split_leakage_ignores_unique_samples() -> None:
    assert find_split_leakage({"train": ["a"], "validation": ["b"], "test": ["c"]}) == {}


def test_binary_metrics() -> None:
    metrics = binary_metrics(
        truth=[1, 1, 1, 0, 0, 0],
        prediction=[1, 1, 0, 1, 0, 0],
    )

    assert metrics.true_positive == 2
    assert metrics.true_negative == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.accuracy == pytest.approx(2.0 / 3.0)
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.recall == pytest.approx(2.0 / 3.0)
    assert metrics.specificity == pytest.approx(2.0 / 3.0)
    assert metrics.f1 == pytest.approx(2.0 / 3.0)


def test_zero_denominator_metric_is_explicitly_zero() -> None:
    metrics = binary_metrics(truth=[0, 0], prediction=[0, 0])

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_binary_metrics_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        binary_metrics(truth=[1], prediction=[1, 0])

    with pytest.raises(ValueError):
        binary_metrics(truth=[], prediction=[])

    with pytest.raises(ValueError):
        binary_metrics(truth=[2], prediction=[1])


def test_assign_split_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        assign_split("")

    with pytest.raises(ValueError):
        assign_split("x", train_fraction=0.9, validation_fraction=0.2)

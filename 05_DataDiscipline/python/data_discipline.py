"""Dataset split, leakage, and binary metric helpers for Module 05."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


FNV_OFFSET_BASIS_64 = 14695981039346656037
FNV_PRIME_64 = 1099511628211
UINT64_MASK = (1 << 64) - 1


def stable_hash64(text: str) -> int:
    value = FNV_OFFSET_BASIS_64
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME_64) & UINT64_MASK
    return value


def assign_split(
    sample_id: str,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> str:
    if not sample_id:
        raise ValueError("sample_id must be non-empty")
    if train_fraction <= 0.0 or validation_fraction < 0.0:
        raise ValueError("split fractions must be non-negative and train > 0")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1")

    bucket = stable_hash64(sample_id) % 10_000
    train_limit = round(train_fraction * 10_000)
    validation_limit = train_limit + round(validation_fraction * 10_000)

    if bucket < train_limit:
        return "train"
    if bucket < validation_limit:
        return "validation"
    return "test"


def find_split_leakage(splits: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for split_name, sample_ids in splits.items():
        for sample_id in sample_ids:
            owners.setdefault(sample_id, []).append(split_name)

    return {
        sample_id: split_names
        for sample_id, split_names in owners.items()
        if len(set(split_names)) > 1
    }


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    @property
    def accuracy(self) -> float:
        return _safe_divide(self.true_positive + self.true_negative, self.total)

    @property
    def precision(self) -> float:
        return _safe_divide(self.true_positive, self.true_positive + self.false_positive)

    @property
    def recall(self) -> float:
        return _safe_divide(self.true_positive, self.true_positive + self.false_negative)

    @property
    def specificity(self) -> float:
        return _safe_divide(self.true_negative, self.true_negative + self.false_positive)

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 0.0 if denominator == 0.0 else 2.0 * self.precision * self.recall / denominator


def binary_metrics(truth: Sequence[int], prediction: Sequence[int]) -> BinaryMetrics:
    if len(truth) != len(prediction):
        raise ValueError("truth and prediction must have the same length")
    if not truth:
        raise ValueError("truth and prediction must be non-empty")

    tp = tn = fp = fn = 0
    for expected, actual in zip(truth, prediction, strict=True):
        if expected not in (0, 1) or actual not in (0, 1):
            raise ValueError("binary labels must be 0 or 1")
        if expected == 1 and actual == 1:
            tp += 1
        elif expected == 0 and actual == 0:
            tn += 1
        elif expected == 0 and actual == 1:
            fp += 1
        else:
            fn += 1

    return BinaryMetrics(tp, tn, fp, fn)


def _safe_divide(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator

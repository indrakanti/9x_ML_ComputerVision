"""From-scratch binary linear classifier for Module 06."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinearBinaryModel:
    weights: tuple[float, ...]
    bias: float
    threshold: float = 0.5
    version: int = 1

    @property
    def input_size(self) -> int:
        return len(self.weights)


def linear_logit(features: Sequence[float], weights: Sequence[float], bias: float) -> float:
    if len(features) != len(weights):
        raise ValueError("features and weights must have the same length")
    if not features:
        raise ValueError("features must be non-empty")
    return sum(feature * weight for feature, weight in zip(features, weights, strict=True)) + bias


def sigmoid(logit: float) -> float:
    if logit >= 0.0:
        exp_neg = math.exp(-logit)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(logit)
    return exp_pos / (1.0 + exp_pos)


def binary_cross_entropy_with_logits(logit: float, target: int) -> float:
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def predict_probability(model: LinearBinaryModel, features: Sequence[float]) -> float:
    return sigmoid(linear_logit(features, model.weights, model.bias))


def predict(model: LinearBinaryModel, features: Sequence[float]) -> int:
    return int(predict_probability(model, features) >= model.threshold)


def mean_loss(
    model: LinearBinaryModel,
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
) -> float:
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    if not features:
        raise ValueError("training data must be non-empty")

    total = 0.0
    for sample, target in zip(features, targets, strict=True):
        total += binary_cross_entropy_with_logits(
            linear_logit(sample, model.weights, model.bias),
            target,
        )
    return total / len(features)


def train_logistic_regression(
    features: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    learning_rate: float = 0.2,
    steps: int = 400,
) -> LinearBinaryModel:
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length")
    if not features:
        raise ValueError("training data must be non-empty")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if steps < 0:
        raise ValueError("steps must be >= 0")

    input_size = len(features[0])
    if input_size == 0:
        raise ValueError("feature vectors must be non-empty")
    if any(len(sample) != input_size for sample in features):
        raise ValueError("all feature vectors must have the same length")
    if any(target not in (0, 1) for target in targets):
        raise ValueError("targets must be binary")

    weights = [0.0] * input_size
    bias = 0.0
    sample_count = float(len(features))

    for _ in range(steps):
        grad_weights = [0.0] * input_size
        grad_bias = 0.0

        for sample, target in zip(features, targets, strict=True):
            probability = sigmoid(linear_logit(sample, weights, bias))
            error = probability - target
            for index, feature in enumerate(sample):
                grad_weights[index] += error * feature
            grad_bias += error

        for index in range(input_size):
            weights[index] -= learning_rate * grad_weights[index] / sample_count
        bias -= learning_rate * grad_bias / sample_count

    return LinearBinaryModel(weights=tuple(weights), bias=bias)


def save_model(model: LinearBinaryModel, path: Path) -> None:
    lines = [
        f"version={model.version}",
        f"input_size={model.input_size}",
        "weights=" + ",".join(f"{weight:.17g}" for weight in model.weights),
        f"bias={model.bias:.17g}",
        f"threshold={model.threshold:.17g}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model(path: Path) -> LinearBinaryModel:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator != "=":
            raise ValueError(f"invalid model artifact line: {raw_line}")
        values[key.strip()] = value.strip()

    required = {"version", "input_size", "weights", "bias", "threshold"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"model artifact missing fields: {sorted(missing)}")

    version = int(values["version"])
    if version != 1:
        raise ValueError(f"unsupported model artifact version: {version}")

    input_size = int(values["input_size"])
    weights = tuple(float(item) for item in values["weights"].split(",") if item)
    if len(weights) != input_size:
        raise ValueError("weight count does not match input_size")

    threshold = float(values["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    return LinearBinaryModel(
        weights=weights,
        bias=float(values["bias"]),
        threshold=threshold,
        version=version,
    )

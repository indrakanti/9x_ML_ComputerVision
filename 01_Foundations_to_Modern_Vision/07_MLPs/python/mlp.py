"""Two-layer MLP reference for Module 07."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLPModel:
    input_size: int
    hidden_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    w1: tuple[tuple[float, ...], ...]
    b1: tuple[float, ...]
    w2: tuple[float, ...]
    b2: float
    threshold: float = 0.5
    version: int = 1


def relu(value: float) -> float:
    return max(0.0, value)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def normalize(
    features: list[float],
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> list[float]:
    if len(features) != len(mean) or len(features) != len(std):
        raise ValueError("normalization shape mismatch")
    if any(value <= 0.0 for value in std):
        raise ValueError("standard deviations must be > 0")
    return [
        (value - mu) / sigma
        for value, mu, sigma in zip(features, mean, std, strict=True)
    ]


def forward_logit(model: MLPModel, features: list[float]) -> float:
    if len(features) != model.input_size:
        raise ValueError("input size mismatch")
    normalized = normalize(features, model.mean, model.std)
    hidden: list[float] = []
    for row, bias in zip(model.w1, model.b1, strict=True):
        if len(row) != model.input_size:
            raise ValueError("w1 shape mismatch")
        hidden.append(
            relu(sum(w * x for w, x in zip(row, normalized, strict=True)) + bias)
        )
    if len(hidden) != model.hidden_size or len(model.w2) != model.hidden_size:
        raise ValueError("hidden shape mismatch")
    return sum(w * h for w, h in zip(model.w2, hidden, strict=True)) + model.b2


def predict_probability(model: MLPModel, features: list[float]) -> float:
    return sigmoid(forward_logit(model, features))


def predict(model: MLPModel, features: list[float]) -> int:
    return int(predict_probability(model, features) >= model.threshold)


def layer_norm(values: list[float], epsilon: float = 1e-5) -> list[float]:
    if not values:
        raise ValueError("values must be non-empty")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    denominator = math.sqrt(variance + epsilon)
    return [(value - mean) / denominator for value in values]


def batch_norm_inference(
    values: list[float],
    running_mean: tuple[float, ...],
    running_var: tuple[float, ...],
    gamma: tuple[float, ...],
    beta: tuple[float, ...],
    epsilon: float = 1e-5,
) -> list[float]:
    size = len(values)
    if not all(
        len(item) == size
        for item in (running_mean, running_var, gamma, beta)
    ):
        raise ValueError("batch normalization shape mismatch")
    if epsilon <= 0.0 or any(value < 0.0 for value in running_var):
        raise ValueError("invalid batch normalization parameters")
    return [
        scale * ((value - mean) / math.sqrt(variance + epsilon)) + shift
        for value, mean, variance, scale, shift in zip(
            values, running_mean, running_var, gamma, beta, strict=True
        )
    ]


def he_initialize(
    fan_in: int,
    fan_out: int,
    seed: int = 7,
) -> tuple[tuple[float, ...], ...]:
    if fan_in <= 0 or fan_out <= 0:
        raise ValueError("fan sizes must be > 0")
    rng = random.Random(seed)
    scale = math.sqrt(2.0 / fan_in)
    return tuple(
        tuple(rng.gauss(0.0, scale) for _ in range(fan_in))
        for _ in range(fan_out)
    )


def load_model(path: Path) -> MLPModel:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"invalid artifact line: {raw}")
        values[key.strip()] = value.strip()

    required = {
        "version",
        "input_size",
        "hidden_size",
        "mean",
        "std",
        "w1",
        "b1",
        "w2",
        "b2",
        "threshold",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")

    def vector(key: str) -> tuple[float, ...]:
        return tuple(float(item) for item in values[key].split(",") if item)

    input_size = int(values["input_size"])
    hidden_size = int(values["hidden_size"])
    rows = tuple(
        tuple(float(item) for item in row.split(",") if item)
        for row in values["w1"].split(";")
        if row
    )

    model = MLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        mean=vector("mean"),
        std=vector("std"),
        w1=rows,
        b1=vector("b1"),
        w2=vector("w2"),
        b2=float(values["b2"]),
        threshold=float(values["threshold"]),
        version=int(values["version"]),
    )
    _validate_model(model)
    return model


def _validate_model(model: MLPModel) -> None:
    if model.version != 1:
        raise ValueError("unsupported model version")
    if len(model.mean) != model.input_size or len(model.std) != model.input_size:
        raise ValueError("normalization metadata shape mismatch")
    if len(model.w1) != model.hidden_size or len(model.b1) != model.hidden_size:
        raise ValueError("hidden layer shape mismatch")
    if any(len(row) != model.input_size for row in model.w1):
        raise ValueError("w1 row shape mismatch")
    if len(model.w2) != model.hidden_size:
        raise ValueError("w2 shape mismatch")
    if not 0.0 <= model.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

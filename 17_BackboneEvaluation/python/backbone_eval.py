"""Backbone evaluation scorecard for Module 17."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Backbone:
    name: str
    accuracy: float
    latency_ms: float
    params_m: float
    readiness: float


@dataclass(frozen=True)
class Weights:
    accuracy: float
    latency: float
    params: float
    readiness: float


def load(path: Path) -> tuple[Weights, list[Backbone]]:
    weights: Weights | None = None
    models: list[Backbone] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("version="):
            continue

        if line.startswith("weights="):
            pairs = dict(
                item.split(":")
                for item in line.removeprefix("weights=").split(",")
            )
            weights = Weights(
                accuracy=float(pairs["accuracy"]),
                latency=float(pairs["latency"]),
                params=float(pairs["params"]),
                readiness=float(pairs["readiness"]),
            )
        elif line.startswith("backbone="):
            parts = line.split(",")
            name = parts[0].split("=", 1)[1]
            pairs = dict(item.split(":") for item in parts[1:])
            models.append(
                Backbone(
                    name=name,
                    accuracy=float(pairs["accuracy"]),
                    latency_ms=float(pairs["latency_ms"]),
                    params_m=float(pairs["params_m"]),
                    readiness=float(pairs["readiness"]),
                )
            )

    if weights is None:
        raise ValueError("missing weights")
    return weights, models


def rank(
    weights: Weights,
    models: list[Backbone],
) -> list[tuple[str, float]]:
    if not models:
        raise ValueError("no backbones")

    min_latency = min(model.latency_ms for model in models)
    max_latency = max(model.latency_ms for model in models)
    min_params = min(model.params_m for model in models)
    max_params = max(model.params_m for model in models)

    def inverse_norm(value: float, low: float, high: float) -> float:
        if high == low:
            return 1.0
        return 1.0 - (value - low) / (high - low)

    scored: list[tuple[str, float]] = []
    for model in models:
        score = (
            weights.accuracy * model.accuracy
            + weights.latency
            * inverse_norm(model.latency_ms, min_latency, max_latency)
            + weights.params
            * inverse_norm(model.params_m, min_params, max_params)
            + weights.readiness * model.readiness
        )
        scored.append((model.name, score))

    return sorted(scored, key=lambda item: item[1], reverse=True)

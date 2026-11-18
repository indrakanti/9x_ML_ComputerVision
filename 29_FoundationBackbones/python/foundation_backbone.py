"""Reusable foundation-feature contract for Module 29."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LinearAdapter:
    outputs: int
    weight: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]


@dataclass(frozen=True)
class FoundationArtifact:
    feature_revision: str
    feature_dimension: int
    backbone_parameters: int
    features: tuple[float, ...]
    classification: LinearAdapter
    geometry: LinearAdapter


def _vector(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def _matrix(text: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row.split(","))
        for row in text.split(";")
    )


def _adapter(
    values: dict[str, str],
    prefix: str,
    feature_dimension: int,
) -> LinearAdapter:
    outputs = int(values[f"{prefix}_outputs"])
    weight = _matrix(values[f"{prefix}_weight"])
    bias = _vector(values[f"{prefix}_bias"])

    if len(weight) != outputs:
        raise ValueError(f"{prefix} row count mismatch")
    if any(len(row) != feature_dimension for row in weight):
        raise ValueError(f"{prefix} feature dimension mismatch")
    if len(bias) != outputs:
        raise ValueError(f"{prefix} bias count mismatch")

    return LinearAdapter(outputs=outputs, weight=weight, bias=bias)


def load_artifact(path: Path) -> FoundationArtifact:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"invalid artifact line: {raw}")
        values[key] = value

    if int(values["version"]) != 1:
        raise ValueError("unsupported artifact version")

    feature_dimension = int(values["feature_dimension"])
    features = _vector(values["features"])
    if len(features) != feature_dimension:
        raise ValueError("shared feature dimension mismatch")

    revision = values["feature_revision"]
    if not revision:
        raise ValueError("feature revision must be non-empty")

    return FoundationArtifact(
        feature_revision=revision,
        feature_dimension=feature_dimension,
        backbone_parameters=int(values["backbone_parameters"]),
        features=features,
        classification=_adapter(
            values,
            "classification",
            feature_dimension,
        ),
        geometry=_adapter(
            values,
            "geometry",
            feature_dimension,
        ),
    )


def adapter_forward(
    features: tuple[float, ...],
    adapter: LinearAdapter,
) -> list[float]:
    return [
        adapter.bias[out]
        + sum(
            adapter.weight[out][index] * features[index]
            for index in range(len(features))
        )
        for out in range(adapter.outputs)
    ]


def adapter_parameter_count(adapter: LinearAdapter) -> int:
    if not adapter.weight:
        return len(adapter.bias)
    return (
        len(adapter.weight) * len(adapter.weight[0])
        + len(adapter.bias)
    )


def total_adapter_parameters(artifact: FoundationArtifact) -> int:
    return (
        adapter_parameter_count(artifact.classification)
        + adapter_parameter_count(artifact.geometry)
    )


def adapter_overhead_ratio(artifact: FoundationArtifact) -> float:
    if artifact.backbone_parameters <= 0:
        raise ValueError("backbone parameter count must be > 0")
    return total_adapter_parameters(artifact) / artifact.backbone_parameters

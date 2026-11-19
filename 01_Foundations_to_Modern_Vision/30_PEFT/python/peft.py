"""Linear probing, adapters, and LoRA for Module 30."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class PEFTArtifact:
    feature_dimension: int
    features: tuple[float, ...]
    probe_outputs: int
    probe_weight: tuple[tuple[float, ...], ...]
    probe_bias: tuple[float, ...]
    adapter_hidden: int
    adapter_down: tuple[tuple[float, ...], ...]
    adapter_down_bias: tuple[float, ...]
    adapter_up: tuple[tuple[float, ...], ...]
    adapter_up_bias: tuple[float, ...]
    lora_outputs: int
    base_weight: tuple[tuple[float, ...], ...]
    base_bias: tuple[float, ...]
    lora_rank: int
    lora_alpha: float
    lora_a: tuple[tuple[float, ...], ...]
    lora_b: tuple[tuple[float, ...], ...]


def _vector(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def _matrix(text: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row.split(","))
        for row in text.split(";")
    )


def _validate_matrix(
    matrix: tuple[tuple[float, ...], ...],
    rows: int,
    cols: int,
    name: str,
) -> None:
    if len(matrix) != rows:
        raise ValueError(f"{name} row count mismatch")
    if any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} column count mismatch")


def load_artifact(path: Path) -> PEFTArtifact:
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
    probe_outputs = int(values["probe_outputs"])
    adapter_hidden = int(values["adapter_hidden"])
    lora_outputs = int(values["lora_outputs"])
    lora_rank = int(values["lora_rank"])

    artifact = PEFTArtifact(
        feature_dimension=feature_dimension,
        features=_vector(values["features"]),
        probe_outputs=probe_outputs,
        probe_weight=_matrix(values["probe_weight"]),
        probe_bias=_vector(values["probe_bias"]),
        adapter_hidden=adapter_hidden,
        adapter_down=_matrix(values["adapter_down"]),
        adapter_down_bias=_vector(values["adapter_down_bias"]),
        adapter_up=_matrix(values["adapter_up"]),
        adapter_up_bias=_vector(values["adapter_up_bias"]),
        lora_outputs=lora_outputs,
        base_weight=_matrix(values["base_weight"]),
        base_bias=_vector(values["base_bias"]),
        lora_rank=lora_rank,
        lora_alpha=float(values["lora_alpha"]),
        lora_a=_matrix(values["lora_a"]),
        lora_b=_matrix(values["lora_b"]),
    )

    if len(artifact.features) != feature_dimension:
        raise ValueError("feature dimension mismatch")

    _validate_matrix(
        artifact.probe_weight,
        probe_outputs,
        feature_dimension,
        "probe_weight",
    )
    if len(artifact.probe_bias) != probe_outputs:
        raise ValueError("probe bias mismatch")

    _validate_matrix(
        artifact.adapter_down,
        adapter_hidden,
        feature_dimension,
        "adapter_down",
    )
    _validate_matrix(
        artifact.adapter_up,
        feature_dimension,
        adapter_hidden,
        "adapter_up",
    )
    if len(artifact.adapter_down_bias) != adapter_hidden:
        raise ValueError("adapter down bias mismatch")
    if len(artifact.adapter_up_bias) != feature_dimension:
        raise ValueError("adapter up bias mismatch")

    _validate_matrix(
        artifact.base_weight,
        lora_outputs,
        feature_dimension,
        "base_weight",
    )
    if len(artifact.base_bias) != lora_outputs:
        raise ValueError("base bias mismatch")
    _validate_matrix(
        artifact.lora_a,
        lora_rank,
        feature_dimension,
        "lora_a",
    )
    _validate_matrix(
        artifact.lora_b,
        lora_outputs,
        lora_rank,
        "lora_b",
    )
    if lora_rank <= 0:
        raise ValueError("LoRA rank must be > 0")

    return artifact


def linear(
    vector: tuple[float, ...] | list[float],
    weight: tuple[tuple[float, ...], ...],
    bias: tuple[float, ...],
) -> list[float]:
    return [
        bias[out]
        + sum(
            weight[out][index] * vector[index]
            for index in range(len(vector))
        )
        for out in range(len(weight))
    ]


def probe_forward(artifact: PEFTArtifact) -> list[float]:
    return linear(
        artifact.features,
        artifact.probe_weight,
        artifact.probe_bias,
    )


def adapter_forward(artifact: PEFTArtifact) -> list[float]:
    hidden = linear(
        artifact.features,
        artifact.adapter_down,
        artifact.adapter_down_bias,
    )
    activated = [max(0.0, value) for value in hidden]
    delta = linear(
        activated,
        artifact.adapter_up,
        artifact.adapter_up_bias,
    )
    return [
        feature + update
        for feature, update in zip(
            artifact.features,
            delta,
            strict=True,
        )
    ]


def lora_forward(artifact: PEFTArtifact) -> list[float]:
    base = linear(
        artifact.features,
        artifact.base_weight,
        artifact.base_bias,
    )
    low_rank = linear(
        artifact.features,
        artifact.lora_a,
        (0.0,) * artifact.lora_rank,
    )
    update = linear(
        low_rank,
        artifact.lora_b,
        (0.0,) * artifact.lora_outputs,
    )
    scale = artifact.lora_alpha / artifact.lora_rank
    return [
        base_value + scale * update_value
        for base_value, update_value in zip(
            base,
            update,
            strict=True,
        )
    ]


def parameter_counts(artifact: PEFTArtifact) -> dict[str, int]:
    probe = (
        artifact.probe_outputs * artifact.feature_dimension
        + artifact.probe_outputs
    )
    adapter = (
        artifact.adapter_hidden * artifact.feature_dimension
        + artifact.adapter_hidden
        + artifact.feature_dimension * artifact.adapter_hidden
        + artifact.feature_dimension
    )
    lora = artifact.lora_rank * (
        artifact.feature_dimension + artifact.lora_outputs
    )
    full_linear = (
        artifact.lora_outputs * artifact.feature_dimension
        + artifact.lora_outputs
    )
    return {
        "probe": probe,
        "adapter": adapter,
        "lora": lora,
        "full_linear": full_linear,
    }


def pytorch_lora_forward(artifact: PEFTArtifact) -> Tensor:
    features = torch.tensor(artifact.features, dtype=torch.float64)
    base_weight = torch.tensor(
        artifact.base_weight,
        dtype=torch.float64,
    )
    base_bias = torch.tensor(artifact.base_bias, dtype=torch.float64)
    lora_a = torch.tensor(artifact.lora_a, dtype=torch.float64)
    lora_b = torch.tensor(artifact.lora_b, dtype=torch.float64)
    scale = artifact.lora_alpha / artifact.lora_rank

    base = base_weight @ features + base_bias
    update = lora_b @ (lora_a @ features)
    return base + scale * update

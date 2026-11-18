"""Distillation, pruning, and quantization for Module 31."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class CompressionArtifact:
    temperature: float
    teacher_logits: tuple[float, ...]
    student_logits: tuple[float, ...]
    prune_threshold: float
    weights: tuple[float, ...]
    quant_values: tuple[float, ...]


def _vector(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def load_artifact(path: Path) -> CompressionArtifact:
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

    artifact = CompressionArtifact(
        temperature=float(values["temperature"]),
        teacher_logits=_vector(values["teacher_logits"]),
        student_logits=_vector(values["student_logits"]),
        prune_threshold=float(values["prune_threshold"]),
        weights=_vector(values["weights"]),
        quant_values=_vector(values["quant_values"]),
    )

    if artifact.temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    if len(artifact.teacher_logits) != len(artifact.student_logits):
        raise ValueError("teacher/student logit length mismatch")
    if artifact.prune_threshold < 0.0:
        raise ValueError("prune threshold must be >= 0")
    if not artifact.quant_values:
        raise ValueError("quantization values must be non-empty")

    return artifact


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def distillation_loss(artifact: CompressionArtifact) -> float:
    temperature = artifact.temperature
    teacher = softmax(
        [value / temperature for value in artifact.teacher_logits]
    )
    student = softmax(
        [value / temperature for value in artifact.student_logits]
    )
    cross_entropy = -sum(
        target * math.log(prediction)
        for target, prediction in zip(
            teacher,
            student,
            strict=True,
        )
    )
    return temperature * temperature * cross_entropy


def prune_weights(
    weights: tuple[float, ...],
    threshold: float,
) -> list[float]:
    return [
        0.0 if abs(value) < threshold else value
        for value in weights
    ]


def sparsity(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    zeros = sum(value == 0.0 for value in values)
    return zeros / len(values)


def quantize_symmetric_int8(
    values: tuple[float, ...],
) -> tuple[float, list[int], list[float], float]:
    maximum = max(abs(value) for value in values)
    scale = 1.0 if maximum == 0.0 else maximum / 127.0

    quantized: list[int] = []
    for value in values:
        rounded = int(round(value / scale))
        quantized.append(max(-127, min(127, rounded)))

    dequantized = [scale * value for value in quantized]
    max_error = max(
        abs(original - restored)
        for original, restored in zip(
            values,
            dequantized,
            strict=True,
        )
    )
    return scale, quantized, dequantized, max_error


def pytorch_distillation_loss(
    artifact: CompressionArtifact,
) -> Tensor:
    temperature = artifact.temperature
    teacher = torch.tensor(
        artifact.teacher_logits,
        dtype=torch.float64,
    )
    student = torch.tensor(
        artifact.student_logits,
        dtype=torch.float64,
    )
    teacher_prob = torch.softmax(teacher / temperature, dim=0)
    student_log_prob = torch.log_softmax(student / temperature, dim=0)
    return (
        -(teacher_prob * student_log_prob).sum()
        * temperature
        * temperature
    )

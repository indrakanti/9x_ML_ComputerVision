"""DINO-style self-distillation math for Module 28."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class DINOArtifact:
    dimension: int
    student_temperature: float
    teacher_temperature: float
    center: tuple[float, ...]
    student_view_a: tuple[float, ...]
    student_view_b: tuple[float, ...]
    teacher_view_a: tuple[float, ...]
    teacher_view_b: tuple[float, ...]
    student_params: tuple[float, ...]
    teacher_params: tuple[float, ...]
    teacher_momentum: float


def _vector(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def load_artifact(path: Path) -> DINOArtifact:
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

    dimension = int(values["dimension"])
    artifact = DINOArtifact(
        dimension=dimension,
        student_temperature=float(values["student_temperature"]),
        teacher_temperature=float(values["teacher_temperature"]),
        center=_vector(values["center"]),
        student_view_a=_vector(values["student_view_a"]),
        student_view_b=_vector(values["student_view_b"]),
        teacher_view_a=_vector(values["teacher_view_a"]),
        teacher_view_b=_vector(values["teacher_view_b"]),
        student_params=_vector(values["student_params"]),
        teacher_params=_vector(values["teacher_params"]),
        teacher_momentum=float(values["teacher_momentum"]),
    )

    vectors = (
        artifact.center,
        artifact.student_view_a,
        artifact.student_view_b,
        artifact.teacher_view_a,
        artifact.teacher_view_b,
    )
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("logit/center dimension mismatch")
    if len(artifact.student_params) != len(artifact.teacher_params):
        raise ValueError("parameter vector length mismatch")
    if artifact.student_temperature <= 0.0:
        raise ValueError("student temperature must be > 0")
    if artifact.teacher_temperature <= 0.0:
        raise ValueError("teacher temperature must be > 0")
    if not 0.0 <= artifact.teacher_momentum <= 1.0:
        raise ValueError("teacher momentum must be in [0, 1]")

    return artifact


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def student_distribution(
    logits: tuple[float, ...],
    temperature: float,
) -> list[float]:
    return softmax([value / temperature for value in logits])


def teacher_distribution(
    logits: tuple[float, ...],
    center: tuple[float, ...],
    temperature: float,
) -> list[float]:
    return softmax(
        [
            (value - center_value) / temperature
            for value, center_value in zip(logits, center, strict=True)
        ]
    )


def cross_entropy(
    target: list[float],
    prediction: list[float],
) -> float:
    return -sum(
        target_value * math.log(prediction_value)
        for target_value, prediction_value in zip(
            target,
            prediction,
            strict=True,
        )
    )


def symmetric_cross_view_loss(artifact: DINOArtifact) -> float:
    student_a = student_distribution(
        artifact.student_view_a,
        artifact.student_temperature,
    )
    student_b = student_distribution(
        artifact.student_view_b,
        artifact.student_temperature,
    )
    teacher_a = teacher_distribution(
        artifact.teacher_view_a,
        artifact.center,
        artifact.teacher_temperature,
    )
    teacher_b = teacher_distribution(
        artifact.teacher_view_b,
        artifact.center,
        artifact.teacher_temperature,
    )
    return 0.5 * (
        cross_entropy(teacher_b, student_a)
        + cross_entropy(teacher_a, student_b)
    )


def ema_teacher_params(artifact: DINOArtifact) -> list[float]:
    momentum = artifact.teacher_momentum
    return [
        momentum * teacher + (1.0 - momentum) * student
        for teacher, student in zip(
            artifact.teacher_params,
            artifact.student_params,
            strict=True,
        )
    ]


def pytorch_loss(artifact: DINOArtifact) -> Tensor:
    center = torch.tensor(artifact.center, dtype=torch.float64)
    student_a = torch.tensor(artifact.student_view_a, dtype=torch.float64)
    student_b = torch.tensor(artifact.student_view_b, dtype=torch.float64)
    teacher_a = torch.tensor(artifact.teacher_view_a, dtype=torch.float64)
    teacher_b = torch.tensor(artifact.teacher_view_b, dtype=torch.float64)

    student_a_log = torch.log_softmax(
        student_a / artifact.student_temperature,
        dim=0,
    )
    student_b_log = torch.log_softmax(
        student_b / artifact.student_temperature,
        dim=0,
    )
    teacher_a_prob = torch.softmax(
        (teacher_a - center) / artifact.teacher_temperature,
        dim=0,
    )
    teacher_b_prob = torch.softmax(
        (teacher_b - center) / artifact.teacher_temperature,
        dim=0,
    )

    first = -(teacher_b_prob * student_a_log).sum()
    second = -(teacher_a_prob * student_b_log).sum()
    return 0.5 * (first + second)

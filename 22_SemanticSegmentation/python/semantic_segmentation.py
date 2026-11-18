"""Semantic segmentation metrics for Module 22."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SemanticArtifact:
    classes: int
    height: int
    width: int
    ignore_label: int
    ground_truth: tuple[int, ...]
    logits: tuple[tuple[float, ...], ...]


def load_artifact(path: Path) -> SemanticArtifact:
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

    classes = int(values["classes"])
    height = int(values["height"])
    width = int(values["width"])
    ground_truth = tuple(
        int(value) for value in values["ground_truth"].split(",")
    )
    logits = tuple(
        tuple(float(value) for value in row.split(","))
        for row in values["logits"].split(";")
    )

    pixels = height * width
    if len(ground_truth) != pixels or len(logits) != pixels:
        raise ValueError("pixel count mismatch")
    if any(len(row) != classes for row in logits):
        raise ValueError("logit width mismatch")

    return SemanticArtifact(
        classes=classes,
        height=height,
        width=width,
        ignore_label=int(values["ignore_label"]),
        ground_truth=ground_truth,
        logits=logits,
    )


def predict_labels(artifact: SemanticArtifact) -> list[int]:
    return [
        max(range(artifact.classes), key=row.__getitem__)
        for row in artifact.logits
    ]


def confusion_matrix(
    artifact: SemanticArtifact,
    predictions: list[int],
) -> list[list[int]]:
    if len(predictions) != len(artifact.ground_truth):
        raise ValueError("prediction size mismatch")

    matrix = [
        [0 for _ in range(artifact.classes)]
        for _ in range(artifact.classes)
    ]

    for truth, prediction in zip(
        artifact.ground_truth,
        predictions,
        strict=True,
    ):
        if truth == artifact.ignore_label:
            continue
        if not 0 <= truth < artifact.classes:
            raise ValueError("ground-truth class out of range")
        if not 0 <= prediction < artifact.classes:
            raise ValueError("prediction class out of range")
        matrix[truth][prediction] += 1
    return matrix


def class_iou(matrix: list[list[int]], class_id: int) -> float:
    tp = matrix[class_id][class_id]
    fp = sum(
        matrix[row][class_id]
        for row in range(len(matrix))
        if row != class_id
    )
    fn = sum(
        matrix[class_id][column]
        for column in range(len(matrix[class_id]))
        if column != class_id
    )
    denominator = tp + fp + fn
    if denominator == 0:
        return float("nan")
    return tp / denominator


def mean_iou(matrix: list[list[int]]) -> float:
    values = [
        class_iou(matrix, class_id)
        for class_id in range(len(matrix))
    ]
    valid = [value for value in values if value == value]
    if not valid:
        raise ValueError("no valid classes for mIoU")
    return sum(valid) / len(valid)

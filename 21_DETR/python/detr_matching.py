"""DETR-style matching reference for Module 21."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class MatchingArtifact:
    classes: int
    pred_boxes: tuple[Box, ...]
    pred_probs: tuple[tuple[float, ...], ...]
    gt_boxes: tuple[Box, ...]
    gt_classes: tuple[int, ...]
    lambda_cls: float
    lambda_l1: float
    lambda_iou: float


def _box(text: str) -> Box:
    values = [float(value) for value in text.split(",")]
    if len(values) != 4:
        raise ValueError("box requires four values")
    return Box(*values)


def _rows(text: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row.split(","))
        for row in text.split(";")
    )


def load_artifact(path: Path) -> MatchingArtifact:
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
    pred_boxes = tuple(
        _box(item)
        for item in values["pred_boxes"].split(";")
    )
    pred_probs = _rows(values["pred_probs"])
    gt_boxes = tuple(_box(item) for item in values["gt_boxes"].split(";"))
    gt_classes = tuple(
        int(value)
        for value in values["gt_classes"].split(",")
    )

    if len(pred_boxes) != len(pred_probs):
        raise ValueError("prediction arrays must have equal length")
    if len(gt_boxes) != len(gt_classes):
        raise ValueError("ground truth arrays must have equal length")
    if any(len(row) != classes for row in pred_probs):
        raise ValueError("class probability width mismatch")

    return MatchingArtifact(
        classes=classes,
        pred_boxes=pred_boxes,
        pred_probs=pred_probs,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        lambda_cls=float(values["lambda_cls"]),
        lambda_l1=float(values["lambda_l1"]),
        lambda_iou=float(values["lambda_iou"]),
    )


def iou(first: Box, second: Box) -> float:
    ix1 = max(first.x1, second.x1)
    iy1 = max(first.y1, second.y1)
    ix2 = min(first.x2, second.x2)
    iy2 = min(first.y2, second.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (first.x2 - first.x1) * (first.y2 - first.y1)
    area_b = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / (area_a + area_b - intersection)


def l1(first: Box, second: Box) -> float:
    return (
        abs(first.x1 - second.x1)
        + abs(first.y1 - second.y1)
        + abs(first.x2 - second.x2)
        + abs(first.y2 - second.y2)
    )


def cost_matrix(artifact: MatchingArtifact) -> list[list[float]]:
    matrix: list[list[float]] = []
    for pred_index, pred_box in enumerate(artifact.pred_boxes):
        row: list[float] = []
        for gt_index, gt_box in enumerate(artifact.gt_boxes):
            target_class = artifact.gt_classes[gt_index]
            cls_cost = -artifact.pred_probs[pred_index][target_class]
            row.append(
                artifact.lambda_cls * cls_cost
                + artifact.lambda_l1 * l1(pred_box, gt_box)
                + artifact.lambda_iou * (1.0 - iou(pred_box, gt_box))
            )
        matrix.append(row)
    return matrix


def optimal_assignment(
    matrix: list[list[float]],
) -> list[tuple[int, int]]:
    if not matrix or not matrix[0]:
        raise ValueError("cost matrix must be non-empty")

    predictions = len(matrix)
    targets = len(matrix[0])
    if predictions < targets:
        raise ValueError("reference matcher expects predictions >= targets")

    best_cost = float("inf")
    best: list[tuple[int, int]] = []
    for chosen_predictions in itertools.permutations(
        range(predictions),
        targets,
    ):
        total = sum(
            matrix[pred_index][target_index]
            for target_index, pred_index in enumerate(chosen_predictions)
        )
        if total < best_cost:
            best_cost = total
            best = [
                (pred_index, target_index)
                for target_index, pred_index in enumerate(chosen_predictions)
            ]
    return best

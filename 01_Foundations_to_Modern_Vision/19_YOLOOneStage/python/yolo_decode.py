"""YOLO-style decode and NMS for Module 19."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class YoloArtifact:
    image_w: int
    image_h: int
    grid_w: int
    grid_h: int
    anchor_w: float
    anchor_h: float
    classes: int
    score_threshold: float
    nms_iou: float
    predictions: tuple[tuple[float, ...], ...]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_artifact(path: Path) -> YoloArtifact:
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
    predictions = tuple(
        tuple(float(value) for value in row.split(","))
        for row in values["predictions"].split(";")
    )
    expected_width = 5 + classes
    if any(len(row) != expected_width for row in predictions):
        raise ValueError("prediction row width mismatch")

    grid_w = int(values["grid_w"])
    grid_h = int(values["grid_h"])
    if len(predictions) != grid_w * grid_h:
        raise ValueError("prediction count must equal grid cells")

    return YoloArtifact(
        image_w=int(values["image_w"]),
        image_h=int(values["image_h"]),
        grid_w=grid_w,
        grid_h=grid_h,
        anchor_w=float(values["anchor_w"]),
        anchor_h=float(values["anchor_h"]),
        classes=classes,
        score_threshold=float(values["score_threshold"]),
        nms_iou=float(values["nms_iou"]),
        predictions=predictions,
    )


def box_iou(first: Detection, second: Detection) -> float:
    ix1 = max(first.x1, second.x1)
    iy1 = max(first.y1, second.y1)
    ix2 = min(first.x2, second.x2)
    iy2 = min(first.y2, second.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
    second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
    return intersection / (first_area + second_area - intersection)


def decode(artifact: YoloArtifact) -> list[Detection]:
    detections: list[Detection] = []

    for index, prediction in enumerate(artifact.predictions):
        gy = index // artifact.grid_w
        gx = index % artifact.grid_w
        tx, ty, tw, th, objectness = prediction[:5]
        class_logits = prediction[5:]

        class_scores = [
            sigmoid(objectness) * sigmoid(logit)
            for logit in class_logits
        ]
        class_id = max(range(artifact.classes), key=class_scores.__getitem__)
        score = class_scores[class_id]
        if score < artifact.score_threshold:
            continue

        cx = (sigmoid(tx) + gx) / artifact.grid_w
        cy = (sigmoid(ty) + gy) / artifact.grid_h
        width = artifact.anchor_w * math.exp(tw) / artifact.image_w
        height = artifact.anchor_h * math.exp(th) / artifact.image_h

        detections.append(
            Detection(
                x1=cx - width / 2.0,
                y1=cy - height / 2.0,
                x2=cx + width / 2.0,
                y2=cy + height / 2.0,
                score=score,
                class_id=class_id,
            )
        )
    return detections


def nms(
    detections: list[Detection],
    threshold: float,
) -> list[Detection]:
    order = sorted(
        detections,
        key=lambda detection: -detection.score,
    )
    kept: list[Detection] = []

    while order:
        current = order.pop(0)
        kept.append(current)
        order = [
            candidate
            for candidate in order
            if candidate.class_id != current.class_id
            or box_iou(current, candidate) <= threshold
        ]
    return kept


def final_detections(artifact: YoloArtifact) -> list[Detection]:
    return nms(decode(artifact), artifact.nms_iou)

"""Detection geometry and metrics for Module 18."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class DetectionArtifact:
    iou_a: Box
    iou_b: Box
    boxes: tuple[Box, ...]
    scores: tuple[float, ...]
    classes: tuple[int, ...]
    nms_iou: float


def parse_box(text: str) -> Box:
    values = [float(value) for value in text.split(",")]
    if len(values) != 4:
        raise ValueError("box requires four coordinates")
    return Box(*values)


def load_artifact(path: Path) -> DetectionArtifact:
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

    boxes = tuple(parse_box(item) for item in values["boxes"].split(";"))
    scores = tuple(float(value) for value in values["scores"].split(","))
    classes = tuple(int(value) for value in values["classes"].split(","))
    if not (len(boxes) == len(scores) == len(classes)):
        raise ValueError("detection arrays must have equal length")

    return DetectionArtifact(
        iou_a=parse_box(values["iou_a"]),
        iou_b=parse_box(values["iou_b"]),
        boxes=boxes,
        scores=scores,
        classes=classes,
        nms_iou=float(values["nms_iou"]),
    )


def iou(first: Box, second: Box) -> float:
    first_area = first.area()
    second_area = second.area()

    ix1 = max(first.x1, second.x1)
    iy1 = max(first.y1, second.y1)
    ix2 = min(first.x2, second.x2)
    iy2 = min(first.y2, second.y2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = first_area + second_area - intersection
    return intersection / union


def class_aware_nms(
    boxes: tuple[Box, ...],
    scores: tuple[float, ...],
    classes: tuple[int, ...],
    threshold: float,
) -> list[int]:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("NMS threshold must be in [0, 1]")

    order = sorted(range(len(boxes)), key=lambda index: (-scores[index], index))
    kept: list[int] = []

    while order:
        current = order.pop(0)
        kept.append(current)
        order = [
            candidate
            for candidate in order
            if classes[candidate] != classes[current]
            or iou(boxes[current], boxes[candidate]) <= threshold
        ]
    return kept


def average_precision(
    scores: list[float],
    true_positive: list[bool],
    ground_truth_count: int,
) -> float:
    if len(scores) != len(true_positive):
        raise ValueError("scores and labels must have equal length")
    if ground_truth_count <= 0:
        raise ValueError("ground_truth_count must be > 0")

    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]

    for index in order:
        if true_positive[index]:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / ground_truth_count)
        precisions.append(tp / (tp + fp))

    precisions.append(0.0)
    recalls.append(recalls[-1])

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    ap = 0.0
    for index in range(1, len(recalls)):
        if recalls[index] != recalls[index - 1]:
            ap += (
                recalls[index] - recalls[index - 1]
            ) * precisions[index]
    return ap

"""Two-stage proposal generation for Module 20."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class ProposalArtifact:
    image_w: int
    image_h: int
    anchors: tuple[Box, ...]
    deltas: tuple[tuple[float, float, float, float], ...]
    objectness: tuple[float, ...]
    pre_nms_topk: int
    post_nms_topk: int
    nms_iou: float


def _box(text: str) -> Box:
    values = [float(value) for value in text.split(",")]
    if len(values) != 4:
        raise ValueError("box requires four values")
    return Box(*values)


def load_artifact(path: Path) -> ProposalArtifact:
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

    anchors = tuple(_box(item) for item in values["anchors"].split(";"))
    deltas = tuple(
        tuple(float(value) for value in item.split(","))
        for item in values["deltas"].split(";")
    )
    objectness = tuple(
        float(value) for value in values["objectness"].split(",")
    )

    if not (len(anchors) == len(deltas) == len(objectness)):
        raise ValueError("proposal arrays must have equal length")
    if any(len(delta) != 4 for delta in deltas):
        raise ValueError("each proposal delta must have four values")

    typed_deltas = tuple(
        (delta[0], delta[1], delta[2], delta[3])
        for delta in deltas
    )
    return ProposalArtifact(
        image_w=int(values["image_w"]),
        image_h=int(values["image_h"]),
        anchors=anchors,
        deltas=typed_deltas,
        objectness=objectness,
        pre_nms_topk=int(values["pre_nms_topk"]),
        post_nms_topk=int(values["post_nms_topk"]),
        nms_iou=float(values["nms_iou"]),
    )


def decode(anchor: Box, delta: tuple[float, float, float, float]) -> Box:
    width = anchor.x2 - anchor.x1
    height = anchor.y2 - anchor.y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError("anchor must have positive dimensions")

    center_x = (anchor.x1 + anchor.x2) / 2.0
    center_y = (anchor.y1 + anchor.y2) / 2.0
    tx, ty, tw, th = delta

    predicted_x = center_x + tx * width
    predicted_y = center_y + ty * height
    predicted_w = width * math.exp(tw)
    predicted_h = height * math.exp(th)

    return Box(
        predicted_x - predicted_w / 2.0,
        predicted_y - predicted_h / 2.0,
        predicted_x + predicted_w / 2.0,
        predicted_y + predicted_h / 2.0,
    )


def clip(box: Box, image_w: int, image_h: int) -> Box:
    return Box(
        max(0.0, min(float(image_w), box.x1)),
        max(0.0, min(float(image_h), box.y1)),
        max(0.0, min(float(image_w), box.x2)),
        max(0.0, min(float(image_h), box.y2)),
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


def proposals(
    artifact: ProposalArtifact,
) -> list[tuple[Box, float]]:
    decoded = [
        (
            clip(
                decode(anchor, delta),
                artifact.image_w,
                artifact.image_h,
            ),
            score,
        )
        for anchor, delta, score in zip(
            artifact.anchors,
            artifact.deltas,
            artifact.objectness,
            strict=True,
        )
    ]

    ordered = sorted(decoded, key=lambda item: -item[1])[
        : artifact.pre_nms_topk
    ]
    kept: list[tuple[Box, float]] = []

    while ordered and len(kept) < artifact.post_nms_topk:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [
            candidate
            for candidate in ordered
            if iou(current[0], candidate[0]) <= artifact.nms_iou
        ]
    return kept


def roi_pool_2x2(
    feature_map: list[list[float]],
    box: Box,
) -> tuple[float, float, float, float]:
    height = len(feature_map)
    width = len(feature_map[0])
    x1 = max(0, min(width - 1, int(math.floor(box.x1))))
    y1 = max(0, min(height - 1, int(math.floor(box.y1))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box.x2))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box.y2))))

    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    x_ranges = ((x1, max(x1 + 1, mid_x)), (mid_x, x2))
    y_ranges = ((y1, max(y1 + 1, mid_y)), (mid_y, y2))

    pooled: list[float] = []
    for ya, yb in y_ranges:
        for xa, xb in x_ranges:
            values = [
                feature_map[y][x]
                for y in range(ya, yb)
                for x in range(xa, xb)
            ]
            pooled.append(max(values))
    return pooled[0], pooled[1], pooled[2], pooled[3]

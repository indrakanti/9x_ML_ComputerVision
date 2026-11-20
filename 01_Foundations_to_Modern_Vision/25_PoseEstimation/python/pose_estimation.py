"""Heatmap keypoint decoding for Module 25."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseArtifact:
    heatmap_h: int
    heatmap_w: int
    image_h: int
    image_w: int
    keypoints: int
    heatmaps: tuple[tuple[float, ...], ...]
    ground_truth: tuple[tuple[float, float], ...]
    visible: tuple[bool, ...]
    normalization: float
    pck_alpha: float


def load_artifact(path: Path) -> PoseArtifact:
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

    heatmap_h = int(values["heatmap_h"])
    heatmap_w = int(values["heatmap_w"])
    keypoints = int(values["keypoints"])
    heatmaps = tuple(
        tuple(float(value) for value in row.split(","))
        for row in values["heatmaps"].split(";")
    )
    ground_truth = tuple(
        tuple(float(value) for value in row.split(","))
        for row in values["ground_truth"].split(";")
    )
    visible = tuple(
        value == "1"
        for value in values["visible"].split(",")
    )

    if len(heatmaps) != keypoints:
        raise ValueError("heatmap count mismatch")
    if any(len(row) != heatmap_h * heatmap_w for row in heatmaps):
        raise ValueError("heatmap pixel count mismatch")
    if len(ground_truth) != keypoints or len(visible) != keypoints:
        raise ValueError("ground-truth metadata count mismatch")
    if any(len(point) != 2 for point in ground_truth):
        raise ValueError("ground-truth point width mismatch")

    typed_ground_truth = tuple(
        (point[0], point[1])
        for point in ground_truth
    )

    return PoseArtifact(
        heatmap_h=heatmap_h,
        heatmap_w=heatmap_w,
        image_h=int(values["image_h"]),
        image_w=int(values["image_w"]),
        keypoints=keypoints,
        heatmaps=heatmaps,
        ground_truth=typed_ground_truth,
        visible=visible,
        normalization=float(values["normalization"]),
        pck_alpha=float(values["pck_alpha"]),
    )


def decode_keypoints(artifact: PoseArtifact) -> list[Keypoint]:
    scale_x = artifact.image_w / artifact.heatmap_w
    scale_y = artifact.image_h / artifact.heatmap_h
    output: list[Keypoint] = []

    for heatmap in artifact.heatmaps:
        index = max(range(len(heatmap)), key=heatmap.__getitem__)
        y = index // artifact.heatmap_w
        x = index % artifact.heatmap_w
        output.append(
            Keypoint(
                x=(x + 0.5) * scale_x - 0.5,
                y=(y + 0.5) * scale_y - 0.5,
                confidence=heatmap[index],
            )
        )
    return output


def pck(
    artifact: PoseArtifact,
    predicted: list[Keypoint],
) -> float:
    if len(predicted) != artifact.keypoints:
        raise ValueError("predicted keypoint count mismatch")

    threshold = artifact.pck_alpha * artifact.normalization
    correct = 0
    valid = 0

    for prediction, truth, visible in zip(
        predicted,
        artifact.ground_truth,
        artifact.visible,
        strict=True,
    ):
        if not visible:
            continue
        valid += 1
        distance = math.hypot(
            prediction.x - truth[0],
            prediction.y - truth[1],
        )
        if distance <= threshold:
            correct += 1

    if valid == 0:
        raise ValueError("no visible keypoints")
    return correct / valid

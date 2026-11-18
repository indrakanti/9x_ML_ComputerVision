"""Promptable segmentation selection logic for Module 24."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class PromptableArtifact:
    height: int
    width: int
    threshold: float
    positive: tuple[Point, ...]
    negative: tuple[Point, ...]
    scores: tuple[float, ...]
    masks: tuple[tuple[float, ...], ...]


def _points(text: str) -> tuple[Point, ...]:
    if not text:
        return ()
    points: list[Point] = []
    for item in text.split(";"):
        x_text, y_text = item.split(",")
        points.append(Point(int(x_text), int(y_text)))
    return tuple(points)


def load_artifact(path: Path) -> PromptableArtifact:
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

    height = int(values["height"])
    width = int(values["width"])
    pixels = height * width
    scores = tuple(float(value) for value in values["scores"].split(","))
    masks = tuple(
        tuple(float(value) for value in row.split(","))
        for row in values["masks"].split(";")
    )
    if len(scores) != len(masks):
        raise ValueError("scores and masks must have equal length")
    if any(len(mask) != pixels for mask in masks):
        raise ValueError("mask pixel count mismatch")

    artifact = PromptableArtifact(
        height=height,
        width=width,
        threshold=float(values["threshold"]),
        positive=_points(values["positive"]),
        negative=_points(values["negative"]),
        scores=scores,
        masks=masks,
    )

    for point in artifact.positive + artifact.negative:
        if not (0 <= point.x < width and 0 <= point.y < height):
            raise ValueError("prompt point out of bounds")

    return artifact


def threshold_mask(
    mask: tuple[float, ...],
    threshold: float,
) -> list[int]:
    return [1 if value >= threshold else 0 for value in mask]


def prompt_consistent(
    artifact: PromptableArtifact,
    binary_mask: list[int],
) -> bool:
    def value_at(point: Point) -> int:
        return binary_mask[point.y * artifact.width + point.x]

    return all(value_at(point) == 1 for point in artifact.positive) and all(
        value_at(point) == 0 for point in artifact.negative
    )


def select_mask(artifact: PromptableArtifact) -> tuple[int, list[int]]:
    order = sorted(
        range(len(artifact.scores)),
        key=lambda index: (-artifact.scores[index], index),
    )
    for index in order:
        binary = threshold_mask(
            artifact.masks[index],
            artifact.threshold,
        )
        if prompt_consistent(artifact, binary):
            return index, binary
    raise ValueError("no candidate mask satisfies prompts")

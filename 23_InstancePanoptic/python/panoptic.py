"""Instance and panoptic composition for Module 23."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Instance:
    class_id: int
    instance_id: int
    score: float
    mask: tuple[float, ...]


@dataclass(frozen=True)
class PanopticArtifact:
    height: int
    width: int
    semantic: tuple[int, ...]
    instances: tuple[Instance, ...]
    threshold: float


def load_artifact(path: Path) -> PanopticArtifact:
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
    semantic = tuple(int(value) for value in values["semantic"].split(","))
    classes = tuple(
        int(value) for value in values["instance_classes"].split(",")
    )
    ids = tuple(int(value) for value in values["instance_ids"].split(","))
    scores = tuple(float(value) for value in values["scores"].split(","))
    masks = tuple(
        tuple(float(value) for value in row.split(","))
        for row in values["masks"].split(";")
    )

    if len(semantic) != pixels:
        raise ValueError("semantic pixel count mismatch")
    if not (len(classes) == len(ids) == len(scores) == len(masks)):
        raise ValueError("instance arrays must have equal length")
    if any(len(mask) != pixels for mask in masks):
        raise ValueError("mask pixel count mismatch")

    instances = tuple(
        Instance(
            class_id=class_id,
            instance_id=instance_id,
            score=score,
            mask=mask,
        )
        for class_id, instance_id, score, mask in zip(
            classes,
            ids,
            scores,
            masks,
            strict=True,
        )
    )

    return PanopticArtifact(
        height=height,
        width=width,
        semantic=semantic,
        instances=instances,
        threshold=float(values["threshold"]),
    )


def encode_segment(class_id: int, instance_id: int) -> int:
    if class_id < 0 or instance_id < 0:
        raise ValueError("segment IDs must be non-negative")
    return class_id * 1000 + instance_id


def compose_panoptic(artifact: PanopticArtifact) -> list[int]:
    output = [
        encode_segment(class_id, 0)
        for class_id in artifact.semantic
    ]
    claimed = [False] * len(output)

    ordered = sorted(
        artifact.instances,
        key=lambda instance: -instance.score,
    )

    for instance in ordered:
        segment_id = encode_segment(
            instance.class_id,
            instance.instance_id,
        )
        for pixel, probability in enumerate(instance.mask):
            if probability >= artifact.threshold and not claimed[pixel]:
                output[pixel] = segment_id
                claimed[pixel] = True

    return output


def panoptic_quality(
    matched_ious: list[float],
    false_positives: int,
    false_negatives: int,
) -> float:
    if false_positives < 0 or false_negatives < 0:
        raise ValueError("FP/FN must be non-negative")

    denominator = (
        len(matched_ious)
        + 0.5 * false_positives
        + 0.5 * false_negatives
    )
    if denominator == 0.0:
        raise ValueError("PQ denominator is zero")
    return sum(matched_ious) / denominator

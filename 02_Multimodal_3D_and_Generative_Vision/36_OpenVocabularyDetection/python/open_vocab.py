#!/usr/bin/env python3
"""Module 36: open-vocabulary region/text scoring and class-aware NMS."""
from __future__ import annotations

import argparse
import math


def cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in a + b):
        raise ValueError("invalid embedding dimensions or values")
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(x * x for x in b))
    if an <= 1e-12 or bn <= 1e-12:
        raise ValueError("zero norm")
    return sum(x * y for x, y in zip(a, b, strict=True)) / (an * bn)


def iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4 or not all(math.isfinite(x) for x in a + b):
        raise ValueError("invalid box")
    if a[2] <= a[0] or a[3] <= a[1] or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("degenerate box")
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def detect(
    boxes: list[list[float]],
    regions: list[list[float]],
    labels: list[str],
    text_embeddings: list[list[float]],
    threshold: float = 0.5,
    nms_iou: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Return (box index, label index, cosine score) in deterministic score order."""
    if not boxes or len(boxes) != len(regions) or not labels or len(labels) != len(text_embeddings):
        raise ValueError("boxes/regions or labels/text must be aligned and nonempty")
    if not all(math.isfinite(x) and -1.0 <= x <= 1.0 for x in [threshold]):
        raise ValueError("invalid score threshold")
    if not math.isfinite(nms_iou) or not 0.0 <= nms_iou <= 1.0:
        raise ValueError("invalid NMS threshold")
    for box in boxes:
        iou(box, box)
    if any(not label for label in labels):
        raise ValueError("empty label")
    candidates: list[tuple[int, int, float]] = []
    for index, region in enumerate(regions):
        scores = [cosine(region, text) for text in text_embeddings]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] >= threshold:
            candidates.append((index, best, scores[best]))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    kept: list[tuple[int, int, float]] = []
    for candidate in candidates:
        if all(
            candidate[1] != prior[1] or iou(boxes[candidate[0]], boxes[prior[0]]) <= nms_iou
            for prior in kept
        ):
            kept.append(candidate)
    return kept


def synthetic() -> list[tuple[int, int, float]]:
    return detect(
        [[0, 0, 2, 2], [0.2, 0.2, 2.2, 2.2], [4, 4, 6, 6]],
        [[1, 0], [0.8, 0.2], [0, 1]],
        ["car", "person"],
        [[1, 0], [0, 1]],
    )


def real_example(image_path: str, labels: list[str], checkpoint: str) -> None:
    """Optional pretrained zero-shot object detector; downloads model weights."""
    from PIL import Image
    from transformers import pipeline

    detector = pipeline("zero-shot-object-detection", model=checkpoint, device=-1)
    with Image.open(image_path) as image:
        result = detector(image.convert("RGB"), candidate_labels=labels)
    for item in result:
        print(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--labels", nargs="+", default=["car", "person", "bicycle"])
    parser.add_argument("--checkpoint", default="google/owlvit-base-patch32")
    args = parser.parse_args()
    if args.synthetic:
        for index, label, score in synthetic():
            print(f"{index} {label} {score:.9f}")
    elif args.image:
        real_example(args.image, args.labels, args.checkpoint)
    else:
        parser.error("provide --synthetic or --image")


if __name__ == "__main__":
    main()

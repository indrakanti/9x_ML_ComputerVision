"""Walkthrough for Module 18 detection fundamentals."""

from __future__ import annotations

from pathlib import Path

from detection_fundamentals import (
    average_precision,
    class_aware_nms,
    iou,
    load_artifact,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "detections.txt")

    print("Pair IoU:", iou(artifact.iou_a, artifact.iou_b))
    print(
        "NMS keep indices:",
        class_aware_nms(
            artifact.boxes,
            artifact.scores,
            artifact.classes,
            artifact.nms_iou,
        ),
    )
    print(
        "Toy AP:",
        average_precision(
            [0.9, 0.8, 0.2],
            [True, True, False],
            2,
        ),
    )


if __name__ == "__main__":
    main()

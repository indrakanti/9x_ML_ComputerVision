"""Walkthrough for Module 22."""

from __future__ import annotations

from pathlib import Path

from semantic_segmentation import (
    class_iou,
    confusion_matrix,
    load_artifact,
    mean_iou,
    predict_labels,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "segmentation.txt")
    predictions = predict_labels(artifact)
    matrix = confusion_matrix(artifact, predictions)

    print("Predictions:", predictions)
    print("Confusion matrix:", matrix)
    print(
        "Class IoU:",
        [class_iou(matrix, class_id) for class_id in range(artifact.classes)],
    )
    print("mIoU:", mean_iou(matrix))


if __name__ == "__main__":
    main()

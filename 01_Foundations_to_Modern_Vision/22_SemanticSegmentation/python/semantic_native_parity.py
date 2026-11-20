"""Module 22 Python ↔ C++ segmentation parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from semantic_segmentation import (
    class_iou,
    confusion_matrix,
    load_artifact,
    mean_iou,
    predict_labels,
)


def run_native(executable: Path, artifact: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    predictions = predict_labels(artifact)
    matrix = confusion_matrix(artifact, predictions)
    expected_iou = [
        class_iou(matrix, class_id)
        for class_id in range(artifact.classes)
    ]
    expected_miou = mean_iou(matrix)

    actual = run_native(args.executable, args.artifact)
    if actual["predictions"] != predictions:
        raise AssertionError("prediction mismatch")
    if actual["confusion"] != matrix:
        raise AssertionError("confusion-matrix mismatch")

    for cpp, py in zip(actual["iou"], expected_iou, strict=True):
        if not math.isclose(float(cpp), py, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("IoU mismatch")

    if not math.isclose(
        float(actual["miou"]),
        expected_miou,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("mIoU mismatch")

    print("Module 22 Python ↔ C++ semantic parity: PASS")


if __name__ == "__main__":
    main()

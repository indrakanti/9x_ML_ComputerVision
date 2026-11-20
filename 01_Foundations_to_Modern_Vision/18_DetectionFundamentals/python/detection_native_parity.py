"""Module 18 Python ↔ C++ IoU/NMS parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from detection_fundamentals import class_aware_nms, iou, load_artifact


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
    expected_iou = iou(artifact.iou_a, artifact.iou_b)
    expected_keep = class_aware_nms(
        artifact.boxes,
        artifact.scores,
        artifact.classes,
        artifact.nms_iou,
    )
    actual = run_native(args.executable, args.artifact)

    if not math.isclose(
        float(actual["iou"]),
        expected_iou,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("IoU mismatch")
    if actual["keep"] != expected_keep:
        raise AssertionError(
            f"NMS mismatch: C++={actual['keep']} Python={expected_keep}"
        )

    print("Module 18 Python ↔ C++ detection parity: PASS")


if __name__ == "__main__":
    main()

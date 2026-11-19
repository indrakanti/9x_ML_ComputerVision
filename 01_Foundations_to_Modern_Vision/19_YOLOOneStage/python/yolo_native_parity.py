"""Module 19 Python ↔ C++ YOLO decode parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from yolo_decode import final_detections, load_artifact


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
    expected = final_detections(artifact)
    actual = run_native(args.executable, args.artifact)["detections"]

    if len(actual) != len(expected):
        raise AssertionError("detection count mismatch")

    for index, (cpp, python) in enumerate(zip(actual, expected, strict=True)):
        if int(cpp["class_id"]) != python.class_id:
            raise AssertionError(f"class mismatch at {index}")
        for key, expected_value in (
            ("x1", python.x1),
            ("y1", python.y1),
            ("x2", python.x2),
            ("y2", python.y2),
            ("score", python.score),
        ):
            if not math.isclose(
                float(cpp[key]),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"{key} mismatch at {index}")

    print("Module 19 Python ↔ C++ YOLO parity: PASS")


if __name__ == "__main__":
    main()

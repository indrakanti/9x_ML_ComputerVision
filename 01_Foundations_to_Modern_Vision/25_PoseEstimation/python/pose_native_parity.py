"""Module 25 Python ↔ C++ keypoint parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from pose_estimation import decode_keypoints, load_artifact, pck


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
    expected = decode_keypoints(artifact)
    expected_pck = pck(artifact, expected)
    actual = run_native(args.executable, args.artifact)

    for cpp, python in zip(
        actual["keypoints"],
        expected,
        strict=True,
    ):
        for key, value in (
            ("x", python.x),
            ("y", python.y),
            ("confidence", python.confidence),
        ):
            if not math.isclose(
                float(cpp[key]),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError(f"{key} mismatch")

    if not math.isclose(
        float(actual["pck"]),
        expected_pck,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("PCK mismatch")

    print("Module 25 Python ↔ C++ pose parity: PASS")


if __name__ == "__main__":
    main()

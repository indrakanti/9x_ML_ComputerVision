"""Module 16 Python ↔ C++ parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from hybrid_backbone import example_image, load_artifact, manual_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = load_artifact(args.artifact)
    expected = manual_forward(artifact, example_image())
    completed = subprocess.run(
        [str(args.executable), "--json", str(args.artifact)],
        check=True, capture_output=True, text=True,
    )
    actual = json.loads(completed.stdout)["values"]
    for i, (cpp, py) in enumerate(zip(actual, expected, strict=True)):
        if not math.isclose(float(cpp), py, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"output[{i}] C++={cpp} Python={py}")
    print("Module 16 Python ↔ C++ hybrid parity: PASS")


if __name__ == "__main__":
    main()

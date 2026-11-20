"""Module 15 Python ↔ C++ parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from window_attention import example_tokens, load_artifact, manual_window_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    expected = manual_window_attention(artifact, example_tokens())
    completed = subprocess.run(
        [str(args.executable), "--json", str(args.artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout)["values"]
    if len(actual) != len(expected):
        raise AssertionError("output size mismatch")
    for i, (cpp, py) in enumerate(zip(actual, expected, strict=True)):
        if not math.isclose(float(cpp), py, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"output[{i}] C++={cpp} Python={py}")
    print("Module 15 Python ↔ C++ window-attention parity: PASS")


if __name__ == "__main__":
    main()

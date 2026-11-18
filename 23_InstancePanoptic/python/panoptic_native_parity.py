"""Module 23 Python ↔ C++ panoptic parity."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from panoptic import compose_panoptic, load_artifact


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

    expected = compose_panoptic(load_artifact(args.artifact))
    actual = run_native(args.executable, args.artifact)["panoptic"]

    if actual != expected:
        raise AssertionError(
            f"panoptic mismatch C++={actual} Python={expected}"
        )

    print("Module 23 Python ↔ C++ panoptic parity: PASS")


if __name__ == "__main__":
    main()

"""Module 24 Python ↔ C++ promptable-mask parity."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from promptable_segmentation import load_artifact, select_mask


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
    expected_index, expected_mask = select_mask(artifact)
    actual = run_native(args.executable, args.artifact)

    if int(actual["selected"]) != expected_index:
        raise AssertionError("selected-candidate mismatch")
    if actual["mask"] != expected_mask:
        raise AssertionError("mask mismatch")

    print("Module 24 Python ↔ C++ promptable-mask parity: PASS")


if __name__ == "__main__":
    main()

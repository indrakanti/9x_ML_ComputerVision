"""Verify Module 10 deployment manifest parity with C++17."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from training_robustness import load_deployment_manifest


def run_native(executable: Path, manifest: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), "--json", str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    expected = load_deployment_manifest(args.manifest)
    actual = run_native(args.executable, args.manifest)

    expected_dict = {
        "selected_checkpoint": expected.selected_checkpoint,
        "dropout_inference": expected.dropout_inference,
        "augmentation_inference": expected.augmentation_inference,
        "preprocessing_revision": expected.preprocessing_revision,
        "class_count": expected.class_count,
    }
    if actual != expected_dict:
        raise AssertionError(f"C++={actual} Python={expected_dict}")

    print("Module 10 Python ↔ C++ manifest parity: PASS")


if __name__ == "__main__":
    main()

"""Module 26 Python ↔ C++ embedding parity."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from clip_contrastive import (
    load_artifact,
    similarity_matrix,
    top_text_matches,
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
    expected_matrix = similarity_matrix(artifact)
    expected_matches = top_text_matches(artifact)
    actual = run_native(args.executable, args.artifact)

    for row_cpp, row_py in zip(
        actual["similarity"],
        expected_matrix,
        strict=True,
    ):
        for cpp, py in zip(row_cpp, row_py, strict=True):
            if not math.isclose(
                float(cpp),
                py,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError("similarity mismatch")

    if actual["top_text"] != expected_matches:
        raise AssertionError("retrieval ranking mismatch")

    print("Module 26 Python ↔ C++ CLIP similarity parity: PASS")


if __name__ == "__main__":
    main()

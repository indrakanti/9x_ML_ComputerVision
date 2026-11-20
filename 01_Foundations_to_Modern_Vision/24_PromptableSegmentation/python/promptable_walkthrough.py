"""Walkthrough for Module 24."""

from __future__ import annotations

from pathlib import Path

from promptable_segmentation import load_artifact, select_mask


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "promptable.txt")
    index, mask = select_mask(artifact)

    print("Selected candidate:", index)
    for row in range(artifact.height):
        start = row * artifact.width
        print(mask[start : start + artifact.width])


if __name__ == "__main__":
    main()

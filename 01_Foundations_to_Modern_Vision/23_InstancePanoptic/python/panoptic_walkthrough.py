"""Walkthrough for Module 23."""

from __future__ import annotations

from pathlib import Path

from panoptic import compose_panoptic, load_artifact, panoptic_quality


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "panoptic.txt")

    print("Panoptic map:", compose_panoptic(artifact))
    print(
        "Toy PQ:",
        panoptic_quality(
            matched_ious=[0.8, 0.7],
            false_positives=1,
            false_negatives=0,
        ),
    )


if __name__ == "__main__":
    main()

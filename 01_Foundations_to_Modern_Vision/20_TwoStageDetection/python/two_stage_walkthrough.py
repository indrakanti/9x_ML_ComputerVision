"""Walkthrough for Module 20 two-stage proposals."""

from __future__ import annotations

from pathlib import Path

from two_stage import load_artifact, proposals


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "proposals.txt")

    print("Final proposals:")
    for box, score in proposals(artifact):
        print(f"score={score:.3f} box={box}")


if __name__ == "__main__":
    main()

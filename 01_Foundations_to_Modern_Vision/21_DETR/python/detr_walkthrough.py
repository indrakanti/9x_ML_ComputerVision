"""Walkthrough for Module 21."""

from __future__ import annotations

from pathlib import Path

from detr_matching import cost_matrix, load_artifact, optimal_assignment


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "matching.txt")
    matrix = cost_matrix(artifact)

    print("Cost matrix:")
    for row in matrix:
        print(row)
    print("Optimal assignment:", optimal_assignment(matrix))


if __name__ == "__main__":
    main()

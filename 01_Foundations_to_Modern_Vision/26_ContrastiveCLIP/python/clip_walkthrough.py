"""Walkthrough for Module 26."""

from __future__ import annotations

from pathlib import Path

from clip_contrastive import (
    load_artifact,
    similarity_matrix,
    symmetric_contrastive_loss,
    top_text_matches,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "embeddings.txt")

    print("Cosine similarity matrix:")
    for row in similarity_matrix(artifact):
        print(row)
    print("Top text match per image:", top_text_matches(artifact))
    print("Symmetric contrastive loss:", symmetric_contrastive_loss(artifact))


if __name__ == "__main__":
    main()

"""Walkthrough for Module 27."""

from __future__ import annotations

from pathlib import Path

from masked_image_modeling import (
    load_artifact,
    masked_mse,
    patchify,
    reconstruct_masked,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "mim.txt")

    print("Patches:")
    for index, patch in enumerate(patchify(artifact)):
        print(index, patch)

    print("Masked indices:", artifact.masked_indices)
    print("Reconstructions:", reconstruct_masked(artifact))
    print("Masked MSE:", masked_mse(artifact))


if __name__ == "__main__":
    main()

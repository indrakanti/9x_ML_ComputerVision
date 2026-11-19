"""Masked image modeling reference for Module 27."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class MIMArtifact:
    image_h: int
    image_w: int
    patch: int
    masked_indices: tuple[int, ...]
    decoder_weight: tuple[float, ...]
    decoder_bias: tuple[float, ...]
    image: tuple[float, ...]


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(","))


def load_artifact(path: Path) -> MIMArtifact:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"invalid artifact line: {raw}")
        values[key] = value

    if int(values["version"]) != 1:
        raise ValueError("unsupported artifact version")

    image_h = int(values["image_h"])
    image_w = int(values["image_w"])
    patch = int(values["patch"])
    masked_indices = tuple(
        int(value) for value in values["masked_indices"].split(",")
    )
    image = _floats(values["image"])
    patch_values = patch * patch
    token_count = (image_h // patch) * (image_w // patch)

    if image_h % patch != 0 or image_w % patch != 0:
        raise ValueError("image dimensions must be divisible by patch size")
    if len(image) != image_h * image_w:
        raise ValueError("image pixel count mismatch")
    if any(index < 0 or index >= token_count for index in masked_indices):
        raise ValueError("masked index out of range")

    decoder_weight = _floats(values["decoder_weight"])
    decoder_bias = _floats(values["decoder_bias"])
    if len(decoder_weight) != patch_values * patch_values:
        raise ValueError("decoder weight count mismatch")
    if len(decoder_bias) != patch_values:
        raise ValueError("decoder bias count mismatch")

    return MIMArtifact(
        image_h=image_h,
        image_w=image_w,
        patch=patch,
        masked_indices=masked_indices,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        image=image,
    )


def patchify(artifact: MIMArtifact) -> list[list[float]]:
    patches: list[list[float]] = []
    for top in range(0, artifact.image_h, artifact.patch):
        for left in range(0, artifact.image_w, artifact.patch):
            patch: list[float] = []
            for y in range(artifact.patch):
                for x in range(artifact.patch):
                    patch.append(
                        artifact.image[
                            (top + y) * artifact.image_w + left + x
                        ]
                    )
            patches.append(patch)
    return patches


def visible_mean(
    patches: list[list[float]],
    masked_indices: tuple[int, ...],
) -> list[float]:
    masked = set(masked_indices)
    visible = [
        patch
        for index, patch in enumerate(patches)
        if index not in masked
    ]
    if not visible:
        raise ValueError("at least one visible patch is required")

    dimensions = len(visible[0])
    return [
        sum(patch[dim] for patch in visible) / len(visible)
        for dim in range(dimensions)
    ]


def reconstruct_masked(
    artifact: MIMArtifact,
) -> list[list[float]]:
    patches = patchify(artifact)
    representation = visible_mean(patches, artifact.masked_indices)
    dimensions = len(representation)

    prediction = [
        artifact.decoder_bias[out]
        + sum(
            artifact.decoder_weight[out * dimensions + index]
            * representation[index]
            for index in range(dimensions)
        )
        for out in range(dimensions)
    ]
    return [prediction.copy() for _ in artifact.masked_indices]


def masked_mse(artifact: MIMArtifact) -> float:
    patches = patchify(artifact)
    reconstructed = reconstruct_masked(artifact)
    squared_error = 0.0
    count = 0

    for prediction, patch_index in zip(
        reconstructed,
        artifact.masked_indices,
        strict=True,
    ):
        target = patches[patch_index]
        for actual, expected in zip(prediction, target, strict=True):
            difference = actual - expected
            squared_error += difference * difference
            count += 1
    return squared_error / count


def pytorch_reconstruction_and_loss(
    artifact: MIMArtifact,
) -> tuple[Tensor, Tensor]:
    patches = torch.tensor(
        patchify(artifact),
        dtype=torch.float64,
    )
    masked = set(artifact.masked_indices)
    visible_indices = [
        index
        for index in range(patches.shape[0])
        if index not in masked
    ]
    representation = patches[visible_indices].mean(dim=0)
    dimensions = representation.shape[0]
    weight = torch.tensor(
        artifact.decoder_weight,
        dtype=torch.float64,
    ).reshape(dimensions, dimensions)
    bias = torch.tensor(artifact.decoder_bias, dtype=torch.float64)
    prediction = representation @ weight.T + bias
    predictions = prediction.repeat(len(artifact.masked_indices), 1)
    targets = patches[list(artifact.masked_indices)]
    loss = torch.mean((predictions - targets) ** 2)
    return predictions, loss

"""Tiny Vision Transformer attention reference for Module 14."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class ViTArtifact:
    image_h: int
    image_w: int
    patch: int
    embed: int
    patch_weight: tuple[float, ...]
    patch_bias: tuple[float, ...]
    position: tuple[float, ...]
    wq: tuple[float, ...]
    wk: tuple[float, ...]
    wv: tuple[float, ...]


def _vector(values: dict[str, str], key: str) -> tuple[float, ...]:
    return tuple(float(v) for v in values[key].split(","))


def load_artifact(path: Path) -> ViTArtifact:
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

    artifact = ViTArtifact(
        image_h=int(values["image_h"]),
        image_w=int(values["image_w"]),
        patch=int(values["patch"]),
        embed=int(values["embed"]),
        patch_weight=_vector(values, "patch_weight"),
        patch_bias=_vector(values, "patch_bias"),
        position=_vector(values, "position"),
        wq=_vector(values, "wq"),
        wk=_vector(values, "wk"),
        wv=_vector(values, "wv"),
    )
    tokens = (artifact.image_h // artifact.patch) * (
        artifact.image_w // artifact.patch
    )
    patch_values = artifact.patch * artifact.patch
    if len(artifact.patch_weight) != artifact.embed * patch_values:
        raise ValueError("patch projection weight count mismatch")
    if len(artifact.patch_bias) != artifact.embed:
        raise ValueError("patch bias count mismatch")
    if len(artifact.position) != tokens * artifact.embed:
        raise ValueError("position count mismatch")
    matrix_size = artifact.embed * artifact.embed
    matrices = (artifact.wq, artifact.wk, artifact.wv)
    if any(len(matrix) != matrix_size for matrix in matrices):
        raise ValueError("attention matrix size mismatch")
    return artifact


def patchify(artifact: ViTArtifact, image: list[float]) -> list[list[float]]:
    if len(image) != artifact.image_h * artifact.image_w:
        raise ValueError("image size mismatch")
    if (
        artifact.image_h % artifact.patch != 0
        or artifact.image_w % artifact.patch != 0
    ):
        raise ValueError("image dimensions must be divisible by patch size")

    patches: list[list[float]] = []
    for top in range(0, artifact.image_h, artifact.patch):
        for left in range(0, artifact.image_w, artifact.patch):
            patch: list[float] = []
            for y in range(artifact.patch):
                for x in range(artifact.patch):
                    patch.append(
                        image[(top + y) * artifact.image_w + left + x]
                    )
            patches.append(patch)
    return patches


def _linear(
    vector: list[float],
    weights: tuple[float, ...],
    bias: tuple[float, ...],
    outputs: int,
) -> list[float]:
    inputs = len(vector)
    return [
        bias[out]
        + sum(
            weights[out * inputs + index] * vector[index]
            for index in range(inputs)
        )
        for out in range(outputs)
    ]


def _matmul_tokens(
    tokens: list[list[float]],
    weights: tuple[float, ...],
    dim: int,
) -> list[list[float]]:
    zero_bias = (0.0,) * dim
    return [_linear(token, weights, zero_bias, dim) for token in tokens]


def manual_attention(
    artifact: ViTArtifact,
    image: list[float],
) -> list[float]:
    patches = patchify(artifact, image)
    patch_values = artifact.patch * artifact.patch
    tokens: list[list[float]] = []

    for token_index, patch in enumerate(patches):
        embedded = [
            artifact.patch_bias[out]
            + sum(
                artifact.patch_weight[out * patch_values + index]
                * patch[index]
                for index in range(patch_values)
            )
            for out in range(artifact.embed)
        ]
        for dim in range(artifact.embed):
            embedded[dim] += artifact.position[
                token_index * artifact.embed + dim
            ]
        tokens.append(embedded)

    q = _matmul_tokens(tokens, artifact.wq, artifact.embed)
    k = _matmul_tokens(tokens, artifact.wk, artifact.embed)
    v = _matmul_tokens(tokens, artifact.wv, artifact.embed)

    scale = math.sqrt(float(artifact.embed))
    output: list[float] = []
    for query in q:
        scores = [
            sum(
                query[dim] * key[dim]
                for dim in range(artifact.embed)
            )
            / scale
            for key in k
        ]
        maximum = max(scores)
        exps = [math.exp(score - maximum) for score in scores]
        total = sum(exps)
        probabilities = [value / total for value in exps]

        for dim in range(artifact.embed):
            output.append(
                sum(
                    probabilities[token] * v[token][dim]
                    for token in range(len(v))
                )
            )
    return output


def pytorch_attention(artifact: ViTArtifact, image: list[float]) -> Tensor:
    patches = torch.tensor(
        patchify(artifact, image),
        dtype=torch.float64,
    )
    patch_values = artifact.patch * artifact.patch
    patch_weight = torch.tensor(
        artifact.patch_weight,
        dtype=torch.float64,
    ).reshape(artifact.embed, patch_values)
    patch_bias = torch.tensor(artifact.patch_bias, dtype=torch.float64)
    position = torch.tensor(
        artifact.position,
        dtype=torch.float64,
    ).reshape(-1, artifact.embed)

    tokens = patches @ patch_weight.T + patch_bias + position
    wq = torch.tensor(artifact.wq, dtype=torch.float64).reshape(
        artifact.embed,
        artifact.embed,
    )
    wk = torch.tensor(artifact.wk, dtype=torch.float64).reshape(
        artifact.embed,
        artifact.embed,
    )
    wv = torch.tensor(artifact.wv, dtype=torch.float64).reshape(
        artifact.embed,
        artifact.embed,
    )

    q = tokens @ wq.T
    k = tokens @ wk.T
    v = tokens @ wv.T
    scores = q @ k.T / math.sqrt(float(artifact.embed))
    attention = torch.softmax(scores, dim=-1)
    return attention @ v


def example_image() -> list[float]:
    return [
        1, 2, 3, 4,
        5, 6, 7, 8,
        8, 7, 6, 5,
        4, 3, 2, 1,
    ]

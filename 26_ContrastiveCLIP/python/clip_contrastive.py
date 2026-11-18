"""CLIP-style contrastive similarity for Module 26."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClipArtifact:
    dimension: int
    temperature: float
    image_embeddings: tuple[tuple[float, ...], ...]
    text_embeddings: tuple[tuple[float, ...], ...]


def _rows(text: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row.split(","))
        for row in text.split(";")
    )


def load_artifact(path: Path) -> ClipArtifact:
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

    dimension = int(values["dimension"])
    image_embeddings = _rows(values["image_embeddings"])
    text_embeddings = _rows(values["text_embeddings"])

    if not image_embeddings or not text_embeddings:
        raise ValueError("embeddings must be non-empty")
    if any(len(row) != dimension for row in image_embeddings):
        raise ValueError("image embedding dimension mismatch")
    if any(len(row) != dimension for row in text_embeddings):
        raise ValueError("text embedding dimension mismatch")

    temperature = float(values["temperature"])
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    return ClipArtifact(
        dimension=dimension,
        temperature=temperature,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
    )


def normalize(vector: tuple[float, ...]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("cannot normalize zero vector")
    return [value / norm for value in vector]


def similarity_matrix(artifact: ClipArtifact) -> list[list[float]]:
    images = [normalize(row) for row in artifact.image_embeddings]
    texts = [normalize(row) for row in artifact.text_embeddings]

    return [
        [
            sum(
                image[dim] * text[dim]
                for dim in range(artifact.dimension)
            )
            for text in texts
        ]
        for image in images
    ]


def scaled_logits(artifact: ClipArtifact) -> list[list[float]]:
    return [
        [value / artifact.temperature for value in row]
        for row in similarity_matrix(artifact)
    ]


def top_text_matches(artifact: ClipArtifact) -> list[int]:
    matrix = similarity_matrix(artifact)
    return [
        max(range(len(row)), key=row.__getitem__)
        for row in matrix
    ]


def _cross_entropy_rows(
    logits: list[list[float]],
    labels: list[int],
) -> float:
    losses: list[float] = []
    for row, label in zip(logits, labels, strict=True):
        maximum = max(row)
        exp_values = [math.exp(value - maximum) for value in row]
        log_denominator = maximum + math.log(sum(exp_values))
        losses.append(-(row[label] - log_denominator))
    return sum(losses) / len(losses)


def symmetric_contrastive_loss(artifact: ClipArtifact) -> float:
    logits = scaled_logits(artifact)
    if len(logits) != len(logits[0]):
        raise ValueError("reference loss expects equal image/text batch sizes")
    labels = list(range(len(logits)))
    image_to_text = _cross_entropy_rows(logits, labels)
    transposed = [
        [logits[row][column] for row in range(len(logits))]
        for column in range(len(logits[0]))
    ]
    text_to_image = _cross_entropy_rows(transposed, labels)
    return 0.5 * (image_to_text + text_to_image)


def pytorch_contrastive_loss(artifact: ClipArtifact) -> Tensor:
    images = torch.tensor(
        artifact.image_embeddings,
        dtype=torch.float64,
    )
    texts = torch.tensor(
        artifact.text_embeddings,
        dtype=torch.float64,
    )
    images = torch.nn.functional.normalize(images, dim=1)
    texts = torch.nn.functional.normalize(texts, dim=1)
    logits = images @ texts.T / artifact.temperature
    labels = torch.arange(logits.shape[0])
    image_to_text = torch.nn.functional.cross_entropy(logits, labels)
    text_to_image = torch.nn.functional.cross_entropy(logits.T, labels)
    return 0.5 * (image_to_text + text_to_image)

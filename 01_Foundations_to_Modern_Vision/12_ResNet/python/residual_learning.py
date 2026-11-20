"""Residual learning reference for Module 12."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResidualArtifact:
    channels: int
    height: int
    width: int
    weights: tuple[float, ...]
    bias: tuple[float, ...]


class TinyResidualBlock(nn.Module):
    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(self.conv(inputs) + inputs)


def load_artifact(path: Path) -> ResidualArtifact:
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

    channels = int(values["channels"])
    artifact = ResidualArtifact(
        channels=channels,
        height=int(values["height"]),
        width=int(values["width"]),
        weights=tuple(float(v) for v in values["weights"].split(",")),
        bias=tuple(float(v) for v in values["bias"].split(",")),
    )
    if len(artifact.weights) != channels * channels:
        raise ValueError("weight count mismatch")
    if len(artifact.bias) != channels:
        raise ValueError("bias count mismatch")
    return artifact


def apply_artifact(model: TinyResidualBlock, artifact: ResidualArtifact) -> None:
    if model.conv.bias is None:
        raise ValueError("residual convolution must be bias-enabled")
    weight = torch.tensor(
        artifact.weights,
        dtype=torch.float64,
    ).reshape(artifact.channels, artifact.channels, 1, 1)
    bias = torch.tensor(artifact.bias, dtype=torch.float64)
    model.double()
    with torch.no_grad():
        model.conv.weight.copy_(weight)
        model.conv.bias.copy_(bias)


def manual_residual_forward(
    artifact: ResidualArtifact,
    inputs: list[float],
) -> list[float]:
    expected = artifact.channels * artifact.height * artifact.width
    if len(inputs) != expected:
        raise ValueError("input size mismatch")

    outputs = [0.0] * expected
    spatial = artifact.height * artifact.width
    for y in range(artifact.height):
        for x in range(artifact.width):
            pixel = y * artifact.width + x
            for out_channel in range(artifact.channels):
                value = artifact.bias[out_channel]
                for in_channel in range(artifact.channels):
                    weight_index = out_channel * artifact.channels + in_channel
                    input_index = in_channel * spatial + pixel
                    value += artifact.weights[weight_index] * inputs[input_index]
                skip_index = out_channel * spatial + pixel
                outputs[skip_index] = max(0.0, value + inputs[skip_index])
    return outputs


def example_input() -> list[float]:
    return [
        1.0,
        -1.0,
        0.5,
        2.0,
        0.25,
        1.5,
        -0.5,
        1.0,
    ]

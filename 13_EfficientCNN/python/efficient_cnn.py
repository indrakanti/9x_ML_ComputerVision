"""Depthwise separable convolution reference for Module 13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class EfficientArtifact:
    channels: int
    height: int
    width: int
    kernel: int
    outputs: int
    depthwise_weights: tuple[float, ...]
    depthwise_bias: tuple[float, ...]
    pointwise_weights: tuple[float, ...]
    pointwise_bias: tuple[float, ...]


def _vector(values: dict[str, str], key: str) -> tuple[float, ...]:
    return tuple(float(v) for v in values[key].split(","))


def load_artifact(path: Path) -> EfficientArtifact:
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

    artifact = EfficientArtifact(
        channels=int(values["channels"]),
        height=int(values["height"]),
        width=int(values["width"]),
        kernel=int(values["kernel"]),
        outputs=int(values["outputs"]),
        depthwise_weights=_vector(values, "depthwise_weights"),
        depthwise_bias=_vector(values, "depthwise_bias"),
        pointwise_weights=_vector(values, "pointwise_weights"),
        pointwise_bias=_vector(values, "pointwise_bias"),
    )
    if len(artifact.depthwise_weights) != (
        artifact.channels * artifact.kernel * artifact.kernel
    ):
        raise ValueError("depthwise weight count mismatch")
    if len(artifact.depthwise_bias) != artifact.channels:
        raise ValueError("depthwise bias count mismatch")
    if len(artifact.pointwise_weights) != artifact.outputs * artifact.channels:
        raise ValueError("pointwise weight count mismatch")
    if len(artifact.pointwise_bias) != artifact.outputs:
        raise ValueError("pointwise bias count mismatch")
    return artifact


def manual_forward(
    artifact: EfficientArtifact,
    inputs: list[float],
) -> list[float]:
    spatial = artifact.height * artifact.width
    if len(inputs) != artifact.channels * spatial:
        raise ValueError("input size mismatch")

    out_h = artifact.height - artifact.kernel + 1
    out_w = artifact.width - artifact.kernel + 1
    depth_spatial = out_h * out_w
    depth = [0.0] * (artifact.channels * depth_spatial)

    for channel in range(artifact.channels):
        for y in range(out_h):
            for x in range(out_w):
                value = artifact.depthwise_bias[channel]
                for ky in range(artifact.kernel):
                    for kx in range(artifact.kernel):
                        input_index = (
                            channel * spatial
                            + (y + ky) * artifact.width
                            + x
                            + kx
                        )
                        weight_index = (
                            channel * artifact.kernel * artifact.kernel
                            + ky * artifact.kernel
                            + kx
                        )
                        value += (
                            inputs[input_index]
                            * artifact.depthwise_weights[weight_index]
                        )
                depth[channel * depth_spatial + y * out_w + x] = value

    output = [0.0] * (artifact.outputs * depth_spatial)
    for out_channel in range(artifact.outputs):
        for pixel in range(depth_spatial):
            value = artifact.pointwise_bias[out_channel]
            for in_channel in range(artifact.channels):
                weight_index = out_channel * artifact.channels + in_channel
                value += (
                    artifact.pointwise_weights[weight_index]
                    * depth[in_channel * depth_spatial + pixel]
                )
            output[out_channel * depth_spatial + pixel] = value
    return output


def pytorch_forward(
    artifact: EfficientArtifact,
    inputs: list[float],
) -> Tensor:
    image = torch.tensor(inputs, dtype=torch.float64).reshape(
        1,
        artifact.channels,
        artifact.height,
        artifact.width,
    )
    depthwise = torch.tensor(
        artifact.depthwise_weights,
        dtype=torch.float64,
    ).reshape(
        artifact.channels,
        1,
        artifact.kernel,
        artifact.kernel,
    )
    depth_bias = torch.tensor(artifact.depthwise_bias, dtype=torch.float64)
    pointwise = torch.tensor(
        artifact.pointwise_weights,
        dtype=torch.float64,
    ).reshape(artifact.outputs, artifact.channels, 1, 1)
    point_bias = torch.tensor(artifact.pointwise_bias, dtype=torch.float64)

    depth = torch.nn.functional.conv2d(
        image,
        depthwise,
        depth_bias,
        groups=artifact.channels,
    )
    return torch.nn.functional.conv2d(depth, pointwise, point_bias)


def mac_counts(artifact: EfficientArtifact) -> tuple[int, int]:
    out_h = artifact.height - artifact.kernel + 1
    out_w = artifact.width - artifact.kernel + 1
    pixels = out_h * out_w
    standard = (
        pixels
        * artifact.outputs
        * artifact.channels
        * artifact.kernel
        * artifact.kernel
    )
    separable = (
        pixels
        * artifact.channels
        * artifact.kernel
        * artifact.kernel
        + pixels * artifact.outputs * artifact.channels
    )
    return standard, separable


def example_input() -> list[float]:
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
    ]

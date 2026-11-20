"""First-principles 2-D convolution reference for Module 08."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Conv2DConfig:
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)


def output_shape(
    input_hw: tuple[int, int],
    kernel_hw: tuple[int, int],
    config: Conv2DConfig,
) -> tuple[int, int]:
    height, width = input_hw
    kernel_h, kernel_w = kernel_hw
    stride_h, stride_w = config.stride
    pad_h, pad_w = config.padding

    if min(height, width, kernel_h, kernel_w, stride_h, stride_w) <= 0:
        raise ValueError("input, kernel and stride dimensions must be > 0")
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding must be >= 0")

    numerator_h = height + 2 * pad_h - kernel_h
    numerator_w = width + 2 * pad_w - kernel_w
    if numerator_h < 0 or numerator_w < 0:
        raise ValueError("kernel is larger than padded input")

    return (
        numerator_h // stride_h + 1,
        numerator_w // stride_w + 1,
    )


def conv2d_chw(
    image: FloatArray,
    weights: FloatArray,
    bias: FloatArray,
    config: Conv2DConfig = Conv2DConfig(),
) -> FloatArray:
    if image.ndim != 3:
        raise ValueError("image must have shape [C,H,W]")
    if weights.ndim != 4:
        raise ValueError("weights must have shape [O,C,KH,KW]")
    if bias.ndim != 1:
        raise ValueError("bias must have shape [O]")

    channels, height, width = image.shape
    outputs, weight_channels, kernel_h, kernel_w = weights.shape
    if channels != weight_channels:
        raise ValueError("input-channel mismatch")
    if bias.shape[0] != outputs:
        raise ValueError("bias size must equal output channels")

    out_h, out_w = output_shape(
        (height, width),
        (kernel_h, kernel_w),
        config,
    )
    result = np.zeros((outputs, out_h, out_w), dtype=np.float64)
    stride_h, stride_w = config.stride
    pad_h, pad_w = config.padding

    for out_channel in range(outputs):
        for out_y in range(out_h):
            for out_x in range(out_w):
                accumulator = float(bias[out_channel])
                for in_channel in range(channels):
                    for kernel_y in range(kernel_h):
                        for kernel_x in range(kernel_w):
                            input_y = out_y * stride_h + kernel_y - pad_h
                            input_x = out_x * stride_w + kernel_x - pad_w
                            if 0 <= input_y < height and 0 <= input_x < width:
                                accumulator += float(
                                    image[in_channel, input_y, input_x]
                                ) * float(
                                    weights[
                                        out_channel,
                                        in_channel,
                                        kernel_y,
                                        kernel_x,
                                    ]
                                )
                result[out_channel, out_y, out_x] = accumulator

    return result


def example_tensors() -> tuple[FloatArray, FloatArray, FloatArray, Conv2DConfig]:
    image = np.asarray(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [
            [
                [[1.0, 0.0], [0.0, -1.0]],
                [[0.5, 0.0], [0.0, -0.5]],
            ],
            [
                [[0.0, 1.0], [-1.0, 0.0]],
                [[0.0, -0.25], [0.25, 0.0]],
            ],
        ],
        dtype=np.float64,
    )
    bias = np.asarray([0.25, -0.5], dtype=np.float64)
    return image, weights, bias, Conv2DConfig(stride=(1, 1), padding=(1, 1))

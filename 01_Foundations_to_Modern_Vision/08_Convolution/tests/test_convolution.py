from __future__ import annotations

import numpy as np
import pytest
import torch
from convolution import Conv2DConfig, conv2d_chw, example_tensors, output_shape


def test_output_shape() -> None:
    config = Conv2DConfig(stride=(2, 2), padding=(1, 1))
    assert output_shape((7, 7), (3, 3), config) == (4, 4)


def test_simple_single_channel_example() -> None:
    image = np.asarray(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
        dtype=np.float64,
    )
    weights = np.asarray(
        [[[[1.0, 0.0], [0.0, -1.0]]]],
        dtype=np.float64,
    )
    bias = np.asarray([0.0], dtype=np.float64)

    output = conv2d_chw(image, weights, bias)

    assert output == pytest.approx(
        np.asarray([[[-4.0, -4.0], [-4.0, -4.0]]], dtype=np.float64)
    )


def test_reference_matches_pytorch() -> None:
    image, weights, bias, config = example_tensors()
    expected = conv2d_chw(image, weights, bias, config)
    actual = torch.nn.functional.conv2d(
        torch.from_numpy(image).unsqueeze(0),
        torch.from_numpy(weights),
        torch.from_numpy(bias),
        stride=config.stride,
        padding=config.padding,
    ).squeeze(0)

    assert actual.numpy() == pytest.approx(expected)


def test_invalid_channel_count_rejected() -> None:
    image = np.zeros((2, 3, 3), dtype=np.float64)
    weights = np.zeros((1, 1, 2, 2), dtype=np.float64)
    bias = np.zeros((1,), dtype=np.float64)

    with pytest.raises(ValueError):
        conv2d_chw(image, weights, bias)


def test_kernel_larger_than_padded_input_rejected() -> None:
    with pytest.raises(ValueError):
        output_shape(
            (2, 2),
            (5, 5),
            Conv2DConfig(),
        )

"""Walkthrough for Module 08 convolution."""

from __future__ import annotations

import torch
from convolution import conv2d_chw, example_tensors


def main() -> None:
    image, weights, bias, config = example_tensors()
    reference = conv2d_chw(image, weights, bias, config)

    torch_output = torch.nn.functional.conv2d(
        torch.from_numpy(image).unsqueeze(0),
        torch.from_numpy(weights),
        torch.from_numpy(bias),
        stride=config.stride,
        padding=config.padding,
    ).squeeze(0)

    print("Input shape:", image.shape)
    print("Weight shape:", weights.shape)
    print("Output shape:", reference.shape)
    print("\nReference output")
    print(reference)
    print("\nMax |Python - PyTorch|")
    print(float(torch.max(torch.abs(torch.from_numpy(reference) - torch_output))))


if __name__ == "__main__":
    main()

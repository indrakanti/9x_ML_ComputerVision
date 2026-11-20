"""Tensor metadata helpers owned by Module 01."""

from __future__ import annotations

from typing import Any

import torch


def contract_from_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "strides": list(tensor.stride()),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "requires_grad": tensor.requires_grad,
    }


def hwc_uint8_to_nchw_float(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("expected HWC tensor with rank 3")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError("expected channel-last image tensor")
    if image.dtype != torch.uint8:
        raise TypeError("expected uint8 image tensor")

    chw = image.permute(2, 0, 1).contiguous()
    return chw.to(torch.float32).div(255.0).unsqueeze(0)

"""Detection geometry for Applied Lab 02."""

from __future__ import annotations

import torch
from torch import Tensor


def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(
        (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            x2 - x1,
            y2 - y1,
        ),
        dim=-1,
    )


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def box_iou_aligned(first: Tensor, second: Tensor) -> Tensor:
    left_top = torch.maximum(first[..., :2], second[..., :2])
    right_bottom = torch.minimum(first[..., 2:], second[..., 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

    first_wh = (first[..., 2:] - first[..., :2]).clamp(min=0.0)
    second_wh = (second[..., 2:] - second[..., :2]).clamp(min=0.0)
    first_area = first_wh[..., 0] * first_wh[..., 1]
    second_area = second_wh[..., 0] * second_wh[..., 1]

    union = first_area + second_area - intersection
    return intersection / union.clamp(min=1e-8)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import mask_to_box, prepare_sample, trimap_to_binary  # noqa: E402
from models import TinyDETR, TinyUNet, TinyYOLO  # noqa: E402


def test_trimap_and_box_conversion() -> None:
    trimap = torch.tensor(
        [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 3, 1, 2],
            [2, 2, 2, 2],
        ],
        dtype=torch.uint8,
    )
    binary = trimap_to_binary(trimap)
    box = mask_to_box(binary)

    assert binary.sum().item() == 4.0
    assert torch.allclose(
        box,
        torch.tensor([0.25, 0.25, 0.75, 0.75]),
    )


def test_prepare_sample_returns_normalized_contract() -> None:
    image = Image.fromarray(
        np.full((8, 8, 3), 127, dtype=np.uint8)
    )
    trimap = Image.fromarray(
        np.array(
            [
                [2, 2, 2, 2, 2, 2, 2, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 1, 2],
                [2, 2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=np.uint8,
        )
    )

    sample, target = prepare_sample(image, trimap, image_size=32)

    assert sample.shape == (3, 32, 32)
    assert target.mask.shape == (32, 32)
    assert target.box_xyxy.shape == (4,)


def test_models_have_expected_shapes() -> None:
    sample = torch.randn(2, 3, 128, 128)

    yolo = TinyYOLO()
    assert yolo(sample).shape == (2, 5, 4, 4)

    detr = TinyDETR(num_queries=5)
    classes, boxes = detr(sample)
    assert classes.shape == (2, 5, 2)
    assert boxes.shape == (2, 5, 4)

    segmenter = TinyUNet()
    assert segmenter(sample).shape == (2, 128, 128)

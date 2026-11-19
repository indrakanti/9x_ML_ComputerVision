from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trimap_and_box_conversion() -> None:
    data = load_module("cv9x_lab02_data", "data.py")
    trimap = torch.tensor(
        [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 3, 1, 2],
            [2, 2, 2, 2],
        ],
        dtype=torch.uint8,
    )
    binary = data.trimap_to_binary(trimap)
    box = data.mask_to_box(binary)

    assert binary.sum().item() == 4.0
    assert torch.allclose(
        box,
        torch.tensor([0.25, 0.25, 0.75, 0.75]),
    )


def test_prepare_sample_returns_normalized_contract() -> None:
    data = load_module("cv9x_lab02_data_sample", "data.py")
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

    sample, target = data.prepare_sample(image, trimap, image_size=32)

    assert sample.shape == (3, 32, 32)
    assert target.mask.shape == (32, 32)
    assert target.box_xyxy.shape == (4,)


def test_models_have_expected_shapes() -> None:
    models = load_module("cv9x_lab02_models", "models.py")
    sample = torch.randn(2, 3, 128, 128)

    yolo = models.TinyYOLO()
    assert yolo(sample).shape == (2, 5, 4, 4)

    detr = models.TinyDETR(num_queries=5)
    classes, boxes = detr(sample)
    assert classes.shape == (2, 5, 2)
    assert boxes.shape == (2, 5, 4)

    segmenter = models.TinyUNet()
    assert segmenter(sample).shape == (2, 128, 128)

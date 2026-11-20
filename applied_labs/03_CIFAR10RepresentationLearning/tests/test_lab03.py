from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_models() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cv9x_lab03_models",
        ROOT / "models.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Lab 03 models")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clip_similarity_shape() -> None:
    models = load_models()
    model = models.TinyCLIP(vocabulary_size=32)
    images = torch.randn(2, 3, 32, 32)
    prompts = torch.randint(1, 32, (10, 16))

    logits = model(images, prompts)

    assert logits.shape == (2, 10)


def test_mim_masked_loss_and_representation() -> None:
    models = load_models()
    model = models.TinyMIM()
    images = torch.randn(2, 3, 32, 32)
    mask = torch.zeros(2, model.tokens, dtype=torch.bool)
    mask[:, : model.tokens // 2] = True

    reconstruction, loss = model(images, mask)
    representation = model.representation(images)

    assert reconstruction.shape == (
        2,
        model.tokens,
        model.patch_dim,
    )
    assert representation.shape == (2, 128)
    assert torch.isfinite(loss)


def test_dino_teacher_is_ema_updated() -> None:
    models = load_models()
    student = models.DINOBackbone()
    teacher = models.make_teacher(student)

    with torch.no_grad():
        first_student = next(student.parameters())
        first_student.add_(1.0)

    first_teacher_before = next(teacher.parameters()).detach().clone()
    models.ema_update(teacher, student, momentum=0.9)
    first_teacher_after = next(teacher.parameters()).detach()

    assert not torch.equal(first_teacher_before, first_teacher_after)

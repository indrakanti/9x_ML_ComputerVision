from __future__ import annotations

from pathlib import Path

import pytest
from dino_self_supervised import (
    ema_teacher_params,
    load_artifact,
    pytorch_loss,
    symmetric_cross_view_loss,
)


def artifact_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "dino.txt"


def test_manual_loss_matches_pytorch() -> None:
    artifact = load_artifact(artifact_path())

    assert symmetric_cross_view_loss(artifact) == pytest.approx(
        float(pytorch_loss(artifact).item()),
        rel=1e-12,
        abs=1e-12,
    )


def test_ema_teacher_moves_toward_student() -> None:
    artifact = load_artifact(artifact_path())
    updated = ema_teacher_params(artifact)

    for teacher, student, value in zip(
        artifact.teacher_params,
        artifact.student_params,
        updated,
        strict=True,
    ):
        assert abs(value - student) <= abs(teacher - student)

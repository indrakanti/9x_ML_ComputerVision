"""Walkthrough for Module 28."""

from __future__ import annotations

from pathlib import Path

from dino_self_supervised import (
    ema_teacher_params,
    load_artifact,
    student_distribution,
    symmetric_cross_view_loss,
    teacher_distribution,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = load_artifact(root / "artifacts" / "dino.txt")

    print(
        "Student A:",
        student_distribution(
            artifact.student_view_a,
            artifact.student_temperature,
        ),
    )
    print(
        "Teacher B:",
        teacher_distribution(
            artifact.teacher_view_b,
            artifact.center,
            artifact.teacher_temperature,
        ),
    )
    print("Cross-view loss:", symmetric_cross_view_loss(artifact))
    print("EMA teacher params:", ema_teacher_params(artifact))


if __name__ == "__main__":
    main()

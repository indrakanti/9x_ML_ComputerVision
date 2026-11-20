"""Walkthrough for Module 10."""

from __future__ import annotations

from pathlib import Path

import torch
from training_robustness import (
    AugmentationConfig,
    DeploymentManifest,
    RegularizedTinyCNN,
    augment_batch,
    build_optimizer_and_scheduler,
    load_deployment_manifest,
    write_deployment_manifest,
)


def main() -> None:
    torch.manual_seed(7)
    model = RegularizedTinyCNN()
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    print("Initial LR:", optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    print("LR after one scheduler step:", optimizer.param_groups[0]["lr"])

    images = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    generator = torch.Generator().manual_seed(7)
    augmented = augment_batch(images, AugmentationConfig(), generator)
    print("Augmented batch shape:", tuple(augmented.shape))

    path = Path("deployment_manifest.txt")
    manifest = DeploymentManifest(
        version=1,
        selected_checkpoint="best",
        dropout_inference="disabled",
        augmentation_inference="disabled",
        preprocessing_revision="synthetic-v1",
        class_count=2,
    )
    write_deployment_manifest(path, manifest)
    print(load_deployment_manifest(path))
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

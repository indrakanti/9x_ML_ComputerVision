"""Walkthrough for Module 09 CNN training."""

from __future__ import annotations

from pathlib import Path

import torch
from cnn_training import (
    artifact_inference,
    export_native_artifact,
    load_native_artifact,
    train_model,
    vertical_example,
)


def main() -> None:
    model, history = train_model()
    print("Initial validation:", history[0])
    print("Final validation:", history[-1])

    artifact_path = Path("tiny_cnn_trained.txt")
    export_native_artifact(model, artifact_path)
    artifact = load_native_artifact(artifact_path)
    print("Artifact logits:", artifact_inference(artifact, vertical_example()))

    image = torch.tensor(vertical_example(), dtype=torch.float32).reshape(1, 1, 4, 4)
    model.eval()
    with torch.no_grad():
        print("PyTorch logits:", model(image).squeeze(0).tolist())

    artifact_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

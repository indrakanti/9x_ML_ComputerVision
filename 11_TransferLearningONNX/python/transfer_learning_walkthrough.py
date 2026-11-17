"""Walkthrough for Module 11 transfer learning and ONNX export."""

from __future__ import annotations

from pathlib import Path

import torch
from transfer_learning import (
    TinyTransferCNN,
    apply_backbone_artifact,
    export_onnx,
    load_backbone_artifact,
    make_transfer_dataset,
    onnx_runtime_logits,
    train_classifier_head,
    vertical_example_tensor,
)


def main() -> None:
    module_root = Path(__file__).resolve().parents[1]
    backbone_path = module_root / "artifacts" / "pretrained_backbone.txt"

    torch.manual_seed(7)
    model = TinyTransferCNN()
    apply_backbone_artifact(model, load_backbone_artifact(backbone_path))

    images, labels = make_transfer_dataset()
    losses = train_classifier_head(model, images, labels)
    print("Initial loss:", losses[0])
    print("Final loss:", losses[-1])

    onnx_path = Path("transfer_model.onnx")
    export_onnx(model, onnx_path)

    sample = vertical_example_tensor()
    model.eval()
    with torch.no_grad():
        pytorch_logits = model(sample).numpy()
    ort_logits = onnx_runtime_logits(onnx_path, sample)

    print("PyTorch logits:", pytorch_logits.tolist())
    print("ONNX Runtime logits:", ort_logits.tolist())
    onnx_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

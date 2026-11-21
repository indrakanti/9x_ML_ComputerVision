"""Test true trainable Module 32 encoders without a dataset download."""
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from production import (
    CLASSES,
    DualEncoder,
    SyntheticImages,
    encode_text,
    evaluate,
    load_model,
    prompt_tokens,
)
from torch.utils.data import DataLoader


def test_tokenizer_contract() -> None:
    assert len(encode_text("a dog")) == 32
    assert encode_text("A")[0] == 34
    with pytest.raises(ValueError):
        encode_text("")
    with pytest.raises(ValueError):
        encode_text("é")


def test_trainable_encoder_and_gradients() -> None:
    torch.manual_seed(2)
    model = DualEncoder()
    images = torch.rand(2, 3, 32, 32)
    logits = model(images, prompt_tokens())
    assert logits.shape == (2, len(CLASSES))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.image.features[0].weight.grad is not None
    assert model.text.embedding.weight.grad is not None
    with torch.no_grad():
        lengths = model.image(images).norm(dim=1)
        torch.testing.assert_close(lengths, torch.ones_like(lengths))


def test_synthetic_dataset_deterministic() -> None:
    dataset = SyntheticImages(5)
    a, label = dataset[3]
    b, other = dataset[3]
    assert label == other == 3
    torch.testing.assert_close(a, b)


def test_train_checkpoint_evaluate(tmp_path: Path) -> None:
    script = Path(__file__).with_name("production.py")
    folder = tmp_path / "trained"
    subprocess.run(
        [sys.executable, str(script), "train", "--dataset", "synthetic",
         "--epochs", "1", "--batch-size", "20", "--output", str(folder)],
        check=True, capture_output=True, text=True,
    )
    report = json.loads((folder / "metrics.json").read_text())
    assert report["metrics"]["dataset_samples"] == 40
    assert 0.0 <= report["metrics"]["top1"] <= 1.0
    assert report["metrics"]["chance_top1"] == pytest.approx(0.1)
    model = load_model(folder / "model.pt")
    with torch.inference_mode():
        values = model(torch.rand(1, 3, 32, 32), prompt_tokens())
    assert values.shape == (1, 10)
    assert np.isfinite(values.numpy()).all()


def test_real_evaluation_metrics_structure() -> None:
    torch.manual_seed(4)
    model = DualEncoder().eval()
    metrics = evaluate(model, DataLoader(SyntheticImages(20), batch_size=5), torch.device("cpu"))
    confusion = cast(list[list[int]], metrics["confusion_matrix"])
    assert sum(sum(row) for row in confusion) == 20
    assert len(metrics["label_order"]) == 10
    assert 0.0 <= cast(float, metrics["macro_f1"]) <= 1.0
    assert metrics["dataset_samples"] == 20

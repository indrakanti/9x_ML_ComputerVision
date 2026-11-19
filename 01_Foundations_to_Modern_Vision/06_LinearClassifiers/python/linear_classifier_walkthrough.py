"""Walkthrough for Module 06 linear classifiers."""

from __future__ import annotations

from pathlib import Path

import torch
from linear_classifier import (
    LinearBinaryModel,
    load_model,
    mean_loss,
    predict,
    predict_probability,
    train_logistic_regression,
)


def main() -> None:
    features = [
        [0.10, 0.20, 0.10, 0.20],
        [0.20, 0.10, 0.20, 0.10],
        [0.80, 0.70, 0.90, 0.60],
        [0.90, 0.80, 0.70, 0.90],
    ]
    targets = [0, 0, 1, 1]

    initial = LinearBinaryModel(weights=(0.0, 0.0, 0.0, 0.0), bias=0.0)
    trained = train_logistic_regression(features, targets)

    print("From-scratch logistic regression")
    print(" initial loss:", mean_loss(initial, features, targets))
    print(" trained loss:", mean_loss(trained, features, targets))
    print(" weights:", trained.weights)
    print(" bias:", trained.bias)

    artifact = Path(__file__).resolve().parents[1] / "artifacts" / "linear_binary_model.txt"
    deployed = load_model(artifact)
    sample = [0.5, 0.2, 0.7, 0.1]

    print("\nArtifact inference")
    print(" probability:", predict_probability(deployed, sample))
    print(" class:", predict(deployed, sample))

    layer = torch.nn.Linear(4, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([deployed.weights], dtype=torch.float64))
        layer.bias.copy_(torch.tensor([deployed.bias], dtype=torch.float64))

    tensor_sample = torch.tensor([sample], dtype=torch.float64)
    pytorch_logit = layer(tensor_sample).item()

    print("\nPyTorch logit using the same parameters")
    print(" logit:", pytorch_logit)


if __name__ == "__main__":
    main()

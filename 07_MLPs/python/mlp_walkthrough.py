"""Walkthrough for Module 07."""

from __future__ import annotations

from pathlib import Path

import torch

from mlp import forward_logit, he_initialize, load_model, predict_probability


def main() -> None:
    artifact = Path(__file__).resolve().parents[1] / "artifacts" / "mlp_binary_model.txt"
    model = load_model(artifact)
    sample = [0.6, 0.2, 0.8, 0.3]

    print("He initialization example")
    print(he_initialize(4, 3))

    print("\nReference MLP")
    print(" logit:", forward_logit(model, sample))
    print(" probability:", predict_probability(model, sample))

    network = torch.nn.Sequential(
        torch.nn.Linear(4, 3, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1, dtype=torch.float64),
    )
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor(model.w1, dtype=torch.float64))
        network[0].bias.copy_(torch.tensor(model.b1, dtype=torch.float64))
        network[2].weight.copy_(torch.tensor([model.w2], dtype=torch.float64))
        network[2].bias.copy_(torch.tensor([model.b2], dtype=torch.float64))

    normalized = torch.tensor(
        [
            [
                (x - m) / s
                for x, m, s in zip(sample, model.mean, model.std, strict=True)
            ]
        ],
        dtype=torch.float64,
    )
    print(" PyTorch logit:", network(normalized).item())


if __name__ == "__main__":
    main()

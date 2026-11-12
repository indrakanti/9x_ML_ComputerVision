"""Optimization walkthrough for Module 03."""

from __future__ import annotations

import torch
from optimization import AdamState, MomentumState, adam_step, momentum_step, optimize_scalar


def quadratic(value: float) -> float:
    return (value - 3.0) ** 2


def quadratic_grad(value: float) -> float:
    return 2.0 * (value - 3.0)


def main() -> None:
    print("Objective: f(w) = (w - 3)^2")

    gd = optimize_scalar(10.0, quadratic_grad, learning_rate=0.1, steps=20)
    print("\nGradient descent")
    print(" start:", gd[0], "end:", gd[-1], "loss:", quadratic(gd[-1]))

    momentum = MomentumState(value=10.0)
    for _ in range(20):
        momentum = momentum_step(momentum, quadratic_grad(momentum.value), learning_rate=0.05)
    print("\nMomentum")
    print(" end:", momentum.value, "loss:", quadratic(momentum.value))

    adam = AdamState(value=10.0)
    for _ in range(100):
        adam = adam_step(adam, quadratic_grad(adam.value), learning_rate=0.1)
    print("\nAdam")
    print(" end:", adam.value, "loss:", quadratic(adam.value))

    parameter = torch.tensor([10.0], requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    for _ in range(20):
        optimizer.zero_grad()
        loss = (parameter - 3.0).pow(2).sum()
        loss.backward()
        optimizer.step()

    print("\nPyTorch SGD")
    print(" end:", float(parameter.detach().item()))
    print(" loss:", float((parameter.detach() - 3.0).pow(2).item()))


if __name__ == "__main__":
    main()

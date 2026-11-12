from __future__ import annotations

import pytest
import torch

from optimization import (
    AdamState,
    MomentumState,
    adam_step,
    gradient_descent_step,
    momentum_step,
    optimize_scalar,
)


def quadratic_grad(value: float) -> float:
    return 2.0 * (value - 3.0)


def test_gradient_descent_step_moves_toward_minimum() -> None:
    updated = gradient_descent_step(10.0, quadratic_grad(10.0), 0.1)
    assert abs(updated - 3.0) < abs(10.0 - 3.0)


def test_gradient_descent_converges_on_quadratic() -> None:
    trajectory = optimize_scalar(10.0, quadratic_grad, learning_rate=0.1, steps=50)
    assert trajectory[-1] == pytest.approx(3.0, abs=1e-4)


def test_momentum_accumulates_state() -> None:
    state = MomentumState(value=10.0)
    first = momentum_step(state, quadratic_grad(state.value), learning_rate=0.01, beta=0.9)
    second = momentum_step(first, quadratic_grad(first.value), learning_rate=0.01, beta=0.9)

    assert first.velocity != 0.0
    assert second.velocity != first.velocity
    assert abs(second.value - 3.0) < abs(state.value - 3.0)


def test_adam_state_advances_and_improves_loss() -> None:
    state = AdamState(value=10.0)

    for _ in range(100):
        state = adam_step(state, quadratic_grad(state.value), learning_rate=0.1)

    assert state.step == 100
    assert abs(state.value - 3.0) < 0.1


def test_manual_sgd_matches_pytorch_on_scalar_quadratic() -> None:
    learning_rate = 0.1
    manual = 10.0

    parameter = torch.tensor([10.0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=learning_rate)

    for _ in range(10):
        manual = gradient_descent_step(manual, quadratic_grad(manual), learning_rate)

        optimizer.zero_grad()
        loss = (parameter - 3.0).pow(2).sum()
        loss.backward()
        optimizer.step()

    assert float(parameter.detach().item()) == pytest.approx(manual, rel=1e-12, abs=1e-12)


def test_invalid_learning_rate_is_rejected() -> None:
    with pytest.raises(ValueError):
        gradient_descent_step(1.0, 1.0, 0.0)


def test_invalid_adam_beta_is_rejected() -> None:
    with pytest.raises(ValueError):
        adam_step(AdamState(value=1.0), 1.0, learning_rate=0.1, beta1=1.0)

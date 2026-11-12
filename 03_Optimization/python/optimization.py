"""First-principles optimization helpers for Module 03."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math


ScalarFn = Callable[[float], float]


@dataclass
class SGDState:
    value: float


@dataclass
class MomentumState:
    value: float
    velocity: float = 0.0


@dataclass
class AdamState:
    value: float
    first_moment: float = 0.0
    second_moment: float = 0.0
    step: int = 0


def gradient_descent_step(value: float, gradient: float, learning_rate: float) -> float:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    return value - learning_rate * gradient


def momentum_step(
    state: MomentumState,
    gradient: float,
    *,
    learning_rate: float,
    beta: float = 0.9,
) -> MomentumState:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")

    velocity = beta * state.velocity + gradient
    value = state.value - learning_rate * velocity
    return MomentumState(value=value, velocity=velocity)


def adam_step(
    state: AdamState,
    gradient: float,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamState:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must satisfy 0 <= beta2 < 1")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")

    step = state.step + 1
    first = beta1 * state.first_moment + (1.0 - beta1) * gradient
    second = beta2 * state.second_moment + (1.0 - beta2) * gradient * gradient

    first_hat = first / (1.0 - beta1**step)
    second_hat = second / (1.0 - beta2**step)

    value = state.value - learning_rate * first_hat / (math.sqrt(second_hat) + epsilon)

    return AdamState(
        value=value,
        first_moment=first,
        second_moment=second,
        step=step,
    )


def optimize_scalar(
    initial_value: float,
    gradient_fn: ScalarFn,
    *,
    learning_rate: float,
    steps: int,
) -> list[float]:
    if steps < 0:
        raise ValueError("steps must be >= 0")

    trajectory = [initial_value]
    value = initial_value

    for _ in range(steps):
        value = gradient_descent_step(value, gradient_fn(value), learning_rate)
        trajectory.append(value)

    return trajectory

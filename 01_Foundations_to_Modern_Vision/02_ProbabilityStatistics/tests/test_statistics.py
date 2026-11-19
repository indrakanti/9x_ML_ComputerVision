from __future__ import annotations

import math

import numpy as np
import pytest
from statistics_helpers import (
    accuracy_confidence_interval,
    bayes_posterior,
    bernoulli_nll,
    correlation,
    covariance,
    gaussian_nll_fixed_variance,
)


def test_covariance_perfect_linear_relationship() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 3.0 * x

    assert covariance(x, y) == pytest.approx(3.75)


def test_correlation_perfect_positive_relationship() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 10.0 + 2.0 * x

    assert correlation(x, y) == pytest.approx(1.0)


def test_correlation_rejects_zero_variance() -> None:
    with pytest.raises(ValueError):
        correlation([1.0, 1.0], [1.0, 2.0])


def test_bayes_rare_event_example() -> None:
    evidence = (0.90 * 0.01) + (0.05 * 0.99)
    posterior = bayes_posterior(
        likelihood=0.90,
        prior=0.01,
        evidence_probability=evidence,
    )

    assert posterior == pytest.approx(0.1538461538)


def test_bernoulli_nll_rewards_better_probability() -> None:
    good = bernoulli_nll(target=1, probability=0.9)
    poor = bernoulli_nll(target=1, probability=0.2)

    assert good < poor


def test_gaussian_nll_differs_from_sse_by_fixed_scale_and_constant() -> None:
    residuals = np.array([1.0, -2.0, 0.5])
    variance = 2.0

    nll = gaussian_nll_fixed_variance(residuals=residuals, variance=variance)
    expected = (
        residuals.size * 0.5 * math.log(2.0 * math.pi * variance)
        + float(np.square(residuals).sum()) / (2.0 * variance)
    )

    assert nll == pytest.approx(expected)


def test_accuracy_interval_shrinks_with_more_samples() -> None:
    small = accuracy_confidence_interval(correct=90, total=100)
    large = accuracy_confidence_interval(correct=900, total=1000)

    small_width = small[1] - small[0]
    large_width = large[1] - large[0]

    assert large_width < small_width


def test_accuracy_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        accuracy_confidence_interval(correct=11, total=10)

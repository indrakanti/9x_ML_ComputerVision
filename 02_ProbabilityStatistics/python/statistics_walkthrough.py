"""Probability and statistics walkthrough for vision ML."""

from __future__ import annotations

import math

import numpy as np

from statistics_helpers import (
    accuracy_confidence_interval,
    bayes_posterior,
    bernoulli_nll,
    correlation,
    covariance,
    gaussian_nll_fixed_variance,
)


def main() -> None:
    rng = np.random.default_rng(9)

    image = np.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        dtype=np.float64,
    )
    print("Image mean:", float(image.mean()))
    print("Image population variance:", float(image.var()))

    x = np.linspace(0.0, 10.0, 100)
    y = 2.0 * x + rng.normal(0.0, 1.0, size=x.shape)

    print("\nCovariance:", covariance(x, y))
    print("Correlation:", correlation(x, y))

    gaussian = rng.normal(loc=100.0, scale=15.0, size=10_000)
    print("\nGaussian empirical mean/std:", float(gaussian.mean()), float(gaussian.std()))

    posterior = bayes_posterior(
        likelihood=0.90,
        prior=0.01,
        evidence_probability=(0.90 * 0.01) + (0.05 * 0.99),
    )
    print("\nRare-event detector posterior:", posterior)

    print("\nBernoulli NLL, y=1 p=0.9:", bernoulli_nll(target=1, probability=0.9))

    residuals = np.array([0.2, -0.1, 0.4, -0.3], dtype=np.float64)
    print(
        "Gaussian fixed-variance NLL:",
        gaussian_nll_fixed_variance(residuals=residuals, variance=1.0),
    )
    print("Squared-error sum:", float(np.square(residuals).sum()))

    low, high = accuracy_confidence_interval(correct=90, total=100)
    print(f"\n90/100 approximate 95% CI: [{low:.3f}, {high:.3f}]")

    empirical = float(np.mean(rng.random(100_000) < 0.25))
    print("\nEmpirical P(U < 0.25):", empirical)
    print("Absolute error:", abs(empirical - 0.25))
    print("Expected Monte Carlo scale ~", 1.0 / math.sqrt(100_000))


if __name__ == "__main__":
    main()

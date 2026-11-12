"""Statistical helpers used across the ML Computer Vision course."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def covariance(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Return population covariance for two one-dimensional numeric arrays."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)

    if xa.ndim != 1 or ya.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if xa.size != ya.size:
        raise ValueError("x and y must have the same length")
    if xa.size == 0:
        raise ValueError("x and y must be non-empty")

    return float(np.mean((xa - xa.mean()) * (ya - ya.mean())))


def correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Return Pearson correlation for two one-dimensional numeric arrays."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)

    cov = covariance(xa, ya)
    sx = float(xa.std())
    sy = float(ya.std())

    if sx == 0.0 or sy == 0.0:
        raise ValueError("correlation is undefined for zero-variance input")

    return cov / (sx * sy)


def bayes_posterior(*, likelihood: float, prior: float, evidence_probability: float) -> float:
    """Compute P(A|B)=P(B|A)P(A)/P(B)."""
    for name, value in (
        ("likelihood", likelihood),
        ("prior", prior),
        ("evidence_probability", evidence_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    if evidence_probability == 0.0:
        raise ValueError("evidence_probability must be > 0")

    posterior = likelihood * prior / evidence_probability
    if not 0.0 <= posterior <= 1.0 + 1e-12:
        raise ValueError("inconsistent probability inputs")

    return min(max(posterior, 0.0), 1.0)


def bernoulli_nll(*, target: int, probability: float, epsilon: float = 1e-12) -> float:
    """Return Bernoulli negative log-likelihood for target 0 or 1."""
    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")

    p = min(max(probability, epsilon), 1.0 - epsilon)
    return float(-(target * math.log(p) + (1 - target) * math.log(1.0 - p)))


def gaussian_nll_fixed_variance(
    *,
    residuals: npt.ArrayLike,
    variance: float,
) -> float:
    """Return Gaussian negative log-likelihood for zero-mean residuals."""
    if variance <= 0.0:
        raise ValueError("variance must be > 0")

    r = np.asarray(residuals, dtype=np.float64)
    if r.size == 0:
        raise ValueError("residuals must be non-empty")

    constant = 0.5 * math.log(2.0 * math.pi * variance)
    return float(np.sum(constant + np.square(r) / (2.0 * variance)))


def accuracy_confidence_interval(
    *,
    correct: int,
    total: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """Approximate two-sided confidence interval for a binomial proportion.

    This uses the simple normal approximation for teaching. Later evaluation
    modules can compare Wilson/bootstrap intervals for small or extreme samples.
    """
    if total <= 0:
        raise ValueError("total must be > 0")
    if correct < 0 or correct > total:
        raise ValueError("correct must satisfy 0 <= correct <= total")
    if z_value <= 0.0:
        raise ValueError("z_value must be > 0")

    p_hat = correct / total
    standard_error = math.sqrt(p_hat * (1.0 - p_hat) / total)
    half_width = z_value * standard_error

    return max(0.0, p_hat - half_width), min(1.0, p_hat + half_width)

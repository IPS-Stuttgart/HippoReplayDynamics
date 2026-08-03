"""Held-out scoring under a posterior inferred from training observations only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class FrozenPosteriorPredictiveScore:
    """Predictive log score obtained without updating the latent posterior."""

    total_log_score: float
    mean_log_score_per_time_bin: float
    per_time_log_score: np.ndarray
    posterior_sha256: str


def normalized_log_posterior(log_posterior: np.ndarray) -> np.ndarray:
    """Validate and row-normalize a time-by-state log posterior."""

    values = np.asarray(log_posterior, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("log_posterior must have non-empty shape (time, state)")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("log_posterior must not contain NaN or +inf")
    normalizer = logsumexp(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(normalizer)):
        raise ValueError("every posterior row must contain finite probability mass")
    return values - normalizer


def posterior_sha256(log_posterior: np.ndarray) -> str:
    """Return a stable digest of the normalized posterior used for prediction."""

    normalized = np.ascontiguousarray(
        normalized_log_posterior(log_posterior),
        dtype="<f8",
    )
    shape = np.asarray(normalized.shape, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(shape.tobytes())
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def frozen_smoothed_marginal_log_score(
    log_posterior: np.ndarray,
    heldout_log_emissions: np.ndarray,
) -> FrozenPosteriorPredictiveScore:
    """Score held-out observations under an unchanged training posterior.

    At each time bin this computes ``log sum_x p(x_t | train) p(test_t | x_t)``
    and sums the resulting logarithmic scores. The smoothed marginal posterior
    is normalized and hashed before held-out emissions are consulted. Held-out
    observations therefore cannot update the latent trajectory.

    This is a frozen marginal posterior score, not the exact joint path-level
    conditional evidence ``log p(train, test) - log p(train)``.
    """

    normalized = normalized_log_posterior(log_posterior)
    emissions = np.asarray(heldout_log_emissions, dtype=float)
    if emissions.shape != normalized.shape:
        raise ValueError(
            "heldout_log_emissions must match log_posterior shape; "
            f"got {emissions.shape} and {normalized.shape}"
        )
    if np.any(np.isnan(emissions)) or np.any(emissions == np.inf):
        raise ValueError("heldout_log_emissions must not contain NaN or +inf")
    if not np.all(np.any(np.isfinite(emissions), axis=1)):
        raise ValueError("every held-out emission row must contain finite mass")

    per_time = logsumexp(normalized + emissions, axis=1)
    if not np.all(np.isfinite(per_time)):
        raise ValueError("frozen posterior predictive score must be finite")
    return FrozenPosteriorPredictiveScore(
        total_log_score=float(per_time.sum()),
        mean_log_score_per_time_bin=float(per_time.mean()),
        per_time_log_score=per_time,
        posterior_sha256=posterior_sha256(normalized),
    )

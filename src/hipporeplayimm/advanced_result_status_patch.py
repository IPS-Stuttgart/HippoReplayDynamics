"""Runtime guards for advanced result diagnostics.

Older score-table artifacts can contain a ``status`` column whose successful rows
round-trip through CSV as blanks or nulls.  The advanced diagnostics used a
literal ``status == 'success'`` filter, which dropped those legacy rows before
margin, wrong-map, and paired-model summaries were computed.

The same patch point also keeps posterior-predictive diagnostics from accepting
impossible count tables.  Those helpers operate on observed counts, expected
Poisson means, and optional predictive variances, all of which must be finite and
nonnegative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .evidence_status_coercion import _status_is_success_or_missing

_PATCHED_FLAG = "_advanced_result_status_patch_applied"


def apply_advanced_result_status_patch() -> None:
    """Patch advanced diagnostics status handling and count-input guards."""

    from . import advanced_result_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
        return

    def successful_rows(scores: pd.DataFrame) -> pd.DataFrame:
        if scores.empty:
            return scores.copy()
        if "status" not in scores.columns:
            return scores.copy()
        mask = scores["status"].map(_status_is_success_or_missing).astype(bool)
        return scores[mask].copy()

    def posterior_predictive_count_checks(
        observed_counts: np.ndarray,
        expected_counts: np.ndarray,
        *,
        variance_counts: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compare observed counts with posterior-predictive expected counts."""

        observed = _validated_nonnegative_matrix(observed_counts, "observed_counts")
        expected = _validated_nonnegative_matrix(expected_counts, "expected_counts")
        if observed.shape != expected.shape:
            raise ValueError("observed_counts and expected_counts must have matching shapes")
        if variance_counts is None:
            variance = np.maximum(expected, np.finfo(float).eps)
        else:
            variance = _validated_nonnegative_matrix(variance_counts, "variance_counts")
            if variance.shape != observed.shape:
                raise ValueError("variance_counts must match observed_counts")

        residual = observed - expected
        z = residual / np.sqrt(np.maximum(variance, np.finfo(float).eps))
        rows = [
            {
                "predictive_check": "total_spike_count",
                "observed": float(observed.sum()),
                "expected": float(expected.sum()),
                "z_score": float(residual.sum() / np.sqrt(np.maximum(variance.sum(), np.finfo(float).eps))),
            },
            {
                "predictive_check": "silent_bin_fraction",
                "observed": float(np.mean(observed.sum(axis=1) == 0.0)),
                "expected": float(np.mean(np.exp(-expected.sum(axis=1)))),
                "z_score": np.nan,
            },
            {
                "predictive_check": "mean_abs_cell_z",
                "observed": float(np.mean(np.abs(z))),
                "expected": 0.0,
                "z_score": np.nan,
            },
            {
                "predictive_check": "max_abs_cell_z",
                "observed": float(np.max(np.abs(z))),
                "expected": 0.0,
                "z_score": np.nan,
            },
        ]
        return pd.DataFrame(rows)

    def posterior_predictive_poisson_log_score(observed_counts: np.ndarray, expected_counts: np.ndarray) -> float:
        """Return a Poisson posterior-predictive log score for observed counts."""

        observed = _validated_nonnegative_matrix(observed_counts, "observed_counts")
        expected = _validated_nonnegative_matrix(expected_counts, "expected_counts")
        if observed.shape != expected.shape:
            raise ValueError("observed_counts and expected_counts must have matching shapes")
        expected = np.maximum(expected, np.finfo(float).tiny)
        return float(np.sum(observed * np.log(expected) - expected - gammaln(observed + 1.0)))

    diagnostics._successful_rows = successful_rows
    diagnostics.posterior_predictive_count_checks = posterior_predictive_count_checks
    diagnostics.posterior_predictive_poisson_log_score = posterior_predictive_poisson_log_score
    setattr(diagnostics, _PATCHED_FLAG, True)


def _validated_nonnegative_matrix(values: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_time, n_cells)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return array


__all__ = ["apply_advanced_result_status_patch"]

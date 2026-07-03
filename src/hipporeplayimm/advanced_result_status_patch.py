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
_SUCCESSFUL_ROWS_WRAPPER_FLAG = "_advanced_result_successful_rows_wrapper"
_COUNT_CHECKS_WRAPPER_FLAG = "_advanced_result_count_checks_wrapper"
_POISSON_SCORE_WRAPPER_FLAG = "_advanced_result_poisson_score_wrapper"


def apply_advanced_result_status_patch() -> None:
    """Patch advanced diagnostics status handling and count-input guards."""

    from . import advanced_result_diagnostics as diagnostics
    from . import advanced_result_margin_duplicate_patch as margin_duplicate_patch

    margin_duplicate_patch.apply_advanced_result_margin_duplicate_patch()
    if getattr(diagnostics, _PATCHED_FLAG, False) and _advanced_result_patch_current(diagnostics):
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

        observed = _validated_nonnegative_matrix(observed_counts, "observed_counts", integral=True)
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

        observed = _validated_nonnegative_matrix(observed_counts, "observed_counts", integral=True)
        expected = _validated_nonnegative_matrix(expected_counts, "expected_counts")
        if observed.shape != expected.shape:
            raise ValueError("observed_counts and expected_counts must have matching shapes")
        expected = np.maximum(expected, np.finfo(float).tiny)
        return float(np.sum(observed * np.log(expected) - expected - gammaln(observed + 1.0)))

    setattr(successful_rows, _SUCCESSFUL_ROWS_WRAPPER_FLAG, True)
    setattr(posterior_predictive_count_checks, _COUNT_CHECKS_WRAPPER_FLAG, True)
    setattr(posterior_predictive_poisson_log_score, _POISSON_SCORE_WRAPPER_FLAG, True)

    diagnostics._successful_rows = successful_rows
    diagnostics.posterior_predictive_count_checks = posterior_predictive_count_checks
    diagnostics.posterior_predictive_poisson_log_score = posterior_predictive_poisson_log_score
    setattr(diagnostics, _PATCHED_FLAG, True)


def _advanced_result_patch_current(diagnostics) -> bool:
    """Return whether advanced-diagnostic aliases still point to this patch."""

    return all(
        getattr(getattr(diagnostics, name, None), flag, False)
        for name, flag in (
            ("_successful_rows", _SUCCESSFUL_ROWS_WRAPPER_FLAG),
            ("posterior_predictive_count_checks", _COUNT_CHECKS_WRAPPER_FLAG),
            ("posterior_predictive_poisson_log_score", _POISSON_SCORE_WRAPPER_FLAG),
        )
    )


def _contains_boolean_values(values: np.ndarray) -> bool:
    """Return True when a count-like matrix contains actual boolean values."""

    if np.issubdtype(values.dtype, np.bool_):
        return True
    if values.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in values.flat)
    return False


def _validated_nonnegative_matrix(values: np.ndarray, name: str, *, integral: bool = False) -> np.ndarray:
    try:
        raw_values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if _contains_boolean_values(raw_values):
        raise ValueError(f"{name} must contain numeric count values, not booleans")
    try:
        array = raw_values.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_time, n_cells)")
    if 0 in array.shape:
        raise ValueError(f"{name} must contain at least one time bin and one cell")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if integral and not np.all(np.isclose(array, np.rint(array), rtol=0.0, atol=1.0e-12)):
        raise ValueError(f"{name} must contain integer count values")
    return array


__all__ = ["apply_advanced_result_status_patch"]

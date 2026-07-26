"""Runtime guards for advanced result diagnostics.

Older score-table artifacts can contain a ``status`` column whose successful rows
round-trip through CSV as blanks or nulls, or through NumPy/HDF5-backed tables as
byte strings. The advanced diagnostics used a literal ``status == 'success'``
filter, which dropped those legacy rows before margin, wrong-map, and paired-model
summaries were computed.

Evidence-support diagnostics can arrive through the same byte-valued table
scalars. Decode and normalize those labels before support classification so exact,
truncated, degenerate, and particle-approximation provenance is not lost.

The same patch point also keeps posterior-predictive diagnostics from accepting
impossible count tables. Those helpers operate on observed counts, expected
Poisson means, and optional predictive variances, all of which must be finite and
nonnegative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln, xlogy

_PATCHED_FLAG = "_advanced_result_status_patch_applied"
_SUCCESSFUL_ROWS_WRAPPER_FLAG = "_advanced_result_successful_rows_wrapper"
_COUNT_CHECKS_WRAPPER_FLAG = "_advanced_result_count_checks_wrapper"
_POISSON_SCORE_WRAPPER_FLAG = "_advanced_result_poisson_score_wrapper"
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_DERIVED_RANGE_ERROR = "posterior-predictive diagnostics exceed floating-point range"


def _decoded_text(value: object) -> str:
    """Return table scalar text, decoding byte-valued values."""

    if isinstance(value, (bytes, bytearray, memoryview, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _status_text(value: object) -> str:
    """Return normalized status text, decoding byte-valued table scalars."""

    return _decoded_text(value).lower()


def _serialized_evidence_support_labels(text: str, known_values: set[str]) -> list[str]:
    """Recover known labels from CSV-serialized list or array cells."""

    normalized = text.lower()
    if normalized in known_values:
        return [normalized]
    for delimiter in "[](){}'\" ,;|":
        normalized = normalized.replace(delimiter, " ")
    recognized = [token for token in normalized.split() if token in known_values]
    return recognized if recognized else [text]


def _evidence_support_labels(value: object) -> list[str]:
    """Extract support labels while decoding bytes and serialized containers."""

    from . import evidence_reporting

    items = (
        [value]
        if isinstance(value, (bytes, bytearray, memoryview, np.bytes_))
        else evidence_reporting._flatten_support_value(value)
    )
    known_values = {
        evidence_reporting.EXACT_EVIDENCE_SUPPORT,
        evidence_reporting.TRUNCATED_EVIDENCE_SUPPORT,
        evidence_reporting.DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
        evidence_reporting.PYRECEST_PARTICLE_EVIDENCE_SUPPORT,
        evidence_reporting.EVIDENCE_COMPARISON_NOT_SCORED,
        evidence_reporting.EVIDENCE_COMPARISON_UNKNOWN,
        "unknown",
    }
    labels: list[str] = []
    for item in items:
        if evidence_reporting._is_missing_scalar(item):
            continue
        text = _decoded_text(item)
        if (
            not text
            or text.lower()
            in evidence_reporting._MISSING_EVIDENCE_SUPPORT_STRINGS
        ):
            continue
        labels.extend(_serialized_evidence_support_labels(text, known_values))
    return labels


def _is_missing_status(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True
    return _status_text(value) in _MISSING_STATUS_VALUES


def _status_is_success_or_missing(value: object) -> bool:
    if _is_missing_status(value):
        return True
    return _status_text(value) == "success"


def _normalize_status_value(value: object) -> object:
    return "success" if _status_is_success_or_missing(value) else value


def _patch_status_helpers() -> None:
    """Keep package-level status and support aliases byte-aware."""

    from . import accuracy_model_probability_status_patch as accuracy_status
    from . import evidence_reliability
    from . import evidence_reporting
    from . import evidence_status_coercion
    from . import recovery_diagnostics_bool_patch
    from . import result_quality_gates
    from . import simulation_best_row_flags

    evidence_status_coercion._status_is_success_or_missing = (
        _status_is_success_or_missing
    )
    evidence_status_coercion._normalize_status_value = _normalize_status_value
    evidence_reporting._is_missing_status = _is_missing_status
    evidence_reporting._status_is_success_or_missing = _status_is_success_or_missing
    evidence_reporting._evidence_support_labels = _evidence_support_labels
    result_quality_gates._is_missing_status = _is_missing_status
    result_quality_gates._status_is_success_or_missing = _status_is_success_or_missing
    simulation_best_row_flags._status_is_success = _status_is_success_or_missing
    recovery_diagnostics_bool_patch._status_is_success_or_missing = (
        _status_is_success_or_missing
    )
    evidence_reliability._status_is_success_or_missing = _status_is_success_or_missing
    accuracy_status._normalize_status_value = _normalize_status_value


def apply_advanced_result_status_patch() -> None:
    """Patch advanced diagnostics status handling and count-input guards."""

    from . import advanced_result_diagnostics as diagnostics
    from . import advanced_result_margin_duplicate_patch as margin_duplicate_patch
    from . import (
        advanced_result_place_field_cell_id_validation as place_field_cell_id_validation,
    )

    margin_duplicate_patch.apply_advanced_result_margin_duplicate_patch()
    place_field_cell_id_validation.apply_advanced_result_place_field_cell_id_validation_patch()
    _patch_status_helpers()
    if getattr(diagnostics, _PATCHED_FLAG, False) and _advanced_result_patch_current(
        diagnostics
    ):
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

        observed = _validated_nonnegative_matrix(
            observed_counts,
            "observed_counts",
            integral=True,
        )
        expected = _validated_nonnegative_matrix(expected_counts, "expected_counts")
        if observed.shape != expected.shape:
            raise ValueError(
                "observed_counts and expected_counts must have matching shapes"
            )
        if variance_counts is None:
            variance = np.maximum(expected, np.finfo(float).eps)
        else:
            variance = _validated_nonnegative_matrix(
                variance_counts,
                "variance_counts",
            )
            if variance.shape != observed.shape:
                raise ValueError("variance_counts must match observed_counts")

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            residual = observed - expected
            stabilized_variance = np.maximum(variance, np.finfo(float).eps)
            z = residual / np.sqrt(stabilized_variance)
            observed_total = np.sum(observed, dtype=float)
            expected_total = np.sum(expected, dtype=float)
            residual_total = np.sum(residual, dtype=float)
            variance_total = np.sum(variance, dtype=float)
            observed_by_time = np.sum(observed, axis=1, dtype=float)
            expected_by_time = np.sum(expected, axis=1, dtype=float)
            total_z = residual_total / np.sqrt(
                np.maximum(variance_total, np.finfo(float).eps)
            )
            silent_expected = np.exp(-expected_by_time)

        _require_finite_derived_diagnostics(
            z,
            observed_total,
            expected_total,
            residual_total,
            variance_total,
            observed_by_time,
            expected_by_time,
            total_z,
            silent_expected,
        )
        rows = [
            {
                "predictive_check": "total_spike_count",
                "observed": float(observed_total),
                "expected": float(expected_total),
                "z_score": float(total_z),
            },
            {
                "predictive_check": "silent_bin_fraction",
                "observed": float(np.mean(observed_by_time == 0.0)),
                "expected": float(np.mean(silent_expected)),
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

    def posterior_predictive_poisson_log_score(
        observed_counts: np.ndarray,
        expected_counts: np.ndarray,
    ) -> float:
        """Return a Poisson posterior-predictive log score for observed counts."""

        observed = _validated_nonnegative_matrix(
            observed_counts,
            "observed_counts",
            integral=True,
        )
        expected = _validated_nonnegative_matrix(expected_counts, "expected_counts")
        if observed.shape != expected.shape:
            raise ValueError(
                "observed_counts and expected_counts must have matching shapes"
            )
        return float(
            np.sum(xlogy(observed, expected) - expected - gammaln(observed + 1.0))
        )

    setattr(successful_rows, _SUCCESSFUL_ROWS_WRAPPER_FLAG, True)
    setattr(posterior_predictive_count_checks, _COUNT_CHECKS_WRAPPER_FLAG, True)
    setattr(
        posterior_predictive_poisson_log_score,
        _POISSON_SCORE_WRAPPER_FLAG,
        True,
    )

    diagnostics._successful_rows = successful_rows
    diagnostics.posterior_predictive_count_checks = posterior_predictive_count_checks
    diagnostics.posterior_predictive_poisson_log_score = (
        posterior_predictive_poisson_log_score
    )
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


def _validated_nonnegative_matrix(
    values: np.ndarray,
    name: str,
    *,
    integral: bool = False,
) -> np.ndarray:
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
        raise ValueError(
            f"{name} must contain at least one time bin and one cell"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if integral and not np.all(
        np.isclose(array, np.rint(array), rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(f"{name} must contain integer count values")
    return array


def _require_finite_derived_diagnostics(*values: object) -> None:
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError(_DERIVED_RANGE_ERROR)


__all__ = ["apply_advanced_result_status_patch"]

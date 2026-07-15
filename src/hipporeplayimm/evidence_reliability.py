"""Reliability flags for event-level replay evidence rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evidence_reporting
from .evidence_status_coercion import _status_is_success_or_missing


DEFAULT_MIN_SPIKES = 3
DEFAULT_MIN_TIME_BINS = 2
DEFAULT_MIN_CANDIDATE_LOG_MASS = np.log(0.95)
DEFAULT_MAX_TERMINAL_ENTROPY = np.inf
RELIABILITY_FLAG_COLUMNS = (
    "event_reliable",
    "event_reliability_reasons",
    "event_low_spike_count",
    "event_too_few_time_bins",
    "event_low_candidate_mass",
    "event_high_terminal_entropy",
    "event_invalid_numeric_metric",
)


def event_reliability_flags(
    row: pd.Series,
    *,
    min_spikes: int = DEFAULT_MIN_SPIKES,
    min_time_bins: int = DEFAULT_MIN_TIME_BINS,
    min_candidate_log_mass: float = DEFAULT_MIN_CANDIDATE_LOG_MASS,
    max_terminal_entropy: float = DEFAULT_MAX_TERMINAL_ENTROPY,
) -> dict[str, object]:
    """Return interpretable reliability flags for one score row."""

    min_spikes = _nonnegative_integer_threshold("min_spikes", min_spikes)
    min_time_bins = _nonnegative_integer_threshold("min_time_bins", min_time_bins)
    min_candidate_log_mass = _real_threshold(
        "min_candidate_log_mass",
        min_candidate_log_mass,
    )
    max_terminal_entropy = _nonnegative_real_threshold(
        "max_terminal_entropy",
        max_terminal_entropy,
    )

    reasons: list[str] = []
    invalid_numeric_metric = False
    status = row.get("status", "success")
    if not _status_is_success_or_missing(status):
        reasons.append("score_failure")
    n_spikes, invalid_metric = _first_finite(row, ("n_spikes", "test_spikes"))
    invalid_numeric_metric |= invalid_metric
    if np.isfinite(n_spikes) and n_spikes < min_spikes:
        reasons.append("low_spike_count")
    n_time, invalid_metric = _as_float(row.get("n_time", np.nan))
    invalid_numeric_metric |= invalid_metric
    if np.isfinite(n_time) and n_time < min_time_bins:
        reasons.append("too_few_time_bins")
    support_labels = _evidence_support_labels_for_reliability(row)
    if evidence_reporting.DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT in support_labels:
        reasons.append("degenerate_single_bin")
    candidate_mass, invalid_metric = _first_finite(
        row,
        (
            "diagnostic_mean_candidate_log_mass",
            "mean_candidate_log_mass",
        ),
    )
    invalid_numeric_metric |= invalid_metric
    if np.isfinite(candidate_mass) and candidate_mass < min_candidate_log_mass:
        reasons.append("low_candidate_mass")
    entropy, invalid_metric = _first_finite(
        row,
        (
            "diagnostic_terminal_posterior_entropy",
            "terminal_posterior_entropy",
        ),
    )
    invalid_numeric_metric |= invalid_metric
    if np.isfinite(entropy) and entropy > max_terminal_entropy:
        reasons.append("high_terminal_entropy")
    if invalid_numeric_metric:
        reasons.append("invalid_numeric_metric")
    return {
        "event_reliable": len(reasons) == 0,
        "event_reliability_reasons": ";".join(reasons),
        "event_low_spike_count": "low_spike_count" in reasons,
        "event_too_few_time_bins": "too_few_time_bins" in reasons,
        "event_low_candidate_mass": "low_candidate_mass" in reasons,
        "event_high_terminal_entropy": "high_terminal_entropy" in reasons,
        "event_invalid_numeric_metric": "invalid_numeric_metric" in reasons,
    }


def add_event_reliability_flags(
    df: pd.DataFrame,
    *,
    min_spikes: int = DEFAULT_MIN_SPIKES,
    min_time_bins: int = DEFAULT_MIN_TIME_BINS,
    min_candidate_log_mass: float = DEFAULT_MIN_CANDIDATE_LOG_MASS,
    max_terminal_entropy: float = DEFAULT_MAX_TERMINAL_ENTROPY,
) -> pd.DataFrame:
    validated_thresholds = {
        "min_spikes": _nonnegative_integer_threshold("min_spikes", min_spikes),
        "min_time_bins": _nonnegative_integer_threshold("min_time_bins", min_time_bins),
        "min_candidate_log_mass": _real_threshold(
            "min_candidate_log_mass",
            min_candidate_log_mass,
        ),
        "max_terminal_entropy": _nonnegative_real_threshold(
            "max_terminal_entropy",
            max_terminal_entropy,
        ),
    }
    base = df.copy()
    existing_flag_columns = [column for column in RELIABILITY_FLAG_COLUMNS if column in base.columns]
    if existing_flag_columns:
        base = base.drop(columns=existing_flag_columns)
    if base.empty:
        return base
    flags = pd.DataFrame(
        [
            event_reliability_flags(row, **validated_thresholds)
            for _, row in base.iterrows()
        ],
        index=base.index,
    )
    return pd.concat([base, flags], axis=1)


def _nonnegative_integer_threshold(name: str, value: object) -> int:
    """Return a canonical nonnegative integer threshold."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    item = raw.item()
    if isinstance(item, (bool, np.bool_, str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(item, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        integer = int(item)
        exact = bool(item == integer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if not exact or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _real_threshold(name: str, value: object) -> float:
    """Return a scalar real threshold while rejecting NaN and coercive types."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar and cannot be NaN") from exc
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a real scalar and cannot be NaN")
    item = raw.item()
    if isinstance(item, (bool, np.bool_, str, bytes, np.str_, np.bytes_)):
        raise ValueError(f"{name} must be a real scalar and cannot be NaN")
    if isinstance(item, (complex, np.complexfloating)):
        raise ValueError(f"{name} must be a real scalar and cannot be NaN")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar and cannot be NaN") from exc
    if np.isnan(numeric):
        raise ValueError(f"{name} must be a real scalar and cannot be NaN")
    return numeric


def _nonnegative_real_threshold(name: str, value: object) -> float:
    """Return a nonnegative scalar real threshold, allowing positive infinity."""

    numeric = _real_threshold(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _evidence_support_labels_for_reliability(row: pd.Series) -> list[str]:
    """Return explicit or diagnostic-derived support labels for reliability checks."""

    explicit_labels = evidence_reporting._evidence_support_labels(row.get("evidence_support", ""))
    if explicit_labels:
        return explicit_labels
    return evidence_reporting._evidence_support_labels(evidence_reporting.evidence_support_from_row(row))


def _as_float(value) -> tuple[float, bool]:
    if _is_missing_scalar(value):
        return float("nan"), False
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return _coerced_metric_float(value)
    if array.shape != ():
        return float("nan"), True
    if np.issubdtype(array.dtype, np.bool_):
        return float("nan"), True
    if np.issubdtype(array.dtype, np.complexfloating):
        return float("nan"), True
    if array.dtype == object:
        try:
            item = array.item()
        except ValueError:
            return float("nan"), True
        if _is_missing_scalar(item):
            return float("nan"), False
        if isinstance(item, (bool, np.bool_)):
            return float("nan"), True
        if isinstance(item, (complex, np.complexfloating)):
            return float("nan"), True
        return _coerced_metric_float(item)
    return _coerced_metric_float(array)


def _coerced_metric_float(value) -> tuple[float, bool]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        integer = int(value)
        try:
            numeric = float(integer)
        except OverflowError:
            finite_limit = np.finfo(float).max
            numeric = finite_limit if integer > 0 else -finite_limit
        return numeric, False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan"), True
    if np.isnan(numeric):
        return float("nan"), False
    if not np.isfinite(numeric):
        return numeric, True
    return numeric, False


def _is_missing_scalar(value) -> bool:
    if value is None:
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.shape != ():
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_finite(row: pd.Series, columns: tuple[str, ...]) -> tuple[float, bool]:
    invalid_numeric_metric = False
    for column in columns:
        value, invalid_metric = _as_float(row.get(column, np.nan))
        invalid_numeric_metric |= invalid_metric
        if np.isfinite(value):
            return value, invalid_numeric_metric
    return float("nan"), invalid_numeric_metric

"""Reliability flags for event-level replay evidence rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    reasons: list[str] = []
    status = row.get("status", "success")
    if not _status_is_success_or_missing(status):
        reasons.append("score_failure")
    n_spikes = _as_float(row.get("n_spikes", row.get("test_spikes", np.nan)))
    if np.isfinite(n_spikes) and n_spikes < min_spikes:
        reasons.append("low_spike_count")
    n_time = _as_float(row.get("n_time", np.nan))
    if np.isfinite(n_time) and n_time < min_time_bins:
        reasons.append("too_few_time_bins")
    support = str(row.get("evidence_support", ""))
    if support == "degenerate_single_bin":
        reasons.append("degenerate_single_bin")
    candidate_mass = _first_finite(
        row,
        (
            "diagnostic_mean_candidate_log_mass",
            "mean_candidate_log_mass",
        ),
    )
    if np.isfinite(candidate_mass) and candidate_mass < min_candidate_log_mass:
        reasons.append("low_candidate_mass")
    entropy = _first_finite(
        row,
        (
            "diagnostic_terminal_posterior_entropy",
            "terminal_posterior_entropy",
        ),
    )
    if np.isfinite(entropy) and entropy > max_terminal_entropy:
        reasons.append("high_terminal_entropy")
    return {
        "event_reliable": len(reasons) == 0,
        "event_reliability_reasons": ";".join(reasons),
        "event_low_spike_count": "low_spike_count" in reasons,
        "event_too_few_time_bins": "too_few_time_bins" in reasons,
        "event_low_candidate_mass": "low_candidate_mass" in reasons,
        "event_high_terminal_entropy": "high_terminal_entropy" in reasons,
    }


def add_event_reliability_flags(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    base = df.copy()
    existing_flag_columns = [column for column in RELIABILITY_FLAG_COLUMNS if column in base.columns]
    if existing_flag_columns:
        base = base.drop(columns=existing_flag_columns)
    if base.empty:
        return base
    flags = pd.DataFrame([event_reliability_flags(row, **kwargs) for _, row in base.iterrows()], index=base.index)
    return pd.concat([base, flags], axis=1)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _first_finite(row: pd.Series, columns: tuple[str, ...]) -> float:
    for column in columns:
        value = _as_float(row.get(column, np.nan))
        if np.isfinite(value):
            return value
    return float("nan")

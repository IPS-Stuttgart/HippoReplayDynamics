"""Runtime fixes for accuracy-upgrade diagnostics and reverse-time emissions."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

from .encoding import LogEmissionTensor
from .evidence_status_coercion import _normalize_status_value

_STATUS_PATCHED_FLAG = "_model_probability_status_patch_applied"
_REVERSE_PATCHED_FLAG = "_accuracy_reverse_duration_patch_applied"
_REVERSE_ORIGINAL_ATTR = "_accuracy_reverse_duration_original"


def apply_model_probability_status_patch() -> None:
    """Install accuracy-upgrade runtime fixes."""

    from . import accuracy_upgrades

    _patch_model_probability_diagnostics(accuracy_upgrades)
    _patch_reverse_emissions(accuracy_upgrades)


def _patch_model_probability_diagnostics(accuracy_upgrades) -> None:
    """Install input normalization for accuracy-upgrade probability summaries."""

    if getattr(accuracy_upgrades, _STATUS_PATCHED_FLAG, False):
        return

    original = accuracy_upgrades.model_probability_diagnostics

    @wraps(original)
    def model_probability_diagnostics(
        scores: pd.DataFrame,
        *,
        evidence_column: str = "log_evidence",
        group_columns=("session", "event_index"),
    ) -> pd.DataFrame:
        normalized = scores
        if not scores.empty and ("status" in scores.columns or evidence_column in scores.columns):
            normalized = scores.copy()
            if "status" in normalized.columns:
                normalized["status"] = normalized["status"].map(_normalize_status_value)
            if evidence_column in normalized.columns:
                normalized[evidence_column] = pd.to_numeric(normalized[evidence_column], errors="coerce")
                normalized = normalized.dropna(subset=[evidence_column])
        return original(
            normalized,
            evidence_column=evidence_column,
            group_columns=group_columns,
        )

    accuracy_upgrades.model_probability_diagnostics = model_probability_diagnostics
    setattr(accuracy_upgrades, _STATUS_PATCHED_FLAG, True)


def _patch_reverse_emissions(accuracy_upgrades) -> None:
    """Build reversed accuracy-upgrade emissions with duration metadata attached."""

    current = accuracy_upgrades.reverse_emissions
    if getattr(current, _REVERSE_PATCHED_FLAG, False):
        return

    @wraps(current)
    def reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
        bin_durations = _reversed_duration_vector(
            getattr(emissions, "bin_durations", None),
            expected_length=emissions.n_time,
            name="bin_durations",
            fallback=float(emissions.dt),
        )
        transition_durations = _reversed_transition_durations(emissions)
        return LogEmissionTensor(
            log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[::-1].copy(),
            spike_counts=np.asarray(emissions.spike_counts)[::-1].copy(),
            times=np.asarray(emissions.times, dtype=float)[::-1].copy(),
            dt=emissions.dt,
            cell_ids=np.asarray(emissions.cell_ids).copy(),
            n_spikes=int(emissions.n_spikes),
            bin_durations=bin_durations,
            transition_durations=transition_durations,
            metadata=dict(getattr(emissions, "metadata", {}) or {}),
        )

    setattr(reverse_emissions, _REVERSE_PATCHED_FLAG, True)
    setattr(reverse_emissions, _REVERSE_ORIGINAL_ATTR, current)
    accuracy_upgrades.reverse_emissions = reverse_emissions


def _reversed_transition_durations(emissions: LogEmissionTensor) -> np.ndarray:
    expected_length = max(emissions.n_time - 1, 0)
    values = getattr(emissions, "transition_durations", None)
    if values is not None:
        return _reversed_duration_vector(
            values,
            expected_length=expected_length,
            name="transition_durations",
        )

    times = np.asarray(getattr(emissions, "times", ()), dtype=float)
    if times.shape == (emissions.n_time,) and emissions.n_time > 1:
        durations = np.diff(times)
        if np.all(np.isfinite(durations)) and np.all(durations > 0.0):
            return durations[::-1].copy()

    return _reversed_duration_vector(
        None,
        expected_length=expected_length,
        name="transition_durations",
        fallback=float(emissions.dt),
    )


def _reversed_duration_vector(
    values: object,
    *,
    expected_length: int,
    name: str,
    fallback: float | None = None,
) -> np.ndarray:
    if values is None:
        if fallback is None:
            return np.empty(int(expected_length), dtype=float)
        return np.full(int(expected_length), float(fallback), dtype=float)

    array = np.asarray(values, dtype=float)
    if array.shape != (int(expected_length),):
        raise ValueError(
            f"{name} must contain {int(expected_length)} values; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive durations")
    return array[::-1].copy()


__all__ = ["apply_model_probability_status_patch"]

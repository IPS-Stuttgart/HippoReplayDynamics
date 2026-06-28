"""Conservative candidate-support quality labels for non-comparable rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_MISSING_SUPPORT_VALUES = {"", "nan", "none", "null", "<na>"}
_NONCOMPARABLE_SUPPORT_VALUES = {
    "degenerate_single_bin",
    "not_scored",
    "unknown_noncomparable",
    "particle_approximation",
}
_TRUNCATED_SUPPORT = "truncated_full_grid"
_SUCCESS_STATUS_VALUES = {"", "success", "nan", "none", "null", "<na>"}
_MIN_LOG_MASS_BOOL_PATCHED_FLAG = "_candidate_min_log_mass_bool_patch_applied"


def apply_candidate_support_quality_patch() -> None:
    """Keep failed/non-comparable rows out of good candidate-support counts."""

    from . import result_improvements as ri

    if not getattr(ri, "_candidate_support_quality_status_patch_applied", False):

        def candidate_support_quality(
            row: pd.Series,
            *,
            min_log_mass: float | None = None,
            good_threshold: float = ri.DEFAULT_GOOD_LOG_MASS_THRESHOLD,
            warning_threshold: float = ri.DEFAULT_WARNING_LOG_MASS_THRESHOLD,
        ) -> str:
            """Return a conservative quality label for one score row.

            Candidate-support quality is meaningful only for successful exact rows or
            candidate-pruned lower-bound rows.  Failed rows and non-comparable
            evidence supports should not be counted as ``exact_or_not_pruned`` merely
            because they are not truncated lower bounds.
            """

            status = _text(row.get("status", "success")).lower()
            if status not in _SUCCESS_STATUS_VALUES:
                return ri.CANDIDATE_SUPPORT_UNKNOWN

            support_values = _evidence_support_values(row)
            if any(value in _NONCOMPARABLE_SUPPORT_VALUES for value in support_values):
                return ri.CANDIDATE_SUPPORT_UNKNOWN
            if _TRUNCATED_SUPPORT not in support_values:
                return ri.CANDIDATE_SUPPORT_EXACT
            mass = _finite_candidate_log_mass(min_log_mass)
            if mass is None:
                return ri.CANDIDATE_SUPPORT_UNKNOWN
            if mass >= good_threshold:
                return ri.CANDIDATE_SUPPORT_GOOD
            if mass >= warning_threshold:
                return ri.CANDIDATE_SUPPORT_WARNING
            return ri.CANDIDATE_SUPPORT_POOR

        ri.candidate_support_quality = candidate_support_quality
        ri._candidate_support_quality_status_patch_applied = True

    _patch_boolean_candidate_log_mass(ri)


def _patch_boolean_candidate_log_mass(ri: Any) -> None:
    """Avoid interpreting boolean diagnostics as finite retained log mass."""

    if getattr(ri, _MIN_LOG_MASS_BOOL_PATCHED_FLAG, False):
        return
    original_first_finite_numeric_value = ri._first_finite_numeric_value

    def _first_finite_numeric_value(value: object) -> float | None:
        if _contains_boolean(value):
            return None
        return original_first_finite_numeric_value(value)

    ri._first_finite_numeric_value = _first_finite_numeric_value
    setattr(ri, _MIN_LOG_MASS_BOOL_PATCHED_FLAG, True)


def _evidence_support_values(row: pd.Series) -> list[str]:
    """Return normalized support labels from canonical and diagnostic columns."""

    values: list[str] = []
    values.extend(_support_value_labels(row.get("evidence_support", "")))
    for column in getattr(row, "index", ()):  # pandas Series in production; duck-typed in tests.
        name = str(column)
        if not name.startswith("diagnostic_") or not name.endswith("_evidence_support"):
            continue
        values.extend(_support_value_labels(row.get(column, "")))
    return list(dict.fromkeys(values))


def _support_value_labels(value: object) -> list[str]:
    """Return normalized non-missing support labels from scalar or array-like cells."""

    labels: list[str] = []
    for item in _flatten_support_value(value):
        text = _text(item).lower()
        if text and text not in _MISSING_SUPPORT_VALUES:
            labels.append(text)
    return labels


def _flatten_support_value(value: object) -> list[object]:
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return [value]
    if array.ndim == 0:
        try:
            return [array.item()]
        except ValueError:
            return []
    if array.size == 0:
        return []
    return list(array.reshape(-1))


def _finite_candidate_log_mass(value: object) -> float | None:
    if value is None or _contains_boolean(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _contains_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    if array.ndim == 0:
        try:
            return isinstance(array.item(), (bool, np.bool_))
        except ValueError:
            return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.reshape(-1))


def _text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


__all__ = ["apply_candidate_support_quality_patch"]
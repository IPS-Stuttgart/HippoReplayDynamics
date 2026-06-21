"""Conservative candidate-support quality labels for non-comparable rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_MISSING_SUPPORT_VALUES = {"", "nan", "none", "null", "<na>"}
_NONCOMPARABLE_SUPPORT_VALUES = {
    "not_scored",
    "unknown_noncomparable",
    "particle_approximation",
}
_TRUNCATED_SUPPORT = "truncated_full_grid"
_SUCCESS_STATUS_VALUES = {"", "success", "nan", "none", "null", "<na>"}


def apply_candidate_support_quality_patch() -> None:
    """Keep failed/non-comparable rows out of good candidate-support counts."""

    from . import result_improvements as ri

    if getattr(ri, "_candidate_support_quality_status_patch_applied", False):
        return

    def candidate_support_quality(
        row: pd.Series,
        *,
        min_log_mass: float | None = None,
        good_threshold: float = ri.DEFAULT_GOOD_LOG_MASS_THRESHOLD,
        warning_threshold: float = ri.DEFAULT_WARNING_LOG_MASS_THRESHOLD,
    ) -> str:
        """Return a conservative quality label for one score row.

        Candidate-support quality is meaningful only for successful exact rows or
        candidate-pruned lower-bound rows.  Failed rows and explicitly
        non-comparable evidence supports should not be counted as
        ``exact_or_not_pruned`` merely because they are not truncated lower
        bounds.
        """

        status = _text(row.get("status", "success")).lower()
        if status not in _SUCCESS_STATUS_VALUES:
            return ri.CANDIDATE_SUPPORT_UNKNOWN

        evidence_support = _text(row.get("evidence_support", "")).lower()
        if evidence_support in _NONCOMPARABLE_SUPPORT_VALUES:
            return ri.CANDIDATE_SUPPORT_UNKNOWN
        if evidence_support and evidence_support not in {
            _TRUNCATED_SUPPORT,
            *_MISSING_SUPPORT_VALUES,
        }:
            return ri.CANDIDATE_SUPPORT_EXACT

        diagnostic_values = _diagnostic_evidence_support_values(row)
        if any(value in _NONCOMPARABLE_SUPPORT_VALUES for value in diagnostic_values):
            return ri.CANDIDATE_SUPPORT_UNKNOWN
        if _TRUNCATED_SUPPORT not in diagnostic_values and evidence_support != _TRUNCATED_SUPPORT:
            return ri.CANDIDATE_SUPPORT_EXACT
        if min_log_mass is None or not np.isfinite(min_log_mass):
            return ri.CANDIDATE_SUPPORT_UNKNOWN
        if min_log_mass >= good_threshold:
            return ri.CANDIDATE_SUPPORT_GOOD
        if min_log_mass >= warning_threshold:
            return ri.CANDIDATE_SUPPORT_WARNING
        return ri.CANDIDATE_SUPPORT_POOR

    ri.candidate_support_quality = candidate_support_quality
    ri._candidate_support_quality_status_patch_applied = True


def _diagnostic_evidence_support_values(row: pd.Series) -> list[str]:
    """Return all non-empty diagnostic evidence-support labels on a score row."""

    values: list[str] = []
    for column in row.index:
        column_name = str(column)
        if not (
            column_name.startswith("diagnostic_")
            and column_name.endswith("_evidence_support")
        ):
            continue
        value = _text(row.get(column, "")).lower()
        if value and value not in _MISSING_SUPPORT_VALUES:
            values.append(value)
    return values


def _text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()

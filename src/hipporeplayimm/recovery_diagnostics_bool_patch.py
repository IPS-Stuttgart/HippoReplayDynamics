"""Use evidence-reporting boolean parsing in recovery diagnostics.

Recovery diagnostic tables are often rebuilt from CSV score artifacts.  Pandas can
round-trip boolean columns as strings such as ``"1.0"`` and ``"0.0"``.  The
shared evidence-reporting parser already handles those values; this patch makes
the diagnostic scalar helpers use the same semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .evidence_reporting import _coerce_bool_series

_PATCHED_FLAG = "_recovery_diagnostics_bool_scalar_patch_applied"
_COERCE_BOOL_WRAPPER_FLAG = "_recovery_diagnostics_bool_coerce_bool_wrapper"
_ROW_BOOL_WRAPPER_FLAG = "_recovery_diagnostics_bool_row_bool_wrapper"
_COERCE_FLOAT_WRAPPER_FLAG = "_recovery_diagnostics_bool_coerce_float_wrapper"
_ROW_FLOAT_WRAPPER_FLAG = "_recovery_diagnostics_bool_row_float_wrapper"
_SUCCESSFUL_FINITE_SCORES_WRAPPER_FLAG = "_recovery_diagnostics_bool_successful_finite_scores_wrapper"
_HELPER_FLAGS = {
    "_coerce_bool": _COERCE_BOOL_WRAPPER_FLAG,
    "_row_bool": _ROW_BOOL_WRAPPER_FLAG,
    "_coerce_float": _COERCE_FLOAT_WRAPPER_FLAG,
    "_row_float": _ROW_FLOAT_WRAPPER_FLAG,
    "_successful_finite_scores": _SUCCESSFUL_FINITE_SCORES_WRAPPER_FLAG,
}
_TRUE_FLOAT_STRINGS = {"true", "yes", "y"}
_FALSE_FLOAT_STRINGS = {"false", "no", "n"}
_MISSING_STATUS_VALUES = {"", "nan", "na", "n/a", "none", "null", "<na>"}


def apply_recovery_diagnostics_bool_patch() -> None:
    """Install shared scalar bool coercion for recovery diagnostics."""

    from . import recovery_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False) and _helpers_are_patched(diagnostics):
        return

    def coerce_bool(value: object, default: bool = False) -> bool:
        try:
            if pd.isna(value):
                return bool(default)
        except (TypeError, ValueError):
            return bool(default)
        return bool(_coerce_bool_series(pd.Series([value]), default=bool(default)).iloc[0])

    def row_bool(row: Any, column: str, default: bool) -> bool:
        if column not in row.index:
            return bool(default)
        return coerce_bool(row[column], default)

    def coerce_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in _TRUE_FLOAT_STRINGS:
                return 1.0
            if text in _FALSE_FLOAT_STRINGS:
                return 0.0
            return float(default)

    def row_float(row: Any, column: str, default: float) -> float:
        if column not in row.index:
            return float(default)
        try:
            if pd.isna(row[column]):
                return float(default)
        except (TypeError, ValueError):
            return float(default)
        return coerce_float(row[column], default)

    def successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
        if "status" in group:
            status_ok = group["status"].map(_status_is_success_or_missing).astype(bool)
        else:
            status_ok = pd.Series(True, index=group.index)
        values = pd.to_numeric(group["log_evidence"], errors="coerce") if "log_evidence" in group else pd.Series(0.0, index=group.index)
        finite = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=group.index)
        return group[status_ok & finite].copy()

    _install_helper(diagnostics, "_coerce_bool", coerce_bool)
    _install_helper(diagnostics, "_row_bool", row_bool)
    _install_helper(diagnostics, "_coerce_float", coerce_float)
    _install_helper(diagnostics, "_row_float", row_float)
    _install_helper(diagnostics, "_successful_finite_scores", successful_finite_scores)
    setattr(diagnostics, _PATCHED_FLAG, True)


def _helpers_are_patched(diagnostics: Any) -> bool:
    """Return whether all recovery-diagnostic scalar helpers are active wrappers."""

    return all(
        getattr(getattr(diagnostics, helper_name, None), wrapper_flag, False)
        for helper_name, wrapper_flag in _HELPER_FLAGS.items()
    )


def _install_helper(diagnostics: Any, helper_name: str, helper: Any) -> None:
    setattr(helper, _HELPER_FLAGS[helper_name], True)
    setattr(diagnostics, helper_name, helper)


def _status_is_success_or_missing(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return True
    text = str(value).strip().lower()
    return text == "success" or text in _MISSING_STATUS_VALUES


__all__ = ["apply_recovery_diagnostics_bool_patch"]
"""Reject complex-valued evidence before score-table ranking."""

from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_evidence_complex_validation_patch_applied"
_REPORTING_FINITE_FLAG = "_evidence_complex_reporting_finite"
_REPORTING_COERCE_FLAG = "_evidence_complex_reporting_coerce"
_STATUS_FINITE_FLAG = "_evidence_complex_status_finite"
_RECOVERY_FINITE_FLAG = "_evidence_complex_recovery_finite"
_NONSCALAR = object()


def _unwrap_scalar(value: object) -> object:
    """Unwrap NumPy scalar containers without coercing complex values to real."""

    seen: set[int] = set()
    while isinstance(value, (np.ndarray, np.generic)):
        if isinstance(value, np.complexfloating):
            return value
        if isinstance(value, np.ndarray):
            if value.size != 1 or id(value) in seen:
                return _NONSCALAR
            seen.add(id(value))
            value = value.reshape(-1)[0]
        else:
            item = value.item()
            # Extended-precision NumPy scalars may return another scalar of the
            # same type from item(), which otherwise makes this loop permanent.
            if isinstance(item, np.generic) and type(item) is type(value):
                return item
            value = item
    return value


def _real_numeric_value(value: object) -> float:
    """Return a real numeric scalar, or NaN for malformed/complex values."""

    value = _unwrap_scalar(value)
    if value is _NONSCALAR or isinstance(
        value,
        (bool, np.bool_, complex, np.complexfloating),
    ):
        return float("nan")
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return float("nan")


def _real_numeric_series(values: object) -> pd.Series:
    """Coerce scalar numeric values while rejecting every complex scalar."""

    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    return series.map(_real_numeric_value).astype(float)


def _finite_evidence_series(frame: pd.DataFrame) -> pd.Series:
    columns = [
        column
        for column in ("log_evidence", "heldout_log_likelihood")
        if column in frame
    ]
    if not columns:
        return pd.Series(True, index=frame.index)

    observed = pd.Series(False, index=frame.index)
    valid = pd.Series(True, index=frame.index)
    for column in columns:
        values = frame[column]
        missing = values.map(_is_missing_real_scalar).astype(bool)
        numeric = _real_numeric_series(values)
        finite = pd.Series(
            np.isfinite(numeric.to_numpy()),
            index=frame.index,
        )
        observed |= ~missing
        valid &= missing | finite
    return observed & valid


def _is_missing_real_scalar(value: object) -> bool:
    """Treat genuine missing scalars as absent without hiding complex values."""

    value = _unwrap_scalar(value)
    if value is _NONSCALAR or isinstance(value, (complex, np.complexfloating)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _coerce_log_evidence_column(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "log_evidence" in out:
        out["log_evidence"] = _real_numeric_series(out["log_evidence"])
    return out


def apply_evidence_complex_validation_patch() -> None:
    """Install complex-evidence rejection across reporting and recovery helpers."""

    from . import evidence_reporting as reporting
    from . import evidence_status_coercion as status

    reporting_current = bool(
        getattr(reporting._finite_evidence_series, _REPORTING_FINITE_FLAG, False)
        and getattr(reporting._coerce_log_evidence_column, _REPORTING_COERCE_FLAG, False)
    )
    status_current = bool(
        getattr(status._finite_log_evidence_mask, _STATUS_FINITE_FLAG, False)
    )

    setattr(_finite_evidence_series, _REPORTING_FINITE_FLAG, True)
    setattr(_coerce_log_evidence_column, _REPORTING_COERCE_FLAG, True)
    reporting._finite_evidence_series = _finite_evidence_series
    reporting._finite_log_evidence_series = _finite_evidence_series
    reporting._coerce_log_evidence_column = _coerce_log_evidence_column

    def finite_log_evidence_mask(frame: pd.DataFrame) -> pd.Series:
        if "log_evidence" not in frame:
            return pd.Series(True, index=frame.index)
        values = _real_numeric_series(frame["log_evidence"])
        return pd.Series(np.isfinite(values.to_numpy()), index=frame.index)

    setattr(finite_log_evidence_mask, _STATUS_FINITE_FLAG, True)
    status._finite_log_evidence_mask = finite_log_evidence_mask

    try:
        from . import recovery_diagnostics as diagnostics
    except ImportError:
        diagnostics = None

    recovery_current = True
    if diagnostics is not None:
        current = diagnostics._successful_finite_scores
        recovery_current = bool(getattr(current, _RECOVERY_FINITE_FLAG, False))
        if not recovery_current:

            @wraps(current)
            def successful_finite_scores(group: pd.DataFrame) -> pd.DataFrame:
                sanitized = group.copy()
                if "log_evidence" in sanitized:
                    sanitized["log_evidence"] = _real_numeric_series(
                        sanitized["log_evidence"]
                    )
                return current(sanitized)

            setattr(successful_finite_scores, _RECOVERY_FINITE_FLAG, True)
            diagnostics._successful_finite_scores = successful_finite_scores

    if reporting_current and status_current and recovery_current:
        setattr(reporting, _PATCHED_FLAG, True)
        return
    setattr(reporting, _PATCHED_FLAG, True)


__all__ = ["apply_evidence_complex_validation_patch"]

"""Keep valid negative-infinite log evidence through runtime validation patches.

``-inf`` is a legitimate log-evidence value: it represents zero probability.
It must remain rankable below every finite evidence value.  NaN, ``+inf``,
complex values, malformed scalars, and rows without any observed evidence remain
invalid.

The package installs several compatibility patches at runtime.  In particular,
``evidence_complex_validation`` runs after the simulation-recovery patches and
historically reinstalled ``np.isfinite``-based filters, which accidentally
removed ``-inf`` again.  This module makes the intended validity rule explicit
and keeps it stable across repeated ``apply_runtime_patches()`` calls.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

_PATCHED_FLAG = "_negative_infinite_evidence_validity_patch_applied"
_COMPLEX_APPLY_WRAPPER_FLAG = "_negative_infinite_complex_apply_wrapper"
_COMPLEX_REPORTING_FINITE_FLAG = "_evidence_complex_reporting_finite"
_RESULT_QUALITY_FINITE_FLAG = "_result_quality_row_specific_evidence_finiteness_patch_applied"
_COMPLEX_STATUS_FINITE_FLAG = "_evidence_complex_status_finite"
_EVIDENCE_COLUMNS = ("log_evidence", "heldout_log_likelihood")


def _usable_numeric_mask(values: np.ndarray) -> np.ndarray:
    """Accept finite values and ``-inf`` while rejecting NaN and ``+inf``."""

    numeric = np.asarray(values, dtype=float)
    return ~(np.isnan(numeric) | np.isposinf(numeric))


def _usable_evidence_series(frame: pd.DataFrame) -> pd.Series:
    """Return rows with at least one observed, valid real evidence value.

    Missing sibling score columns are ignored row-wise, matching the existing
    result-quality compatibility behavior.  Scalar coercion is delegated to the
    complex-validation helpers so booleans, complex values, non-scalars, and
    malformed objects stay rejected.
    """

    from . import evidence_complex_validation as complex_validation

    columns = [column for column in _EVIDENCE_COLUMNS if column in frame.columns]
    if not columns:
        return pd.Series(True, index=frame.index)

    observed = pd.Series(False, index=frame.index)
    valid = pd.Series(True, index=frame.index)
    for column in columns:
        values = frame[column]
        missing = values.map(complex_validation._is_missing_real_scalar).astype(bool)
        numeric = complex_validation._real_numeric_series(values)
        usable = pd.Series(
            _usable_numeric_mask(numeric.to_numpy(dtype=float)),
            index=frame.index,
        )
        observed |= ~missing
        valid &= missing | usable
    return observed & valid


def _usable_log_evidence_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a row mask for usable scalar real ``log_evidence`` values."""

    from . import evidence_complex_validation as complex_validation

    if "log_evidence" not in frame.columns:
        return pd.Series(True, index=frame.index)
    numeric = complex_validation._real_numeric_series(frame["log_evidence"])
    return pd.Series(
        _usable_numeric_mask(numeric.to_numpy(dtype=float)),
        index=frame.index,
    )


def _install_reporting_validity(reporting: Any, status: Any) -> None:
    """Install ``-inf``-aware evidence validity into live reporting helpers."""

    for flag in (_PATCHED_FLAG, _COMPLEX_REPORTING_FINITE_FLAG, _RESULT_QUALITY_FINITE_FLAG):
        setattr(_usable_evidence_series, flag, True)
    setattr(_usable_log_evidence_mask, _COMPLEX_STATUS_FINITE_FLAG, True)

    reporting._finite_evidence_series = _usable_evidence_series
    reporting._finite_log_evidence_series = _usable_evidence_series
    status._finite_log_evidence_mask = _usable_log_evidence_mask


def apply_negative_infinite_evidence_validity_patch() -> None:
    """Preserve the ``-inf`` validity rule across later runtime patch layers."""

    from . import evidence_complex_validation as complex_validation
    from . import evidence_reporting as reporting
    from . import evidence_status_coercion as status

    # evidence_complex_validation is applied later in the package patch order.
    # Replace the helper it will install, then wrap its public patch hook so the
    # status mask is repaired as well after every later/repeated invocation.
    complex_validation._finite_evidence_series = _usable_evidence_series
    _install_reporting_validity(reporting, status)

    current = complex_validation.apply_evidence_complex_validation_patch
    if getattr(current, _COMPLEX_APPLY_WRAPPER_FLAG, False):
        return

    @wraps(current)
    def apply_complex_validation_preserving_negative_infinity() -> None:
        current()
        complex_validation._finite_evidence_series = _usable_evidence_series
        _install_reporting_validity(reporting, status)

    setattr(
        apply_complex_validation_preserving_negative_infinity,
        _COMPLEX_APPLY_WRAPPER_FLAG,
        True,
    )
    complex_validation.apply_evidence_complex_validation_patch = (
        apply_complex_validation_preserving_negative_infinity
    )


__all__ = ["apply_negative_infinite_evidence_validity_patch"]

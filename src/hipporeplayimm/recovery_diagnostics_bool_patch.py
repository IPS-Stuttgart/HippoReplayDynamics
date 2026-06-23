"""Use evidence-reporting boolean parsing in recovery diagnostics.

Recovery diagnostic tables are often rebuilt from CSV score artifacts.  Pandas can
round-trip boolean columns as strings such as ``"1.0"`` and ``"0.0"``.  The
shared evidence-reporting parser already handles those values; this patch makes
the diagnostic scalar helpers use the same semantics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .evidence_reporting import _coerce_bool_series

_PATCHED_FLAG = "_recovery_diagnostics_bool_scalar_patch_applied"


def apply_recovery_diagnostics_bool_patch() -> None:
    """Install shared scalar bool coercion for recovery diagnostics."""

    from . import recovery_diagnostics as diagnostics

    if getattr(diagnostics, _PATCHED_FLAG, False):
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

    diagnostics._coerce_bool = coerce_bool
    diagnostics._row_bool = row_bool
    setattr(diagnostics, _PATCHED_FLAG, True)


__all__ = ["apply_recovery_diagnostics_bool_patch"]

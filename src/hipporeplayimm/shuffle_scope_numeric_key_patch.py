"""Normalize numeric shuffle p-value scope keys."""

from __future__ import annotations

import numpy as np

_PATCHED_FLAG = "_shuffle_scope_numeric_key_patch_applied"


def apply_shuffle_scope_numeric_key_patch() -> None:
    """Patch shuffle-control scope labels so int/float CSV dtypes match."""

    from . import shuffle_controls

    if getattr(shuffle_controls, _PATCHED_FLAG, False):
        return
    original_scope_label = shuffle_controls._scope_label

    def scope_label(value: object) -> str:
        numeric = _numeric_scope_label(value)
        if numeric is not None:
            return repr(("numeric", numeric))
        return original_scope_label(value)

    shuffle_controls._scope_label = scope_label
    setattr(shuffle_controls, _PATCHED_FLAG, True)


def _numeric_scope_label(value: object) -> str | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return format(numeric, ".17g")


__all__ = ["apply_shuffle_scope_numeric_key_patch"]

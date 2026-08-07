"""Validate sparse-momentum valid-bin masks with shared state-space semantics."""

from __future__ import annotations

from functools import wraps
import sys
from typing import Any

import numpy as np

from .state_space_utils import _coerce_valid_bin_mask as _coerce_shared_valid_bin_mask

_PATCHED_ATTR = "_sparse_momentum_valid_bin_mask_validation_patch_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _synchronize_imported_aliases(stale: object, active: object) -> None:
    """Refresh package modules that imported the sparse helper before patching."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, "_coerce_valid_bin_mask", None) is stale:
            setattr(module, "_coerce_valid_bin_mask", active)


def apply_sparse_momentum_valid_bin_mask_validation_patch() -> None:
    """Patch sparse momentum to reject non-boolean/non-binary valid-bin masks."""

    from . import state_space_sparse_momentum as sparse_momentum

    current = sparse_momentum._coerce_valid_bin_mask
    if getattr(current, _PATCHED_ATTR, False):
        original = getattr(current, _ORIGINAL_ATTR, getattr(current, "__wrapped__", None))
        if original is not None:
            _synchronize_imported_aliases(original, current)
        return

    @wraps(current)
    def coerce_valid_bin_mask(mask: Any, n_bins: int) -> np.ndarray | None:
        return _coerce_shared_valid_bin_mask(mask, n_bins)

    setattr(coerce_valid_bin_mask, _PATCHED_ATTR, True)
    setattr(coerce_valid_bin_mask, _ORIGINAL_ATTR, current)
    sparse_momentum._coerce_valid_bin_mask = coerce_valid_bin_mask
    _synchronize_imported_aliases(current, coerce_valid_bin_mask)


__all__ = ["apply_sparse_momentum_valid_bin_mask_validation_patch"]

"""Validate sparse-momentum valid-bin masks with shared state-space semantics."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from .state_space_utils import _coerce_valid_bin_mask as _coerce_shared_valid_bin_mask

_PATCHED_ATTR = "_sparse_momentum_valid_bin_mask_validation_patch_applied"


def apply_sparse_momentum_valid_bin_mask_validation_patch() -> None:
    """Patch sparse momentum to reject non-boolean/non-binary valid-bin masks."""

    from . import state_space_sparse_momentum as sparse_momentum

    current = sparse_momentum._coerce_valid_bin_mask
    if getattr(current, _PATCHED_ATTR, False):
        return

    @wraps(current)
    def coerce_valid_bin_mask(mask: Any, n_bins: int) -> np.ndarray | None:
        return _coerce_shared_valid_bin_mask(mask, n_bins)

    setattr(coerce_valid_bin_mask, _PATCHED_ATTR, True)
    sparse_momentum._coerce_valid_bin_mask = coerce_valid_bin_mask


__all__ = ["apply_sparse_momentum_valid_bin_mask_validation_patch"]

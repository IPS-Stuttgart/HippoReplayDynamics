"""Validate sparse-momentum valid-bin masks with shared state-space semantics."""

from __future__ import annotations

import sys


def _synchronize_imported_aliases(active: object) -> None:
    """Point every package alias at the canonical shared mask validator."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if hasattr(module, "_coerce_valid_bin_mask"):
            setattr(module, "_coerce_valid_bin_mask", active)


def apply_sparse_momentum_valid_bin_mask_validation_patch() -> None:
    """Patch sparse momentum to reject non-boolean/non-binary valid-bin masks."""

    from . import state_space_sparse_momentum as sparse_momentum
    from . import state_space_utils

    # Reuse the shared validator directly. Wrapping an imported-by-value alias
    # creates a second identity on every module reload and can also retain an
    # older permissive helper in its closure.
    active = state_space_utils._coerce_valid_bin_mask
    sparse_momentum._coerce_valid_bin_mask = active
    _synchronize_imported_aliases(active)


__all__ = ["apply_sparse_momentum_valid_bin_mask_validation_patch"]

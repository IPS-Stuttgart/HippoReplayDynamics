"""Make fixed-top-k candidate selection deterministic for tied emissions."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_PATCHED_ATTR = "_stable_candidate_tie_order_applied"
_ORIGINAL_ATTR = "__hipporeplayimm_original__"


def _stable_descending_top_k(log_emission: Any, top_k: int) -> np.ndarray:
    """Return top-k indices by score, breaking ties by the original bin index."""

    values = np.asarray(log_emission)
    indices = np.arange(values.shape[0], dtype=int)
    # lexsort uses the last key as the primary key.  Sorting by -score gives
    # descending likelihood while the index key keeps equal scores in their
    # original spatial-bin order.
    order = np.lexsort((indices, -values))
    return np.asarray(order[: int(top_k)], dtype=int)


def _patch_selector(
    module: ModuleType,
    *,
    validate_top_k: Callable[[str, object], None] | None = None,
) -> None:
    current = module._top_candidate_indices
    if getattr(current, _PATCHED_ATTR, False):
        return

    @wraps(current)
    def stable_top_candidate_indices(log_emission: Any, top_k: int) -> np.ndarray:
        if validate_top_k is not None:
            validate_top_k("top_k", top_k)
        values = np.asarray(log_emission)
        if top_k <= 0 or top_k >= values.shape[0]:
            return current(log_emission, top_k)
        return _stable_descending_top_k(values, int(top_k))

    setattr(stable_top_candidate_indices, _PATCHED_ATTR, True)
    setattr(stable_top_candidate_indices, _ORIGINAL_ATTR, current)
    module._top_candidate_indices = stable_top_candidate_indices


def apply_candidate_tie_order_patch() -> None:
    """Patch both legacy and state-space fixed-top-k candidate selectors."""

    from . import models, state_space_utils

    _patch_selector(models)
    _patch_selector(
        state_space_utils,
        validate_top_k=state_space_utils._reject_boolean_count,
    )


__all__ = ["apply_candidate_tie_order_patch"]

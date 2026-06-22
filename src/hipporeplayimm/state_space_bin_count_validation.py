"""Validation patch for shared state-space spatial support sizes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any


def _positive_bin_count(n_bins: int) -> int:
    try:
        count = int(n_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_bins must be positive") from exc
    if count <= 0:
        raise ValueError("n_bins must be positive")
    return count


def _mark_patched(wrapper: Callable[..., Any], original: Callable[..., Any]) -> Callable[..., Any]:
    setattr(wrapper, "__hipporeplayimm_original__", original)
    setattr(wrapper, "__hipporeplayimm_bin_count_validation_patch__", True)
    return wrapper


def apply_state_space_bin_count_validation_patch() -> None:
    """Reject empty state-space supports before uniform prior construction."""

    from . import state_space_utils as utils

    if getattr(utils._coerce_valid_bin_mask, "__hipporeplayimm_bin_count_validation_patch__", False):
        return

    original_coerce_valid_bin_mask = utils._coerce_valid_bin_mask
    original_uniform_log_prior = utils._uniform_log_prior
    original_uniform_probabilities = utils._uniform_probabilities
    original_valid_bin_count = utils._valid_bin_count

    def _coerce_valid_bin_mask(valid_bin_mask: Any, n_bins: int):
        return original_coerce_valid_bin_mask(valid_bin_mask, _positive_bin_count(n_bins))

    def _uniform_log_prior(n_bins: int, valid_bin_mask: Any = None):
        return original_uniform_log_prior(_positive_bin_count(n_bins), valid_bin_mask)

    def _uniform_probabilities(n_bins: int, valid_bin_mask: Any = None):
        return original_uniform_probabilities(_positive_bin_count(n_bins), valid_bin_mask)

    def _valid_bin_count(n_bins: int, valid_bin_mask: Any = None) -> int:
        return original_valid_bin_count(_positive_bin_count(n_bins), valid_bin_mask)

    patched = {
        "_coerce_valid_bin_mask": _mark_patched(
            _coerce_valid_bin_mask,
            original_coerce_valid_bin_mask,
        ),
        "_uniform_log_prior": _mark_patched(_uniform_log_prior, original_uniform_log_prior),
        "_uniform_probabilities": _mark_patched(
            _uniform_probabilities,
            original_uniform_probabilities,
        ),
        "_valid_bin_count": _mark_patched(_valid_bin_count, original_valid_bin_count),
    }
    originals = {
        "_coerce_valid_bin_mask": original_coerce_valid_bin_mask,
        "_uniform_log_prior": original_uniform_log_prior,
        "_uniform_probabilities": original_uniform_probabilities,
        "_valid_bin_count": original_valid_bin_count,
    }

    for name, func in patched.items():
        setattr(utils, name, func)

    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("hipporeplayimm"):
            continue
        for name, original in originals.items():
            if getattr(module, name, None) is original:
                setattr(module, name, patched[name])

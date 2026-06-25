"""Validation patch for shared state-space spatial support sizes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np


def _positive_bin_count(n_bins: int) -> int:
    if isinstance(n_bins, (bool, np.bool_)):
        raise ValueError("n_bins must be a positive integer")
    try:
        numeric = float(n_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_bins must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("n_bins must be a positive integer")
    count = int(round(numeric))
    if not np.isclose(numeric, count, rtol=0.0, atol=0.0):
        raise ValueError("n_bins must be a positive integer")
    return count


def _coerce_bool_mask(valid_bin_mask: Any, n_bins: int) -> np.ndarray | None:
    """Return a boolean mask without treating arbitrary numeric values as truthy."""

    if valid_bin_mask is None:
        return None
    raw = np.asarray(valid_bin_mask)
    if raw.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if np.issubdtype(raw.dtype, np.bool_):
        return raw.astype(bool, copy=False)
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("valid_bin_mask must contain boolean or 0/1 values") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("valid_bin_mask must contain finite boolean or 0/1 values")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("valid_bin_mask must contain boolean or 0/1 values")
    return numeric.astype(bool)


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
        count = _positive_bin_count(n_bins)
        return original_coerce_valid_bin_mask(_coerce_bool_mask(valid_bin_mask, count), count)

    def _uniform_log_prior(n_bins: int, valid_bin_mask: Any = None):
        count = _positive_bin_count(n_bins)
        return original_uniform_log_prior(count, _coerce_bool_mask(valid_bin_mask, count))

    def _uniform_probabilities(n_bins: int, valid_bin_mask: Any = None):
        count = _positive_bin_count(n_bins)
        return original_uniform_probabilities(count, _coerce_bool_mask(valid_bin_mask, count))

    def _valid_bin_count(n_bins: int, valid_bin_mask: Any = None) -> int:
        count = _positive_bin_count(n_bins)
        return original_valid_bin_count(count, _coerce_bool_mask(valid_bin_mask, count))

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

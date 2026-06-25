"""Validation patch for shared state-space spatial support sizes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np


def _integer_count(name: str, value: Any) -> int:
    """Return an integer-valued scalar count without silent truncation."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    raw = np.asarray(value)
    if raw.ndim != 0:
        raise TypeError(f"{name} must be an integer")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must be an integer")
    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite integer")
    count = int(round(numeric))
    if not np.isclose(numeric, count, rtol=0.0, atol=0.0):
        raise TypeError(f"{name} must be an integer")
    return count


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
    """Reject invalid state-space counts before support construction."""

    from . import state_space_utils as utils

    if getattr(utils._coerce_valid_bin_mask, "__hipporeplayimm_bin_count_validation_patch__", False):
        return

    original_coerce_valid_bin_mask = utils._coerce_valid_bin_mask
    original_uniform_log_prior = utils._uniform_log_prior
    original_uniform_probabilities = utils._uniform_probabilities
    original_valid_bin_count = utils._valid_bin_count
    original_mass_retaining_candidate_indices = utils._mass_retaining_candidate_indices

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

    def _mass_retaining_candidate_indices(
        log_emission: np.ndarray,
        mass_threshold: float | None = None,
        *,
        top_k: int | None = None,
        min_k: int = 1,
        max_k: int = 0,
    ) -> np.ndarray:
        top_k_count = None if top_k is None else _integer_count("top_k", top_k)
        if mass_threshold is None or float(mass_threshold) <= 0.0:
            return original_mass_retaining_candidate_indices(
                log_emission,
                mass_threshold,
                top_k=top_k_count,
                min_k=min_k,
                max_k=max_k,
            )
        return original_mass_retaining_candidate_indices(
            log_emission,
            mass_threshold,
            top_k=top_k_count,
            min_k=_integer_count("min_k", min_k),
            max_k=_integer_count("max_k", max_k),
        )

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
        "_mass_retaining_candidate_indices": _mark_patched(
            _mass_retaining_candidate_indices,
            original_mass_retaining_candidate_indices,
        ),
    }
    originals = {
        "_coerce_valid_bin_mask": original_coerce_valid_bin_mask,
        "_uniform_log_prior": original_uniform_log_prior,
        "_uniform_probabilities": original_uniform_probabilities,
        "_valid_bin_count": original_valid_bin_count,
        "_mass_retaining_candidate_indices": original_mass_retaining_candidate_indices,
    }

    for name, func in patched.items():
        setattr(utils, name, func)

    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("hipporeplayimm"):
            continue
        for name, original in originals.items():
            if getattr(module, name, None) is original:
                setattr(module, name, patched[name])

"""Validate state-space candidate count helper inputs.

Candidate support sizes are integer counts.  Python's loose numeric coercions can
otherwise silently truncate floats through ``int(...)`` in mass-retaining support
selection, while string scalars may be accepted by the same path.  Keep the guard
at the shared helper boundary and refresh already-imported public aliases so all
state-space candidate-selection paths enforce the same contract.
"""

from __future__ import annotations

import operator
import sys
from functools import wraps
from typing import Any

import numpy as np

_TOP_K_PATCHED_FLAG = "_state_space_top_candidate_count_validation_patch_applied"
_MASS_PATCHED_FLAG = "_state_space_mass_candidate_count_validation_patch_applied"
_STRING_SCALAR_TYPES = (str, bytes, np.str_, np.bytes_)


def _is_string_scalar(value: Any) -> bool:
    if isinstance(value, _STRING_SCALAR_TYPES):
        return True
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.ndim != 0:
        return False
    if np.issubdtype(raw.dtype, np.str_) or np.issubdtype(raw.dtype, np.bytes_):
        return True
    if raw.dtype == object:
        try:
            return isinstance(raw.item(), _STRING_SCALAR_TYPES)
        except ValueError:
            return False
    return False


def _coerce_count(name: str, value: Any) -> int:
    """Return an integer count without accepting bool, float, string, or arrays."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer count, not boolean")
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be an integer count, not string")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc
    if raw.ndim != 0:
        raise TypeError(f"{name} must be an integer scalar")
    if np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(f"{name} must be an integer count, not boolean")
    if raw.dtype == object:
        try:
            item = raw.item()
        except ValueError as exc:
            raise TypeError(f"{name} must be an integer scalar") from exc
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer count, not boolean")
        if isinstance(item, _STRING_SCALAR_TYPES):
            raise TypeError(f"{name} must be an integer count, not string")
    else:
        item = raw.item()
    try:
        return int(operator.index(item))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer scalar") from exc


def _patch_top_candidate_indices() -> None:
    from . import state_space_utils

    current = state_space_utils._top_candidate_indices
    if getattr(current, _TOP_K_PATCHED_FLAG, False):
        setattr(state_space_utils, _TOP_K_PATCHED_FLAG, True)
        return

    @wraps(current)
    def top_candidate_indices(log_emission, top_k):
        return current(log_emission, _coerce_count("top_k", top_k))

    setattr(top_candidate_indices, _TOP_K_PATCHED_FLAG, True)
    setattr(top_candidate_indices, "__hipporeplayimm_original__", current)
    state_space_utils._top_candidate_indices = top_candidate_indices
    setattr(state_space_utils, _TOP_K_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_top_candidate_indices", None) is current:
            module._top_candidate_indices = top_candidate_indices


def _patch_mass_retaining_candidate_indices() -> None:
    from . import state_space_utils

    current = state_space_utils._mass_retaining_candidate_indices
    if getattr(current, _MASS_PATCHED_FLAG, False):
        setattr(state_space_utils, _MASS_PATCHED_FLAG, True)
        return

    @wraps(current)
    def mass_retaining_candidate_indices(
        log_emission,
        mass_threshold=None,
        *,
        top_k=None,
        min_k=1,
        max_k=0,
    ):
        coerced_top_k = None if top_k is None else _coerce_count("top_k", top_k)
        return current(
            log_emission,
            mass_threshold,
            top_k=coerced_top_k,
            min_k=_coerce_count("min_k", min_k),
            max_k=_coerce_count("max_k", max_k),
        )

    setattr(mass_retaining_candidate_indices, _MASS_PATCHED_FLAG, True)
    setattr(mass_retaining_candidate_indices, "__hipporeplayimm_original__", current)
    state_space_utils._mass_retaining_candidate_indices = mass_retaining_candidate_indices
    setattr(state_space_utils, _MASS_PATCHED_FLAG, True)

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if module_name.startswith("hipporeplayimm") and getattr(module, "_mass_retaining_candidate_indices", None) is current:
            module._mass_retaining_candidate_indices = mass_retaining_candidate_indices


def apply_candidate_count_validation_patch() -> None:
    """Install idempotent validation for candidate-count helper parameters."""

    _patch_top_candidate_indices()
    _patch_mass_retaining_candidate_indices()


__all__ = ["apply_candidate_count_validation_patch"]

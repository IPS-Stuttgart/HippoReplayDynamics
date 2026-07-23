"""Validate state-space candidate-support counts as integer scalars.

Candidate-count parameters are used both as NumPy partition indices and as
support-size bounds.  Letting float values reach those paths produces
inconsistent behavior: fixed ``top_k`` values fail inside NumPy, while other
counts are silently truncated by ``int``.  Validate the public helpers and the
model configuration at one boundary instead.
"""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

from .state_space_utils import _coerce_integer_count

_HELPER_PATCHED_FLAG = "_candidate_integer_count_validation_patch_applied"
_MODEL_PATCHED_FLAG = "_candidate_config_integer_count_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)
_CONFIG_COUNT_NAMES = (
    "momentum_candidate_top_k",
    "momentum_candidate_min_k",
    "momentum_candidate_max_k",
    "momentum_predicted_candidate_top_k",
)


def _is_string_scalar(value: object) -> bool:
    if isinstance(value, _STRING_TYPES):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.ndim != 0:
        return False
    if np.issubdtype(array.dtype, np.str_) or np.issubdtype(array.dtype, np.bytes_):
        return True
    if array.dtype == object:
        try:
            return isinstance(array.item(), _STRING_TYPES)
        except ValueError:
            return False
    return False


def _validated_integer_count(name: str, value: object) -> int:
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be an integer scalar, not string")
    return _coerce_integer_count(name, value)


def _wrapper_chain_has_marker(function: object, marker: str) -> bool:
    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        seen.add(current_id)
        if getattr(current, marker, False):
            return True
        current = getattr(current, "__hipporeplayimm_original__", None)
    return False


def _replace_imported_module_aliases(attribute_name: str, original: object, replacement: object) -> None:
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, attribute_name, None) is original:
            setattr(module, attribute_name, replacement)


def _validate_candidate_config(config: object) -> None:
    for name in _CONFIG_COUNT_NAMES:
        value = getattr(config, name, None)
        if value is not None:
            _validated_integer_count(name, value)


def apply_state_space_candidate_count_validation_patch() -> None:
    """Install strict integer validation for candidate-support counts."""

    from . import state_space_model
    from . import state_space_utils

    current_top_candidates = state_space_utils._top_candidate_indices
    if not _wrapper_chain_has_marker(current_top_candidates, _HELPER_PATCHED_FLAG):

        @wraps(current_top_candidates)
        def top_candidate_indices(log_emission, top_k: int):
            return current_top_candidates(
                log_emission,
                _validated_integer_count("top_k", top_k),
            )

        setattr(top_candidate_indices, _HELPER_PATCHED_FLAG, True)
        setattr(top_candidate_indices, "__hipporeplayimm_original__", current_top_candidates)
        state_space_utils._top_candidate_indices = top_candidate_indices
        _replace_imported_module_aliases(
            "_top_candidate_indices",
            current_top_candidates,
            top_candidate_indices,
        )

    current_mass_candidates = state_space_utils._mass_retaining_candidate_indices
    if not _wrapper_chain_has_marker(current_mass_candidates, _HELPER_PATCHED_FLAG):

        @wraps(current_mass_candidates)
        def mass_retaining_candidate_indices(
            log_emission,
            mass_threshold=None,
            *,
            top_k=None,
            min_k=1,
            max_k=0,
        ):
            validated_top_k = None if top_k is None else _validated_integer_count("top_k", top_k)
            return current_mass_candidates(
                log_emission,
                mass_threshold,
                top_k=validated_top_k,
                min_k=_validated_integer_count("min_k", min_k),
                max_k=_validated_integer_count("max_k", max_k),
            )

        setattr(mass_retaining_candidate_indices, _HELPER_PATCHED_FLAG, True)
        setattr(
            mass_retaining_candidate_indices,
            "__hipporeplayimm_original__",
            current_mass_candidates,
        )
        state_space_utils._mass_retaining_candidate_indices = mass_retaining_candidate_indices
        _replace_imported_module_aliases(
            "_mass_retaining_candidate_indices",
            current_mass_candidates,
            mass_retaining_candidate_indices,
        )

    current_candidate_indices = state_space_model.StateSpaceReplayModel.candidate_indices
    if _wrapper_chain_has_marker(current_candidate_indices, _MODEL_PATCHED_FLAG):
        return

    @wraps(current_candidate_indices)
    def candidate_indices(self, emissions, bin_centers=None, valid_bin_mask=None):
        config = getattr(self, "config", None)
        if config is not None:
            _validate_candidate_config(config)
        return current_candidate_indices(
            self,
            emissions,
            bin_centers=bin_centers,
            valid_bin_mask=valid_bin_mask,
        )

    setattr(candidate_indices, _MODEL_PATCHED_FLAG, True)
    setattr(candidate_indices, "__hipporeplayimm_original__", current_candidate_indices)
    state_space_model.StateSpaceReplayModel.candidate_indices = candidate_indices


__all__ = ["apply_state_space_candidate_count_validation_patch"]

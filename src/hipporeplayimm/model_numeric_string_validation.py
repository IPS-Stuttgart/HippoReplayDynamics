"""Reject string-valued replay-model numeric parameters."""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

_PATCHED_FLAG = "_model_numeric_string_validation_patch_applied"
_STATE_SPACE_UTILS_PATCHED_FLAG = "_state_space_numeric_string_validation_patch_applied"
_STATE_SPACE_MODEL_PATCHED_FLAG = "_state_space_model_numeric_string_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)
_VALIDATOR_NAMES = (
    "_validate_positive_parameter",
    "_validate_nonnegative_parameter",
    "_validate_probability_parameter",
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


def _reject_string_scalar(name: str, value: object) -> None:
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not string")


def _reject_string_count(name: str, value: object) -> None:
    if _is_string_scalar(value):
        raise TypeError(f"{name} must be an integer scalar, not string")


def _replace_imported_module_aliases(attribute_name: str, original: object, replacement: object) -> None:
    """Replace by-value imports of patched state-space helpers."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, attribute_name, None) is original:
            setattr(module, attribute_name, replacement)


def _validate_state_space_candidate_config(config: object) -> None:
    threshold = getattr(config, "momentum_candidate_mass_threshold", None)
    if threshold is not None:
        _reject_string_scalar("momentum_candidate_mass_threshold", threshold)
    for name in (
        "momentum_candidate_top_k",
        "momentum_candidate_min_k",
        "momentum_candidate_max_k",
        "momentum_predicted_candidate_top_k",
    ):
        value = getattr(config, name, None)
        if value is not None:
            _reject_string_count(name, value)


def _patch_state_space_numeric_string_validation() -> None:
    from . import state_space_model
    from . import state_space_utils

    if not getattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, False):
        original_integer_count = state_space_utils._coerce_integer_count
        original_unit_probability = state_space_utils._coerce_unit_probability
        original_top_candidates = state_space_utils._top_candidate_indices
        original_mass_candidates = state_space_utils._mass_retaining_candidate_indices

        @wraps(original_integer_count)
        def coerce_integer_count(name: str, value: object) -> int:
            _reject_string_count(name, value)
            return original_integer_count(name, value)

        @wraps(original_unit_probability)
        def coerce_unit_probability(name: str, value: object) -> float:
            _reject_string_scalar(name, value)
            return original_unit_probability(name, value)

        @wraps(original_top_candidates)
        def top_candidate_indices(log_emission, top_k: int):
            _reject_string_count("top_k", top_k)
            return original_top_candidates(log_emission, top_k)

        @wraps(original_mass_candidates)
        def mass_retaining_candidate_indices(
            log_emission,
            mass_threshold=None,
            *,
            top_k=None,
            min_k=1,
            max_k=0,
        ):
            if mass_threshold is not None:
                _reject_string_scalar("mass_threshold", mass_threshold)
            if top_k is not None:
                _reject_string_count("top_k", top_k)
            _reject_string_count("min_k", min_k)
            _reject_string_count("max_k", max_k)
            return original_mass_candidates(
                log_emission,
                mass_threshold,
                top_k=top_k,
                min_k=min_k,
                max_k=max_k,
            )

        setattr(coerce_integer_count, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        setattr(coerce_unit_probability, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        setattr(top_candidate_indices, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        setattr(mass_retaining_candidate_indices, _STATE_SPACE_UTILS_PATCHED_FLAG, True)
        setattr(coerce_integer_count, "__hipporeplayimm_original__", original_integer_count)
        setattr(coerce_unit_probability, "__hipporeplayimm_original__", original_unit_probability)
        setattr(top_candidate_indices, "__hipporeplayimm_original__", original_top_candidates)
        setattr(mass_retaining_candidate_indices, "__hipporeplayimm_original__", original_mass_candidates)

        state_space_utils._coerce_integer_count = coerce_integer_count
        state_space_utils._coerce_unit_probability = coerce_unit_probability
        state_space_utils._top_candidate_indices = top_candidate_indices
        state_space_utils._mass_retaining_candidate_indices = mass_retaining_candidate_indices
        _replace_imported_module_aliases("_coerce_integer_count", original_integer_count, coerce_integer_count)
        _replace_imported_module_aliases("_coerce_unit_probability", original_unit_probability, coerce_unit_probability)
        _replace_imported_module_aliases("_top_candidate_indices", original_top_candidates, top_candidate_indices)
        _replace_imported_module_aliases(
            "_mass_retaining_candidate_indices",
            original_mass_candidates,
            mass_retaining_candidate_indices,
        )
        setattr(state_space_utils, _STATE_SPACE_UTILS_PATCHED_FLAG, True)

    current_candidate_indices = state_space_model.StateSpaceReplayModel.candidate_indices
    if getattr(current_candidate_indices, _STATE_SPACE_MODEL_PATCHED_FLAG, False):
        return

    @wraps(current_candidate_indices)
    def candidate_indices(self, emissions, bin_centers=None, valid_bin_mask=None):
        config = getattr(self, "config", None)
        if config is not None:
            _validate_state_space_candidate_config(config)
        return current_candidate_indices(
            self,
            emissions,
            bin_centers=bin_centers,
            valid_bin_mask=valid_bin_mask,
        )

    setattr(candidate_indices, _STATE_SPACE_MODEL_PATCHED_FLAG, True)
    setattr(candidate_indices, "__hipporeplayimm_original__", current_candidate_indices)
    state_space_model.StateSpaceReplayModel.candidate_indices = candidate_indices


def apply_model_numeric_string_validation_patch() -> None:
    """Install string-scalar guards around model parameter validators."""

    from . import models

    if not getattr(models, _PATCHED_FLAG, False):
        for validator_name in _VALIDATOR_NAMES:
            current = getattr(models, validator_name)

            @wraps(current)
            def validator(name: str, value: object, *, _current=current):
                _reject_string_scalar(name, value)
                return _current(name, value)

            setattr(validator, _PATCHED_FLAG, True)
            setattr(validator, "__hipporeplayimm_original__", current)
            setattr(models, validator_name, validator)

        setattr(models, _PATCHED_FLAG, True)

    _patch_state_space_numeric_string_validation()


__all__ = ["apply_model_numeric_string_validation_patch"]

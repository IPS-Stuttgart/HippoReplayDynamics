"""Reject string-valued replay-model numeric parameters."""

from __future__ import annotations

import sys
from functools import wraps

import numpy as np

from .model_parameter_validation import _reject_boolean_scalar, _validate_unit_interval_parameter

_PATCHED_FLAG = "_model_numeric_string_validation_patch_applied"
_PATCH_VERSION_ATTR = "_model_numeric_string_validation_patch_version"
_PATCH_VERSION = 2
_STATE_SPACE_UTILS_PATCHED_FLAG = "_state_space_numeric_string_validation_patch_applied"
_STATE_SPACE_MODEL_PATCHED_FLAG = "_state_space_model_numeric_string_validation_patch_applied"
_TRAJECTORY_IMM_PATCHED_FLAG = "_trajectory_imm_numeric_string_validation_patch_applied"
_STRING_TYPES = (str, bytes, np.str_, np.bytes_)
_VALIDATOR_NAMES = (
    "_validate_positive_parameter",
    "_validate_nonnegative_parameter",
    "_validate_probability_parameter",
)
_STATE_SPACE_HELPER_NAMES = (
    "_coerce_integer_count",
    "_coerce_unit_probability",
    "_top_candidate_indices",
    "_mass_retaining_candidate_indices",
)
_TRAJECTORY_IMM_HELPER_NAMES = (
    "_trajectory_imm_mode_stickiness",
    "_trajectory_imm_mode_prior",
    "_trajectory_imm_mode_transition_matrix",
    "_trajectory_imm_mode_transition_matrices",
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


def _wrapper_chain_has_marker(function: object, marker: str, patch_version: int | None = None) -> bool:
    """Return True when a wrapper in ``function``'s original chain has ``marker``.

    When ``patch_version`` is supplied, only wrappers carrying the matching
    patch-version attribute count as current.  This lets runtime patch refresh
    stale wrappers that were marked by an earlier implementation but lack newer
    guards.
    """

    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        seen.add(current_id)
        if getattr(current, marker, False):
            if patch_version is None or getattr(current, _PATCH_VERSION_ATTR, None) == patch_version:
                return True
        current = getattr(current, "__hipporeplayimm_original__", None)
    return False


def _replace_imported_module_aliases(attribute_name: str, original: object, replacement: object) -> None:
    """Replace by-value imports of patched state-space helpers."""

    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if getattr(module, attribute_name, None) is original:
            setattr(module, attribute_name, replacement)


def _state_space_helpers_are_string_guarded(state_space_utils: object) -> bool:
    """Return True only when every helper still carries the string guard wrapper."""

    return all(
        bool(getattr(getattr(state_space_utils, helper_name, None), _STATE_SPACE_UTILS_PATCHED_FLAG, False))
        for helper_name in _STATE_SPACE_HELPER_NAMES
    )


def _trajectory_imm_helpers_are_string_guarded(state_space_trajectory_imm: object) -> bool:
    """Return True only when every trajectory-IMM helper still rejects string scalars."""

    return all(
        bool(getattr(getattr(state_space_trajectory_imm, helper_name, None), _TRAJECTORY_IMM_PATCHED_FLAG, False))
        for helper_name in _TRAJECTORY_IMM_HELPER_NAMES
    )


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

    if not _state_space_helpers_are_string_guarded(state_space_utils):
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


def _patch_trajectory_imm_numeric_string_validation() -> None:
    from . import state_space_trajectory_imm

    if _trajectory_imm_helpers_are_string_guarded(state_space_trajectory_imm):
        return

    original_mode_stickiness = state_space_trajectory_imm._trajectory_imm_mode_stickiness
    original_mode_prior = state_space_trajectory_imm._trajectory_imm_mode_prior
    original_mode_transition_matrix = state_space_trajectory_imm._trajectory_imm_mode_transition_matrix
    original_mode_transition_matrices = state_space_trajectory_imm._trajectory_imm_mode_transition_matrices

    @wraps(original_mode_stickiness)
    def trajectory_imm_mode_stickiness(config):
        explicit_value = getattr(config, "trajectory_imm_mode_stickiness", None)
        if explicit_value is None:
            _reject_string_scalar("imm_mode_stickiness", getattr(config, "imm_mode_stickiness", 0.95))
        else:
            _reject_string_scalar("trajectory_imm_mode_stickiness", explicit_value)
        return original_mode_stickiness(config)

    @wraps(original_mode_prior)
    def trajectory_imm_mode_prior(config):
        value = getattr(config, "trajectory_imm_momentum_initial_probability", None)
        if value is not None:
            _reject_string_scalar("trajectory_imm_momentum_initial_probability", value)
        return original_mode_prior(config)

    @wraps(original_mode_transition_matrix)
    def trajectory_imm_mode_transition_matrix(config, stickiness):
        _reject_string_scalar("trajectory_imm_mode_stickiness", stickiness)
        momentum_switch = getattr(config, "trajectory_imm_momentum_switch_probability", None)
        if momentum_switch is not None:
            _reject_string_scalar("trajectory_imm_momentum_switch_probability", momentum_switch)
        return original_mode_transition_matrix(config, stickiness)

    @wraps(original_mode_transition_matrices)
    def trajectory_imm_mode_transition_matrices(config, stickiness, durations):
        _reject_string_scalar("trajectory_imm_mode_stickiness", stickiness)
        _reject_string_scalar("imm_switch_tau_s", getattr(config, "imm_switch_tau_s", 0.0))
        momentum_switch = getattr(config, "trajectory_imm_momentum_switch_probability", None)
        if momentum_switch is not None:
            _reject_string_scalar("trajectory_imm_momentum_switch_probability", momentum_switch)
        return original_mode_transition_matrices(config, stickiness, durations)

    setattr(trajectory_imm_mode_stickiness, _TRAJECTORY_IMM_PATCHED_FLAG, True)
    setattr(trajectory_imm_mode_prior, _TRAJECTORY_IMM_PATCHED_FLAG, True)
    setattr(trajectory_imm_mode_transition_matrix, _TRAJECTORY_IMM_PATCHED_FLAG, True)
    setattr(trajectory_imm_mode_transition_matrices, _TRAJECTORY_IMM_PATCHED_FLAG, True)
    setattr(trajectory_imm_mode_stickiness, "__hipporeplayimm_original__", original_mode_stickiness)
    setattr(trajectory_imm_mode_prior, "__hipporeplayimm_original__", original_mode_prior)
    setattr(trajectory_imm_mode_transition_matrix, "__hipporeplayimm_original__", original_mode_transition_matrix)
    setattr(trajectory_imm_mode_transition_matrices, "__hipporeplayimm_original__", original_mode_transition_matrices)
    state_space_trajectory_imm._trajectory_imm_mode_stickiness = trajectory_imm_mode_stickiness
    state_space_trajectory_imm._trajectory_imm_mode_prior = trajectory_imm_mode_prior
    state_space_trajectory_imm._trajectory_imm_mode_transition_matrix = trajectory_imm_mode_transition_matrix
    state_space_trajectory_imm._trajectory_imm_mode_transition_matrices = trajectory_imm_mode_transition_matrices
    setattr(state_space_trajectory_imm, _TRAJECTORY_IMM_PATCHED_FLAG, True)


def apply_model_numeric_string_validation_patch() -> None:
    """Install string-scalar guards around model parameter validators."""

    from . import models

    for validator_name in _VALIDATOR_NAMES:
        current = getattr(models, validator_name)
        if _wrapper_chain_has_marker(current, _PATCHED_FLAG, _PATCH_VERSION):
            continue

        @wraps(current)
        def validator(name: str, value: object, *, _current=current, _validator_name=validator_name):
            _reject_string_scalar(name, value)
            _reject_boolean_scalar(name, value)
            if _validator_name == "_validate_nonnegative_parameter" and name == "velocity_decay":
                _validate_unit_interval_parameter(name, value)
                return None
            return _current(name, value)

        setattr(validator, _PATCHED_FLAG, True)
        setattr(validator, _PATCH_VERSION_ATTR, _PATCH_VERSION)
        setattr(validator, "__hipporeplayimm_original__", current)
        setattr(models, validator_name, validator)

    setattr(models, _PATCHED_FLAG, True)
    _patch_state_space_numeric_string_validation()
    _patch_trajectory_imm_numeric_string_validation()


__all__ = ["apply_model_numeric_string_validation_patch"]

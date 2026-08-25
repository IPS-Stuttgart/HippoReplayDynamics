"""Validate all state-space candidate-count configuration fields.

The shared candidate helpers validate counts that are active for the selected
support strategy. ``momentum_candidate_min_k`` and
``momentum_candidate_max_k`` are not routed through those helpers when fixed
``top_k`` support is used, yet they are still emitted in diagnostics via
``int(...)``. Validate every configured count before support construction or
scoring so fractional or negative inactive bounds cannot be silently truncated.
"""

from __future__ import annotations

from functools import wraps

from .state_space_bin_count_validation import _nonnegative_integer_count

_PATCHED_FLAG = "_candidate_config_count_validation_patch_applied"
_SCORE_PATCHED_FLAG = "_candidate_config_count_score_validation_patch_applied"
_MASS_BOUND_PATCHED_FLAG = "_candidate_config_mass_bound_validation_patch_applied"
_CONFIG_COUNT_NAMES = (
    "momentum_candidate_top_k",
    "momentum_candidate_min_k",
    "momentum_candidate_max_k",
    "momentum_predicted_candidate_top_k",
)


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


def _validate_candidate_config(config: object) -> None:
    for name in _CONFIG_COUNT_NAMES:
        _nonnegative_integer_count(name, getattr(config, name))


def _patch_mass_retaining_candidate_bound(state_space_model: object) -> None:
    """Make a positive configured ``max_k`` a true upper support bound."""

    current = state_space_model._mass_retaining_candidate_indices
    if _wrapper_chain_has_marker(current, _MASS_BOUND_PATCHED_FLAG):
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
        max_count = _nonnegative_integer_count("max_k", max_k)
        candidates = current(
            log_emission,
            mass_threshold,
            top_k=top_k,
            min_k=min_k,
            max_k=max_count,
        )
        if max_count > 0 and len(candidates) > max_count:
            raise ValueError(
                "max_k is smaller than the effective candidate lower bound; "
                "increase max_k or reduce top_k/min_k"
            )
        return candidates

    setattr(mass_retaining_candidate_indices, _MASS_BOUND_PATCHED_FLAG, True)
    setattr(mass_retaining_candidate_indices, "__hipporeplayimm_original__", current)
    state_space_model._mass_retaining_candidate_indices = mass_retaining_candidate_indices


def _patch_candidate_indices(state_space_model: object) -> None:
    current = state_space_model.StateSpaceReplayModel.candidate_indices
    if _wrapper_chain_has_marker(current, _PATCHED_FLAG):
        return

    @wraps(current)
    def candidate_indices(self, emissions, bin_centers=None, valid_bin_mask=None):
        config = getattr(self, "config", None)
        if config is not None:
            _validate_candidate_config(config)
        return current(
            self,
            emissions,
            bin_centers=bin_centers,
            valid_bin_mask=valid_bin_mask,
        )

    setattr(candidate_indices, _PATCHED_FLAG, True)
    setattr(candidate_indices, "__hipporeplayimm_original__", current)
    state_space_model.StateSpaceReplayModel.candidate_indices = candidate_indices


def _patch_score(state_space_model: object) -> None:
    current = state_space_model.StateSpaceReplayModel.score
    if _wrapper_chain_has_marker(current, _SCORE_PATCHED_FLAG):
        return

    @wraps(current)
    def score(
        self,
        emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory: bool = True,
    ):
        config = getattr(self, "config", None)
        if config is not None:
            _validate_candidate_config(config)
        return current(
            self,
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
            return_trajectory=return_trajectory,
        )

    setattr(score, _SCORE_PATCHED_FLAG, True)
    setattr(score, "__hipporeplayimm_original__", current)
    if getattr(current, "_native_duration_occupancy_aware", False):
        setattr(score, "_native_duration_occupancy_aware", True)
    state_space_model.StateSpaceReplayModel.score = score


def apply_state_space_candidate_count_validation_patch() -> None:
    """Install unconditional validation for candidate-count config fields."""

    from . import state_space_model
    from .state_space_candidate_bin_center_validation import (
        apply_state_space_candidate_bin_center_validation_patch,
    )

    _patch_mass_retaining_candidate_bound(state_space_model)
    _patch_candidate_indices(state_space_model)
    _patch_score(state_space_model)
    # ``state_space_model`` can be reloaded independently.  Its fresh class loses
    # the bin-center wrapper normally installed by importing ``state_space``;
    # restore it after the count wrapper so repeated runtime refreshes remain
    # idempotent and preserve the documented 1D-bin-center support.
    apply_state_space_candidate_bin_center_validation_patch()


__all__ = ["apply_state_space_candidate_count_validation_patch"]

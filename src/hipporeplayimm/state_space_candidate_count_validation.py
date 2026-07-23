"""Validate all state-space candidate-count configuration fields.

The shared candidate helpers validate counts that are active for the selected
support strategy.  ``momentum_candidate_min_k`` and
``momentum_candidate_max_k`` are not routed through those helpers when fixed
``top_k`` support is used, yet they are still emitted in diagnostics via
``int(...)``.  Validate every configured count before support construction so
fractional or negative inactive bounds cannot be silently truncated.
"""

from __future__ import annotations

from functools import wraps

from .state_space_bin_count_validation import _nonnegative_integer_count

_PATCHED_FLAG = "_candidate_config_count_validation_patch_applied"
_CONFIG_COUNT_NAMES = (
    "momentum_candidate_top_k",
    "momentum_candidate_min_k",
    "momentum_candidate_max_k",
    "momentum_predicted_candidate_top_k",
)


def _wrapper_chain_has_marker(function: object) -> bool:
    seen: set[int] = set()
    current = function
    while current is not None:
        current_id = id(current)
        if current_id in seen:
            return False
        seen.add(current_id)
        if getattr(current, _PATCHED_FLAG, False):
            return True
        current = getattr(current, "__hipporeplayimm_original__", None)
    return False


def _validate_candidate_config(config: object) -> None:
    for name in _CONFIG_COUNT_NAMES:
        _nonnegative_integer_count(name, getattr(config, name))


def apply_state_space_candidate_count_validation_patch() -> None:
    """Install unconditional validation for candidate-count config fields."""

    from . import state_space_model

    current = state_space_model.StateSpaceReplayModel.candidate_indices
    if _wrapper_chain_has_marker(current):
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


__all__ = ["apply_state_space_candidate_count_validation_patch"]

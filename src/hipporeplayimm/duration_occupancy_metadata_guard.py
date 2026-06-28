"""Runtime guard for occupancy-masked emission metadata isolation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from functools import wraps
import operator
from typing import Any, Callable

import numpy as np

_EVIDENCE_ONLY_DIAGNOSTIC_PATCH_ATTR = "_duration_occupancy_evidence_only_diagnostics_patch_applied"
_EVIDENCE_ONLY_DIAGNOSTIC_ORIGINAL_ATTR = "_duration_occupancy_evidence_only_diagnostics_original"
_WRAPPER_DIAGNOSTIC_PATCH_ATTR = "_state_space_evidence_only_diagnostics_patch_applied"


def apply_duration_occupancy_metadata_guard_patch() -> None:
    """Ensure derived duration/occupancy helper inputs stay isolated and valid."""

    from . import duration_occupancy as _duration_occupancy
    from . import state_space_utils as _state_space_utils

    _apply_transition_duration_validation()
    _patch_evidence_only_path_diagnostics(_duration_occupancy)

    if getattr(_duration_occupancy, "_metadata_guard_patch_applied", False):
        return

    previous_candidate_selection = _duration_occupancy._candidate_selection_emissions
    if not hasattr(_duration_occupancy, "_uniform_probabilities"):
        _duration_occupancy._uniform_probabilities = _state_space_utils._uniform_probabilities
    previous_uniform_probabilities = _duration_occupancy._uniform_probabilities

    def _candidate_selection_emissions(emissions, valid_bin_mask):
        restricted = previous_candidate_selection(emissions, valid_bin_mask)
        if restricted is emissions:
            return restricted
        metadata = dict(getattr(restricted, "metadata", {}))
        return replace(restricted, metadata=metadata)

    def _uniform_probabilities(n_bins: int, valid_bin_mask=None):
        return previous_uniform_probabilities(_positive_integer_bin_count(n_bins), valid_bin_mask)

    _candidate_selection_emissions.__name__ = previous_candidate_selection.__name__
    _candidate_selection_emissions.__doc__ = previous_candidate_selection.__doc__
    _uniform_probabilities.__name__ = previous_uniform_probabilities.__name__
    _uniform_probabilities.__doc__ = previous_uniform_probabilities.__doc__

    _duration_occupancy._candidate_selection_emissions = _candidate_selection_emissions
    _duration_occupancy._uniform_probabilities = _uniform_probabilities
    _duration_occupancy._metadata_guard_patch_applied = True


def _patch_evidence_only_path_diagnostics(duration_occupancy: Any) -> None:
    """Patch evidence-only path-model diagnostics without requiring state_space to be imported."""

    scorer = getattr(duration_occupancy, "_score_state_space_duration_with_occupancy", None)
    if scorer is None:
        return
    if getattr(scorer, _EVIDENCE_ONLY_DIAGNOSTIC_PATCH_ATTR, False) or getattr(scorer, _WRAPPER_DIAGNOSTIC_PATCH_ATTR, False):
        return

    @wraps(scorer)
    def _score_state_space_duration_with_occupancy(
        self,
        emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory: bool = True,
    ):
        result = scorer(
            self,
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
            return_trajectory=return_trajectory,
        )
        if return_trajectory is False:
            _mark_path_model_evidence_only(self, result)
        return result

    setattr(
        _score_state_space_duration_with_occupancy,
        _EVIDENCE_ONLY_DIAGNOSTIC_PATCH_ATTR,
        True,
    )
    setattr(
        _score_state_space_duration_with_occupancy,
        _EVIDENCE_ONLY_DIAGNOSTIC_ORIGINAL_ATTR,
        scorer,
    )
    duration_occupancy._score_state_space_duration_with_occupancy = _score_state_space_duration_with_occupancy


def _mark_path_model_evidence_only(model: Any, result: Any) -> None:
    diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    mode = str(getattr(model, "mode", diagnostics.get("state_space_mode", "")))
    if mode in {"momentum", "momentum-exact-sparse"}:
        diagnostics["state_space_momentum_trajectory_posterior"] = "not_returned_evidence_only"
    elif mode == "imm":
        diagnostics["state_space_imm_trajectory_posterior"] = "not_returned_evidence_only"
    else:
        return
    result.diagnostics = diagnostics


def _contains_boolean_values(values: object) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, bytearray)):
        try:
            if any(isinstance(value, (bool, np.bool_)) for value in values):
                return True
        except TypeError:
            pass
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        raw = np.asarray(values, dtype=object)
    if raw.size == 0:
        return False
    if np.issubdtype(raw.dtype, np.bool_):
        return True
    if raw.dtype == object:
        return any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    return False


def _positive_integer_bin_count(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("n_bins must be a positive integer")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise ValueError("n_bins must be a positive integer") from exc
    if count <= 0:
        raise ValueError("n_bins must be a positive integer")
    return int(count)


def _apply_transition_duration_validation() -> None:
    from . import state_space_displacement_imm as displacement_imm
    from . import state_space_displacement_momentum as displacement_momentum
    from . import state_space_sparse_momentum as sparse_momentum
    from . import state_space_trajectory_imm as trajectory_imm

    sparse_decay = _validated_decay_helper(sparse_momentum._duration_adjusted_decays)
    displacement_decay = _validated_decay_helper(displacement_momentum._duration_adjusted_decays)

    if not getattr(sparse_momentum, "_transition_duration_validation_patch_applied", False):
        sparse_momentum._coerce_transition_durations = _coerce_transition_durations
        sparse_momentum._duration_adjusted_decays = sparse_decay
        sparse_momentum._transition_duration_validation_patch_applied = True

    if not getattr(trajectory_imm, "_transition_duration_validation_patch_applied", False):
        trajectory_imm._coerce_transition_durations = _coerce_transition_durations
        trajectory_imm._duration_adjusted_decays = sparse_decay
        trajectory_imm._transition_duration_validation_patch_applied = True

    if not getattr(displacement_momentum, "_transition_duration_validation_patch_applied", False):
        displacement_momentum._coerce_transition_durations = _coerce_transition_durations
        displacement_momentum._duration_adjusted_decays = displacement_decay
        displacement_momentum._transition_duration_validation_patch_applied = True

    if not getattr(displacement_imm, "_transition_duration_validation_patch_applied", False):
        displacement_imm._coerce_transition_durations = _coerce_transition_durations
        displacement_imm._duration_adjusted_decays = displacement_decay
        displacement_imm._transition_duration_validation_patch_applied = True


def _coerce_transition_durations(
    values: Iterable[float],
    *,
    n_time: int,
    fallback_dt: float,
) -> np.ndarray:
    expected = max(int(n_time) - 1, 0)
    raw_values = list(values)
    if len(raw_values) == 0:
        dt = _positive_finite_scalar("fallback dt", fallback_dt)
        return np.full(expected, dt, dtype=float)

    out = np.asarray(raw_values, dtype=float)
    if out.ndim != 1:
        raise ValueError("transition durations must be one-dimensional")
    _validate_transition_durations(raw_values)
    if out.shape != (expected,):
        raise ValueError(
            "transition durations must contain one finite positive value per transition; "
            f"expected shape {(expected,)}, got {out.shape}"
        )
    return out


def _validated_decay_helper(helper: Callable[[Any, np.ndarray, float], np.ndarray]):
    if getattr(helper, "_transition_duration_validation_wrapped", False):
        return helper

    def duration_adjusted_decays(config: Any, durations: np.ndarray, reference_dt: float) -> np.ndarray:
        _validate_transition_durations(durations)
        durations = np.asarray(durations, dtype=float)
        return helper(config, durations, reference_dt)

    duration_adjusted_decays._transition_duration_validation_wrapped = True  # type: ignore[attr-defined]
    return duration_adjusted_decays


def _validate_transition_durations(durations: np.ndarray) -> None:
    if _contains_boolean_values(durations):
        raise ValueError("transition durations must be numeric durations, not boolean values")
    values = np.asarray(durations, dtype=float)
    if values.size == 0:
        return
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("transition durations must be finite and positive")


def _positive_finite_scalar(name: str, value: float) -> float:
    if _contains_boolean_values(value):
        raise ValueError(f"{name} must be a numeric duration, not boolean")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar

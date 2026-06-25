"""Keep duration-candidate metadata and diagnostics isolated."""

from __future__ import annotations

from functools import wraps

import numpy as np

_CANDIDATE_METADATA_PATCHED_FLAG = "_candidate_emission_metadata_patch_applied"
_MOMENTUM_DIAGNOSTICS_PATCHED_FLAG = "_duration_momentum_diagnostics_patch_applied"
_MOMENTUM_DIAGNOSTICS_SCORE_WRAPPED_FLAG = "_duration_momentum_diagnostics_score_wrapped"
_MOMENTUM_DIAGNOSTICS_ORIGINAL_SCORE_ATTR = "_duration_momentum_diagnostics_original_score"


def apply_duration_candidate_metadata_patch() -> None:
    """Patch duration-candidate metadata handling and momentum diagnostics."""

    from . import duration_occupancy

    if not getattr(duration_occupancy, _CANDIDATE_METADATA_PATCHED_FLAG, False):
        original_candidate_selection_emissions = duration_occupancy._candidate_selection_emissions

        def _candidate_selection_emissions_with_metadata_copy(emissions, valid_bin_mask):
            restricted = original_candidate_selection_emissions(emissions, valid_bin_mask)
            if restricted is emissions:
                return restricted
            metadata = getattr(restricted, "metadata", None)
            if metadata is None:
                return restricted
            restricted.metadata = dict(metadata)
            return restricted

        _candidate_selection_emissions_with_metadata_copy.__name__ = (
            original_candidate_selection_emissions.__name__
        )
        _candidate_selection_emissions_with_metadata_copy.__doc__ = (
            original_candidate_selection_emissions.__doc__
        )
        duration_occupancy._candidate_selection_emissions = _candidate_selection_emissions_with_metadata_copy
        setattr(duration_occupancy, _CANDIDATE_METADATA_PATCHED_FLAG, True)

    _patch_duration_momentum_diagnostics(duration_occupancy)


def _patch_duration_momentum_diagnostics(duration_occupancy) -> None:
    """Expose the per-transition momentum parameters used by duration scoring."""

    current_score = duration_occupancy._score_state_space_duration_with_occupancy
    if _contains_duration_momentum_diagnostics(current_score):
        setattr(duration_occupancy, _MOMENTUM_DIAGNOSTICS_PATCHED_FLAG, True)
        return

    original_score = current_score

    @wraps(original_score)
    def _score_state_space_duration_with_momentum_diagnostics(
        self,
        emissions,
        bin_centers,
        candidate_indices=None,
        *,
        occupancy_s=None,
        return_trajectory: bool = True,
    ):
        score = original_score(
            self,
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
            occupancy_s=occupancy_s,
            return_trajectory=return_trajectory,
        )
        _add_duration_momentum_diagnostics(duration_occupancy, score, self, emissions)
        return score

    setattr(
        _score_state_space_duration_with_momentum_diagnostics,
        _MOMENTUM_DIAGNOSTICS_SCORE_WRAPPED_FLAG,
        True,
    )
    setattr(
        _score_state_space_duration_with_momentum_diagnostics,
        _MOMENTUM_DIAGNOSTICS_ORIGINAL_SCORE_ATTR,
        original_score,
    )
    duration_occupancy._score_state_space_duration_with_occupancy = _score_state_space_duration_with_momentum_diagnostics
    _synchronize_loaded_state_space_score(original_score, _score_state_space_duration_with_momentum_diagnostics)
    setattr(duration_occupancy, _MOMENTUM_DIAGNOSTICS_PATCHED_FLAG, True)


def _contains_duration_momentum_diagnostics(score) -> bool:
    seen: set[int] = set()
    current = score
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, _MOMENTUM_DIAGNOSTICS_SCORE_WRAPPED_FLAG, False):
            return True
        current = getattr(current, "__wrapped__", None)
    return False


def _synchronize_loaded_state_space_score(previous_score, patched_score) -> None:
    """Update already-patched StateSpaceReplayModel aliases to the wrapped scorer."""

    import sys

    for module_name in ("hipporeplayimm.state_space", "hipporeplayimm.state_space_model"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        model_type = getattr(module, "StateSpaceReplayModel", None)
        if model_type is None:
            continue
        if getattr(model_type, "score", None) is previous_score:
            setattr(model_type, "score", patched_score)


def _add_duration_momentum_diagnostics(duration_occupancy, score, model, emissions) -> None:
    """Fill duration-aware momentum diagnostics that the native scorer uses internally."""

    mode = str(getattr(model, "mode", ""))
    if mode not in {"momentum", "imm"}:
        return
    config = getattr(model, "config", None)
    diagnostics = getattr(score, "diagnostics", None)
    if config is None or not isinstance(diagnostics, dict):
        return

    durations = np.asarray(duration_occupancy.transition_durations_s(emissions), dtype=float)
    fallback_dt = float(getattr(emissions, "dt", np.nan))
    momentum_sigmas = duration_occupancy._per_transition_sigmas(
        config.momentum_sigma_cm_sqrt_s,
        durations,
    )
    initial_sigmas = duration_occupancy._per_transition_sigmas(
        config.momentum_initial_sigma_cm_sqrt_s,
        durations,
    )
    velocity_decays = duration_occupancy._duration_adjusted_decays(
        config,
        durations,
        fallback_dt,
    )
    transition_sigma = duration_occupancy._representative_sigma(
        config.momentum_sigma_cm_sqrt_s,
        durations,
        fallback_dt,
    )
    if initial_sigmas.size:
        initial_sigma = float(initial_sigmas[0])
    else:
        initial_sigma = duration_occupancy._per_bin_sigma(
            config.momentum_initial_sigma_cm_sqrt_s,
            fallback_dt,
        )

    diagnostics.setdefault("state_space_transition_durations_s", _format_float_series(durations))
    diagnostics.setdefault("state_space_momentum_transition_sigma_cm", float(transition_sigma))
    diagnostics.setdefault("state_space_momentum_initial_transition_sigma_cm", float(initial_sigma))
    diagnostics.setdefault("state_space_momentum_transition_sigma_cm_per_step", _format_float_series(momentum_sigmas))
    diagnostics.setdefault("state_space_momentum_initial_transition_sigma_cm_per_step", _format_float_series(initial_sigmas))
    diagnostics.setdefault("state_space_momentum_velocity_decay_effective", _representative_value(velocity_decays, config.momentum_velocity_decay))
    diagnostics.setdefault("state_space_momentum_velocity_decay_per_step", _format_float_series(velocity_decays))


def _format_float_series(values) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    return ",".join(f"{float(value):.12g}" for value in array)


def _representative_value(values, fallback: float) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(fallback)
    return float(np.median(finite))

"""Keep duration-candidate metadata and diagnostics isolated."""

from __future__ import annotations

import inspect
from functools import wraps

import numpy as np

_CANDIDATE_METADATA_PATCHED_FLAG = "_candidate_emission_metadata_patch_applied"
_CANDIDATE_METADATA_SELECTION_WRAPPED_FLAG = "_duration_candidate_metadata_selection_wrapped"
_CANDIDATE_METADATA_ORIGINAL_SELECTION_ATTR = "_duration_candidate_metadata_original_selection"
_CANDIDATE_SIGNATURE_PATCHED_FLAG = "_duration_candidate_signature_patch_applied"
_CANDIDATE_SIGNATURE_WRAPPED_FLAG = "_duration_candidate_signature_wrapped"
_CANDIDATE_SIGNATURE_ORIGINAL_ATTR = "_duration_candidate_signature_original"
_MOMENTUM_DIAGNOSTICS_PATCHED_FLAG = "_duration_momentum_diagnostics_patch_applied"
_MOMENTUM_DIAGNOSTICS_SCORE_WRAPPED_FLAG = "_duration_momentum_diagnostics_score_wrapped"
_MOMENTUM_DIAGNOSTICS_ORIGINAL_SCORE_ATTR = "_duration_momentum_diagnostics_original_score"


def apply_duration_candidate_metadata_patch() -> None:
    """Patch duration-candidate metadata handling and momentum diagnostics."""

    from . import duration_occupancy

    _patch_candidate_selection_metadata_copy(duration_occupancy)
    _patch_candidate_indices_signature_compat(duration_occupancy)
    _patch_duration_momentum_diagnostics(duration_occupancy)


def _patch_candidate_selection_metadata_copy(duration_occupancy) -> None:
    """Ensure restricted candidate-emission metadata is isolated from callers."""

    current = duration_occupancy._candidate_selection_emissions
    if getattr(current, _CANDIDATE_METADATA_SELECTION_WRAPPED_FLAG, False):
        setattr(duration_occupancy, _CANDIDATE_METADATA_PATCHED_FLAG, True)
        return

    @wraps(current)
    def _candidate_selection_emissions_with_metadata_copy(emissions, valid_bin_mask):
        restricted = current(emissions, valid_bin_mask)
        if restricted is emissions:
            return restricted
        metadata = getattr(restricted, "metadata", None)
        if metadata is None:
            return restricted
        restricted.metadata = dict(metadata)
        return restricted

    setattr(
        _candidate_selection_emissions_with_metadata_copy,
        _CANDIDATE_METADATA_SELECTION_WRAPPED_FLAG,
        True,
    )
    setattr(
        _candidate_selection_emissions_with_metadata_copy,
        _CANDIDATE_METADATA_ORIGINAL_SELECTION_ATTR,
        current,
    )
    duration_occupancy._candidate_selection_emissions = _candidate_selection_emissions_with_metadata_copy
    setattr(duration_occupancy, _CANDIDATE_METADATA_PATCHED_FLAG, True)


def _patch_candidate_indices_signature_compat(duration_occupancy) -> None:
    """Keep duration scoring compatible with legacy candidate generators."""

    current = duration_occupancy._duration_candidates
    if getattr(current, _CANDIDATE_SIGNATURE_WRAPPED_FLAG, False):
        setattr(duration_occupancy, _CANDIDATE_SIGNATURE_PATCHED_FLAG, True)
        return

    @wraps(current)
    def _duration_candidates_with_signature_compat(
        ss,
        model,
        emissions,
        bin_centers,
        candidate_indices,
        valid_bin_mask,
    ):
        if candidate_indices is not None:
            return current(
                ss,
                model,
                emissions,
                bin_centers,
                candidate_indices,
                valid_bin_mask,
            )

        candidate_emissions = duration_occupancy._candidate_selection_emissions(
            emissions,
            valid_bin_mask,
        )
        candidates = _call_candidate_indices_with_optional_mask(
            model.candidate_indices,
            candidate_emissions,
            bin_centers,
            valid_bin_mask,
        )
        candidates = ss._validate_candidate_indices(
            candidates,
            emissions.n_time,
            emissions.n_bins,
        )
        candidates = ss._restrict_candidates_to_valid_bins(
            candidates,
            emissions.log_likelihood,
            valid_bin_mask,
        )
        return ss._validate_candidate_indices(
            candidates,
            emissions.n_time,
            emissions.n_bins,
        )

    setattr(
        _duration_candidates_with_signature_compat,
        _CANDIDATE_SIGNATURE_WRAPPED_FLAG,
        True,
    )
    setattr(
        _duration_candidates_with_signature_compat,
        _CANDIDATE_SIGNATURE_ORIGINAL_ATTR,
        current,
    )
    duration_occupancy._duration_candidates = _duration_candidates_with_signature_compat
    setattr(duration_occupancy, _CANDIDATE_SIGNATURE_PATCHED_FLAG, True)


def _call_candidate_indices_with_optional_mask(
    candidate_indices,
    emissions,
    bin_centers,
    valid_bin_mask,
):
    """Call a candidate generator without hiding implementation ``TypeError``s."""

    try:
        signature = inspect.signature(candidate_indices)
    except (TypeError, ValueError):
        return candidate_indices(emissions, bin_centers)

    parameters = signature.parameters
    mask_parameter = parameters.get("valid_bin_mask")
    if (
        mask_parameter is not None
        and mask_parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    ):
        positional = tuple(
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        mask_position = positional.index(mask_parameter)
        if mask_position == 1:
            return candidate_indices(emissions, valid_bin_mask)
        if mask_position == 2:
            return candidate_indices(emissions, bin_centers, valid_bin_mask)

    supports_mask_keyword = (
        mask_parameter is not None
        and mask_parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not supports_mask_keyword:
        return _call_candidate_indices_legacy(candidate_indices, signature, emissions, bin_centers)

    kwargs = {"valid_bin_mask": valid_bin_mask}
    positional = tuple(
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters.values()
    ) or len(positional) >= 2:
        return candidate_indices(emissions, bin_centers, **kwargs)
    if "bin_centers" in parameters:
        return candidate_indices(emissions, bin_centers=bin_centers, **kwargs)
    if "centers" in parameters:
        return candidate_indices(emissions, centers=bin_centers, **kwargs)
    return candidate_indices(emissions, **kwargs)


def _call_candidate_indices_legacy(candidate_indices, signature, emissions, bin_centers):
    """Call historical one- or two-argument candidate generators."""

    parameters = tuple(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters):
        return candidate_indices(emissions, bin_centers)
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if len(positional) >= 2:
        return candidate_indices(emissions, bin_centers)
    if "bin_centers" in signature.parameters:
        return candidate_indices(emissions, bin_centers=bin_centers)
    if "centers" in signature.parameters:
        return candidate_indices(emissions, centers=bin_centers)
    return candidate_indices(emissions)


def _patch_duration_momentum_diagnostics(duration_occupancy) -> None:
    """Expose the per-transition momentum parameters used by duration scoring."""

    current_score = duration_occupancy._score_state_space_duration_with_occupancy
    if _contains_duration_momentum_diagnostics(current_score):
        original_score = _duration_momentum_diagnostics_original_score(current_score)
        if original_score is not None:
            _synchronize_loaded_state_space_score(original_score, current_score)
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


def _duration_momentum_diagnostics_original_score(score):
    seen: set[int] = set()
    current = score
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        original_score = getattr(current, _MOMENTUM_DIAGNOSTICS_ORIGINAL_SCORE_ATTR, None)
        if original_score is not None:
            return original_score
        current = getattr(current, "__wrapped__", None)
    return None


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
    config = getattr(model, "config", None)
    diagnostics = getattr(score, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return

    durations = np.asarray(duration_occupancy.transition_durations_s(emissions), dtype=float)
    diagnostics.setdefault("state_space_transition_durations_s", _format_float_series(durations))

    if mode not in {"momentum", "imm"}:
        return
    if config is None:
        return

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

    if mode == "imm":
        diffusion_sigmas = duration_occupancy._per_transition_sigmas(
            config.diffusion_sigma_cm_sqrt_s,
            durations,
        )
        diagnostics.setdefault("state_space_diffusion_transition_sigma_cm_per_step", _format_float_series(diffusion_sigmas))

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

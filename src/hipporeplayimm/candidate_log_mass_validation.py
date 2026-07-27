"""Validate candidate-path retained emission masses before scoring.

Candidate-pruned path recursions normalize posterior marginals after retaining
only a spatial support subset. If a retained support has no finite emission mass,
the path evidence is zero and the posterior cannot be normalized. Guard that
case before dynamic-program arrays can propagate NaN or LOG_ZERO-normalized
fallbacks.
"""

from __future__ import annotations

from functools import wraps

import numpy as np
from scipy.special import logsumexp

_PATCHED_FLAG = "_candidate_log_mass_validation_patch_applied"
_DURATION_OCCUPANCY_PATCHED_FLAG = "_duration_occupancy_candidate_log_mass_patch_applied"
_REPORTED_MINIMUM_WRAPPER_ATTR = "_candidate_reported_log_mass_minimum_wrapper"
_REPORTED_COLUMN_MINIMUM_WRAPPER_ATTR = "_candidate_reported_log_mass_column_minimum_wrapper"
_UNPRUNED_SUPPORT_WRAPPER_ATTR = "_candidate_unpruned_exact_support_wrapper"


def _candidate_log_masses(log_likelihood: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
    """Return finite retained emission masses for candidate supports.

    The retained-mass diagnostic is also an early consistency check for
    candidate-pruned recursions: every emission row must contain finite mass on
    the full support and on the supplied candidate support.
    """

    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    if np.any(np.isnan(values)) or np.any(values == np.inf):
        raise ValueError("log_likelihood must not contain NaN or +inf")
    if len(candidates) != values.shape[0]:
        raise ValueError("candidate_indices must contain one array per emission time bin")

    masses: list[float] = []
    for time_index, curr in enumerate(candidates):
        current = np.asarray(curr)
        if current.ndim != 1:
            raise ValueError(f"candidate_indices[{time_index}] must be one-dimensional")
        if current.size == 0:
            raise ValueError(f"candidate_indices[{time_index}] must not be empty")
        if not np.issubdtype(current.dtype, np.integer):
            raise TypeError(f"candidate_indices[{time_index}] must contain integer bin indices")
        current = current.astype(np.intp, copy=False)
        if np.any((current < 0) | (current >= values.shape[1])):
            raise ValueError(f"candidate_indices[{time_index}] contains an out-of-range bin")
        if np.unique(current).size != current.size:
            raise ValueError(f"candidate_indices[{time_index}] contains duplicate bins")

        row = values[time_index]
        total = logsumexp(row)
        if not np.isfinite(total):
            raise ValueError(f"log_likelihood row {time_index} must contain at least one finite spatial-bin value")

        selected = logsumexp(row[current])
        if not np.isfinite(selected):
            raise ValueError(f"candidate_indices[{time_index}] select no finite likelihood mass")

        masses.append(float(selected - total))
    return masses


def _candidate_log_masses_on_active_support(
    log_likelihood: np.ndarray,
    candidates: list[np.ndarray],
    valid_bin_mask: np.ndarray | None,
) -> list[float]:
    """Return retained masses normalized over the scored occupancy support.

    Occupancy-aware state-space scoring masks invalid spatial bins before the
    dynamic program sees them. The retained-mass diagnostic should use that same
    active support; otherwise an exact active-support candidate set can look
    artificially pruned merely because excluded bins had high emission mass.
    """

    if valid_bin_mask is None:
        return _candidate_log_masses(log_likelihood, candidates)

    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must be two-dimensional")
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != (values.shape[1],):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")

    active_values = values.copy()
    active_values[:, ~mask] = -np.inf
    return _candidate_log_masses(active_values, candidates)


def _candidate_log_masses_for_model(emissions, candidates: list[np.ndarray]) -> list[float]:
    """Compatibility wrapper for legacy CandidateKinematicModel helpers."""

    return _candidate_log_masses(emissions.log_likelihood, candidates)


def _is_full_grid_candidate_support(candidates: list[np.ndarray], n_time: int, n_bins: int) -> bool:
    """Return whether every time bin retains each spatial bin exactly once."""

    if len(candidates) != int(n_time):
        return False
    expected = np.arange(int(n_bins), dtype=np.intp)
    for current in candidates:
        values = np.asarray(current)
        if values.ndim != 1 or values.size != int(n_bins):
            return False
        if not np.issubdtype(values.dtype, np.integer):
            return False
        if not np.array_equal(np.sort(values.astype(np.intp, copy=False)), expected):
            return False
    return True


def _patch_unpruned_candidate_evidence_support(models) -> None:
    """Label the legacy ``top_k=0`` path as exact rather than truncated."""

    current = models.CandidateKinematicModel.score
    if getattr(current, _UNPRUNED_SUPPORT_WRAPPER_ATTR, False):
        return

    @wraps(current)
    def score(self, emissions, bin_centers, candidate_indices=None):
        result = current(
            self,
            emissions,
            bin_centers,
            candidate_indices=candidate_indices,
        )
        if candidate_indices is not None or int(getattr(self, "top_k", -1)) != 0:
            return result
        if int(getattr(emissions, "n_time", 0)) <= 1:
            return result

        candidates = self.candidate_indices(emissions)
        if not _is_full_grid_candidate_support(
            candidates,
            emissions.n_time,
            emissions.n_bins,
        ):
            return result

        diagnostics = dict(getattr(result, "diagnostics", {}) or {})
        if diagnostics.get("candidate_evidence_support") == "truncated_full_grid":
            diagnostics["candidate_evidence_support"] = "exact_full_grid"
            result.diagnostics = diagnostics
        return result

    setattr(score, _UNPRUNED_SUPPORT_WRAPPER_ATTR, True)
    setattr(score, "__hipporeplayimm_original__", current)
    models.CandidateKinematicModel.score = score


def _current_patch_installed(
    models,
    state_space,
    state_space_candidates,
    state_space_candidates_momentum,
    state_space_utils,
) -> bool:
    return (
        getattr(state_space, _PATCHED_FLAG, False)
        and getattr(state_space_utils, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(state_space, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(state_space_candidates, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(state_space_candidates_momentum, "_candidate_log_masses", None) is _candidate_log_masses
        and getattr(models, "_candidate_log_masses", None) is _candidate_log_masses_for_model
    )


def _patch_duration_occupancy_candidate_masses() -> None:
    """Keep duration/occupancy path diagnostics normalized on active support."""

    from . import duration_occupancy

    if getattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, False):
        return

    original_momentum = duration_occupancy._score_momentum_duration
    original_imm = duration_occupancy._score_imm_duration

    @wraps(original_momentum)
    def score_momentum_duration(
        ss,
        emissions,
        bin_centers,
        candidates,
        *,
        sigmas_cm,
        initial_sigma_cm,
        velocity_decays,
        time_scales,
        valid_bin_mask=None,
    ):
        logp, trajectory, masses = original_momentum(
            ss,
            emissions,
            bin_centers,
            candidates,
            sigmas_cm=sigmas_cm,
            initial_sigma_cm=initial_sigma_cm,
            velocity_decays=velocity_decays,
            time_scales=time_scales,
            valid_bin_mask=valid_bin_mask,
        )
        if valid_bin_mask is not None and int(getattr(emissions, "n_time", 0)) > 1:
            masses = _candidate_log_masses_on_active_support(
                emissions.log_likelihood,
                candidates,
                valid_bin_mask,
            )
        return logp, trajectory, masses

    @wraps(original_imm)
    def score_imm_duration(
        ss,
        emissions,
        bin_centers,
        candidates,
        *,
        stationary_sigma_cm,
        diffusion_sigmas_cm,
        momentum_sigmas_cm,
        initial_momentum_sigma_cm,
        velocity_decays,
        time_scales,
        mode_stickiness,
        mode_transitions=None,
        valid_bin_mask=None,
    ):
        logp, trajectory, mode_posterior, masses = original_imm(
            ss,
            emissions,
            bin_centers,
            candidates,
            stationary_sigma_cm=stationary_sigma_cm,
            diffusion_sigmas_cm=diffusion_sigmas_cm,
            momentum_sigmas_cm=momentum_sigmas_cm,
            initial_momentum_sigma_cm=initial_momentum_sigma_cm,
            velocity_decays=velocity_decays,
            time_scales=time_scales,
            mode_stickiness=mode_stickiness,
            mode_transitions=mode_transitions,
            valid_bin_mask=valid_bin_mask,
        )
        if valid_bin_mask is not None and int(getattr(emissions, "n_time", 0)) > 1:
            masses = _candidate_log_masses_on_active_support(
                emissions.log_likelihood,
                candidates,
                valid_bin_mask,
            )
        return logp, trajectory, mode_posterior, masses

    duration_occupancy._score_momentum_duration = score_momentum_duration
    duration_occupancy._score_imm_duration = score_imm_duration
    setattr(duration_occupancy, _DURATION_OCCUPANCY_PATCHED_FLAG, True)


def _patch_reported_candidate_log_mass_minimum() -> None:
    """Use the worst finite value from array-like minimum-mass diagnostics."""

    from . import result_improvements

    current = result_improvements._first_finite_numeric_value
    if getattr(current, _REPORTED_MINIMUM_WRAPPER_ATTR, False):
        return

    @wraps(current)
    def minimum_finite_numeric_value(value: object) -> float | None:
        try:
            values = np.asarray(value, dtype=object)
        except (TypeError, ValueError):
            return current(value)
        if values.ndim == 0:
            return current(values.item())
        items = list(values.reshape(-1))
        if any(isinstance(item, (bool, np.bool_)) for item in items):
            return None
        finite = [number for item in items if (number := current(item)) is not None]
        return min(finite) if finite else None

    setattr(minimum_finite_numeric_value, _REPORTED_MINIMUM_WRAPPER_ATTR, True)
    setattr(minimum_finite_numeric_value, "__hipporeplayimm_original__", current)
    result_improvements._first_finite_numeric_value = minimum_finite_numeric_value


def _patch_reported_candidate_log_mass_across_columns() -> None:
    """Use the worst finite minimum-mass diagnostic available on a row."""

    from . import result_improvements

    current = result_improvements._candidate_min_log_mass
    if getattr(current, _REPORTED_COLUMN_MINIMUM_WRAPPER_ATTR, False):
        return

    @wraps(current)
    def minimum_candidate_log_mass(row) -> float:
        columns: list[object] = list(result_improvements._CANDIDATE_MIN_LOG_MASS_COLUMNS)
        seen = {str(column) for column in columns}
        for column in getattr(row, "index", ()):
            name = str(column)
            if name.endswith("_min_candidate_log_mass") and name not in seen:
                columns.append(column)
                seen.add(name)

        finite: list[float] = []
        for column in columns:
            value = row.get(column)
            scalar = result_improvements._first_finite_numeric_value(value)
            if scalar is not None:
                finite.append(scalar)
        return min(finite) if finite else float("nan")

    setattr(minimum_candidate_log_mass, _REPORTED_COLUMN_MINIMUM_WRAPPER_ATTR, True)
    setattr(minimum_candidate_log_mass, "__hipporeplayimm_original__", current)
    result_improvements._candidate_min_log_mass = minimum_candidate_log_mass


def apply_candidate_log_mass_validation_patch() -> None:
    """Install finite retained-mass validation on candidate-pruned scorers."""

    from . import models
    from . import state_space
    from . import state_space_candidates
    from . import state_space_candidates_momentum
    from . import state_space_utils

    if not _current_patch_installed(
        models,
        state_space,
        state_space_candidates,
        state_space_candidates_momentum,
        state_space_utils,
    ):
        state_space_utils._candidate_log_masses = _candidate_log_masses
        state_space._candidate_log_masses = _candidate_log_masses
        state_space_candidates._candidate_log_masses = _candidate_log_masses
        state_space_candidates_momentum._candidate_log_masses = _candidate_log_masses
        models._candidate_log_masses = _candidate_log_masses_for_model
        setattr(state_space, _PATCHED_FLAG, True)

    _patch_unpruned_candidate_evidence_support(models)
    _patch_duration_occupancy_candidate_masses()
    _patch_reported_candidate_log_mass_minimum()
    _patch_reported_candidate_log_mass_across_columns()


__all__ = [
    "apply_candidate_log_mass_validation_patch",
]

"""Exact finite-displacement IMM state-space decoder.

This module combines the exact first-order IMM idea with the finite-displacement
momentum surrogate.  It provides an exact evidence over a declared augmented
state space ``(mode, position, displacement)`` and is intended as the
paper-facing comparable counterpart to candidate-pruned full pairwise IMM.

The model uses four destination modes:

* stationary: first-order Gaussian transition with a fresh displacement prior;
* diffusion: first-order diffusion transition with a fresh displacement prior;
* displacement-momentum: displacement evolves with an AR(1)-like Gaussian
  transition and shifts position by the current displacement; and
* jump: position is redrawn from the valid-bin prior with a fresh displacement
  prior.

The non-momentum modes keep displacement as a nuisance variable.  They resample
it from the same zero-centred displacement prior at each step, which keeps the
joint state space fixed while avoiding accidental carry-over of momentum in
stationary/diffusion/jump modes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .encoding import LogEmissionTensor
from .state_space_displacement_momentum import (
    _as_2d_centers,
    _coerce_transition_durations,
    _coerce_valid_bin_mask,
    _default_position_sigma_cm,
    _displacement_lattice,
    _displacement_transition_matrix,
    _duration_adjusted_decays,
    _duration_scale_at,
    _format_float_series,
    _format_vector_series,
    _per_bin_sigma,
    _positive_config_value,
    _shifted_gaussian_transition_matrix,
    _time_scales,
    _transition_sigmas_cm,
    _uniform_position_prior,
    _valid_count,
    _zero_centered_displacement_prior,
)
from .state_space_utils import (
    _as_log_probs,
    _gaussian_transition_matrix,
    _mean_entropy,
    _mode_transition_matrix,
    _scaled_emissions,
)

_DISPLACEMENT_IMM_MODES = (
    "stationary",
    "diffusion",
    "displacement-momentum",
    "jump",
)


def _score_displacement_imm_exact(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    config: object,
    transition_durations_s: Iterable[float],
    *,
    valid_bin_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Return exact evidence/posteriors for a finite-displacement IMM.

    The returned tuple is ``(log_evidence, position_log_posterior,
    mode_posterior, displacement_log_posterior, diagnostics)``.
    """

    centers = _as_2d_centers(bin_centers)
    if emissions.n_time <= 0:
        raise ValueError("emissions must contain at least one time bin")
    if emissions.n_bins != centers.shape[0]:
        raise ValueError("emissions.n_bins must match bin_centers rows")

    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, emissions.n_bins)
    scaled, offsets = _scaled_emissions(emissions.log_likelihood, valid_bin_mask=valid_mask)
    if valid_mask is not None:
        scaled[:, ~valid_mask] = 0.0

    durations = _coerce_transition_durations(
        transition_durations_s,
        n_time=emissions.n_time,
        fallback_dt=float(emissions.dt),
    )
    reference_dt = float(np.median(durations)) if durations.size else float(emissions.dt)
    vectors = _displacement_lattice(
        centers,
        radius_bins=int(getattr(config, "displacement_radius_bins", 2)),
    )
    n_modes = len(_DISPLACEMENT_IMM_MODES)
    n_bins = emissions.n_bins
    n_displacements = vectors.shape[0]
    valid_count = _valid_count(valid_mask, n_bins)
    if n_displacements <= 0:
        raise ValueError("displacement lattice must contain at least one state")

    displacement_transition_sigmas = _transition_sigmas_cm(config, durations)
    diffusion_sigmas = np.asarray(
        [
            _per_bin_sigma(float(getattr(config, "diffusion_sigma_cm_sqrt_s", 85.0)), duration)
            for duration in durations
        ],
        dtype=float,
    )
    position_sigma = _positive_config_value(
        config,
        "displacement_position_sigma_cm",
        default=_default_position_sigma_cm(centers),
    )
    prior_sigma = _positive_config_value(
        config,
        "displacement_prior_sigma_cm",
        default=_per_bin_sigma(
            float(getattr(config, "momentum_initial_sigma_cm_sqrt_s", 85.0)),
            reference_dt,
        ),
    )
    decays = _duration_adjusted_decays(config, durations, float(emissions.dt))
    time_scales = _time_scales(durations)
    mode_transition = _mode_transition_matrix(
        n_modes,
        float(getattr(config, "imm_mode_stickiness", 0.95)),
    )

    position_prior = _uniform_position_prior(n_bins, valid_mask)
    displacement_prior = _zero_centered_displacement_prior(vectors, prior_sigma)
    mode_prior = np.full(n_modes, 1.0 / n_modes, dtype=float)
    alpha = (
        mode_prior[:, None, None]
        * position_prior[None, :, None]
        * displacement_prior[None, None, :]
        * scaled[0][None, :, None]
    )
    scales = np.zeros(emissions.n_time, dtype=float)
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered = np.zeros(
        (emissions.n_time, n_modes, n_bins, n_displacements),
        dtype=float,
    )
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    stationary_transition = _gaussian_transition_matrix(
        centers,
        float(getattr(config, "stationary_sigma_cm", 2.0)),
        float(getattr(config, "max_step_sigma", 4.0)),
        valid_bin_mask=valid_mask,
    )
    transition_cache: list[dict[str, object]] = []
    for time_index in range(1, emissions.n_time):
        transition_index = time_index - 1
        diffusion_transition = _gaussian_transition_matrix(
            centers,
            float(diffusion_sigmas[transition_index]),
            float(getattr(config, "max_step_sigma", 4.0)),
            valid_bin_mask=valid_mask,
        )
        displacement_transition = _displacement_transition_matrix(
            vectors,
            sigma_cm=float(displacement_transition_sigmas[transition_index]),
            decay=float(decays[transition_index]) * float(time_scales[transition_index]),
        )
        duration_scale = _duration_scale_at(durations, transition_index, reference_dt)
        momentum_spatial_transitions = [
            _shifted_gaussian_transition_matrix(
                centers,
                displacement=duration_scale * vectors[disp_index],
                sigma_cm=position_sigma,
                max_step_sigma=float(getattr(config, "max_step_sigma", 4.0)),
                valid_bin_mask=valid_mask,
            )
            for disp_index in range(n_displacements)
        ]
        cache = {
            "diffusion_transition": diffusion_transition,
            "displacement_transition": displacement_transition,
            "momentum_spatial_transitions": momentum_spatial_transitions,
        }
        transition_cache.append(cache)

        predicted = np.zeros_like(alpha)
        for dst_mode_index, dst_mode in enumerate(_DISPLACEMENT_IMM_MODES):
            mixed = np.zeros((n_bins, n_displacements), dtype=float)
            for src_mode_index in range(n_modes):
                mixed += mode_transition[src_mode_index, dst_mode_index] * alpha[src_mode_index]
            predicted[dst_mode_index] = _advance_mode(
                mixed,
                mode=dst_mode,
                stationary_transition=stationary_transition,
                diffusion_transition=diffusion_transition,
                displacement_transition=displacement_transition,
                momentum_spatial_transitions=momentum_spatial_transitions,
                displacement_prior=displacement_prior,
                position_prior=position_prior,
            )
        alpha = predicted * scaled[time_index][None, :, None]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_modes, n_bins, n_displacements), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(emissions.n_time - 1, 0, -1):
        cache = transition_cache[time_index - 1]
        beta_prev = np.zeros_like(beta)
        for src_mode_index in range(n_modes):
            total = np.zeros((n_bins, n_displacements), dtype=float)
            for dst_mode_index, dst_mode in enumerate(_DISPLACEMENT_IMM_MODES):
                values = scaled[time_index][:, None] * beta[dst_mode_index]
                total += mode_transition[src_mode_index, dst_mode_index] * _backward_mode(
                    values,
                    mode=dst_mode,
                    stationary_transition=stationary_transition,
                    diffusion_transition=cache["diffusion_transition"],
                    displacement_transition=cache["displacement_transition"],
                    momentum_spatial_transitions=cache["momentum_spatial_transitions"],
                    displacement_prior=displacement_prior,
                    valid_bin_mask=valid_mask,
                )
            beta_prev[src_mode_index] = total / scales[time_index]
        beta = beta_prev
        gamma = filtered[time_index - 1] * beta
        total_gamma = float(gamma.sum())
        smoothed[time_index - 1] = (
            gamma / total_gamma if total_gamma > 0.0 else filtered[time_index - 1]
        )

    position_posterior = smoothed.sum(axis=(1, 3))
    mode_posterior = smoothed.sum(axis=(2, 3))
    displacement_posterior = smoothed.sum(axis=(1, 2))
    trajectory = _as_log_probs(position_posterior)
    mode_log_posterior = _as_log_probs(mode_posterior)
    displacement_log_posterior = _as_log_probs(displacement_posterior)

    median_displacement_sigma = _median_or_fallback(
        displacement_transition_sigmas,
        _per_bin_sigma(
            float(getattr(config, "momentum_sigma_cm_sqrt_s", 85.0)),
            reference_dt,
        ),
    )
    median_diffusion_sigma = _median_or_fallback(
        diffusion_sigmas,
        _per_bin_sigma(
            float(getattr(config, "diffusion_sigma_cm_sqrt_s", 85.0)),
            reference_dt,
        ),
    )
    median_decay = _median_or_fallback(
        decays,
        float(getattr(config, "momentum_velocity_decay", 0.95)),
    )
    diagnostics: dict[str, float | int | str] = {
        "state_space_displacement_imm_evidence_support": "exact_full_grid",
        "state_space_displacement_imm_state_support": "finite_displacement_grid",
        "state_space_displacement_imm_modes": ",".join(_DISPLACEMENT_IMM_MODES),
        "state_space_displacement_imm_mode_count": int(n_modes),
        "state_space_displacement_imm_state_count": int(n_modes * valid_count * n_displacements),
        "state_space_displacement_state_count": int(n_displacements),
        "state_space_displacement_joint_state_count": int(valid_count * n_displacements),
        "state_space_displacement_radius_bins": int(getattr(config, "displacement_radius_bins", 2)),
        "state_space_displacement_position_sigma_cm": float(position_sigma),
        "state_space_displacement_prior_sigma_cm": float(prior_sigma),
        "state_space_displacement_imm_transition_sigma_cm": float(median_displacement_sigma),
        "state_space_displacement_imm_diffusion_transition_sigma_cm": float(median_diffusion_sigma),
        "state_space_displacement_imm_velocity_decay_effective": float(median_decay),
        "state_space_displacement_imm_transition_sigma_cm_per_step": _format_float_series(
            displacement_transition_sigmas
        ),
        "state_space_displacement_imm_diffusion_transition_sigma_cm_per_step": _format_float_series(
            diffusion_sigmas
        ),
        "state_space_displacement_imm_velocity_decay_per_step": _format_float_series(decays),
        "state_space_displacement_imm_mean_mode_entropy": _mean_entropy(mode_log_posterior),
        "state_space_displacement_imm_mean_displacement_entropy": _mean_entropy(
            displacement_log_posterior
        ),
        "state_space_displacement_vectors_cm": _format_vector_series(vectors),
    }
    return (
        float(logp),
        trajectory,
        mode_posterior,
        displacement_log_posterior,
        diagnostics,
    )


def _advance_mode(
    mixed: np.ndarray,
    *,
    mode: str,
    stationary_transition,
    diffusion_transition,
    displacement_transition: np.ndarray,
    momentum_spatial_transitions: list[object],
    displacement_prior: np.ndarray,
    position_prior: np.ndarray,
) -> np.ndarray:
    if mode == "displacement-momentum":
        mixed_displacements = mixed @ displacement_transition.T
        predicted = np.zeros_like(mixed)
        for disp_index, transition in enumerate(momentum_spatial_transitions):
            predicted[:, disp_index] = np.asarray(
                transition @ mixed_displacements[:, disp_index],
                dtype=float,
            )
        return predicted

    if mode == "stationary":
        predicted_position = np.asarray(stationary_transition @ mixed.sum(axis=1), dtype=float)
    elif mode == "diffusion":
        predicted_position = np.asarray(diffusion_transition @ mixed.sum(axis=1), dtype=float)
    elif mode == "jump":
        predicted_position = position_prior * float(mixed.sum())
    else:
        raise ValueError(f"Unknown displacement IMM mode: {mode}")
    return predicted_position[:, None] * displacement_prior[None, :]


def _backward_mode(
    values: np.ndarray,
    *,
    mode: str,
    stationary_transition,
    diffusion_transition,
    displacement_transition: np.ndarray,
    momentum_spatial_transitions: list[object],
    displacement_prior: np.ndarray,
    valid_bin_mask: np.ndarray | None,
) -> np.ndarray:
    if mode == "displacement-momentum":
        spatial_backward = np.zeros_like(values)
        for disp_index, transition in enumerate(momentum_spatial_transitions):
            spatial_backward[:, disp_index] = np.asarray(
                transition.T @ values[:, disp_index],
                dtype=float,
            )
        return spatial_backward @ displacement_transition

    weighted_values = values @ displacement_prior
    if mode == "stationary":
        previous_position = np.asarray(stationary_transition.T @ weighted_values, dtype=float)
    elif mode == "diffusion":
        previous_position = np.asarray(diffusion_transition.T @ weighted_values, dtype=float)
    elif mode == "jump":
        previous_position = _uniform_backward(weighted_values, valid_bin_mask)
    else:
        raise ValueError(f"Unknown displacement IMM mode: {mode}")
    return np.repeat(previous_position[:, None], values.shape[1], axis=1)


def _uniform_backward(values: np.ndarray, valid_bin_mask: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if valid_bin_mask is None:
        return np.full(values.shape, float(values.sum()) / values.shape[0], dtype=float)
    mask = np.asarray(valid_bin_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(mask):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    out = np.zeros(values.shape, dtype=float)
    out[mask] = float(values[mask].sum()) / int(np.sum(mask))
    return out


def _median_or_fallback(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(fallback) if arr.size == 0 else float(np.median(arr))

"""Exact sparse trajectory-family IMM state-space decoder.

This module combines exact first-order stationary/diffusion/fragmented modes
with the exact sparse finite-radius pair-grid momentum mode.  The dynamic
program keeps first-order modes as position marginals and keeps only the
momentum mode on sparse pair states ``(x[t-1], x[t])``.  This preserves the
history required for momentum while avoiding a dense pair grid for modes that do
not need one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from .encoding import LogEmissionTensor
from .state_space_sparse_momentum import (
    _as_2d_centers,
    _coalesce_pairs,
    _coerce_transition_durations,
    _coerce_valid_bin_mask,
    _duration_adjusted_decays,
    _finite_gaussian_row,
    _format_float_series,
    _per_bin_sigma,
    _per_transition_sigmas,
    _time_scales,
    _uniform_position_prior,
    _valid_indices,
)
from .state_space_utils import (
    _as_log_probs,
    _gaussian_transition_matrix,
    _mean_entropy,
    _mode_transition_matrix,
    _scaled_emissions,
)

_TRAJECTORY_IMM_MODES = (
    "stationary",
    "diffusion",
    "fragmented",
    "momentum-exact-sparse",
)
_FIRST_ORDER_MODE_COUNT = 3
_MOMENTUM_MODE_INDEX = 3


@dataclass
class _ForwardState:
    first_order: np.ndarray
    momentum_prev: np.ndarray
    momentum_curr: np.ndarray
    momentum_alpha: np.ndarray


@dataclass
class _BackwardState:
    first_order: np.ndarray
    momentum_beta: np.ndarray


def _score_trajectory_imm_exact_sparse(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    config: object,
    transition_durations_s: Iterable[float],
    *,
    valid_bin_mask: np.ndarray | None = None,
    return_trajectory: bool = True,
) -> tuple[float, np.ndarray | None, np.ndarray, np.ndarray | None, dict[str, float | int | str]]:
    """Return evidence and posterior diagnostics for exact sparse trajectory IMM."""

    centers = _as_2d_centers(bin_centers)
    if emissions.n_time <= 0:
        raise ValueError("emissions must contain at least one time bin")
    if emissions.n_bins != centers.shape[0]:
        raise ValueError("emissions.n_bins must match bin_centers rows")

    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, emissions.n_bins)
    valid = _valid_indices(valid_mask, emissions.n_bins)
    scaled, offsets = _scaled_emissions(emissions.log_likelihood, valid_bin_mask=valid_mask)
    if valid_mask is not None:
        scaled[:, ~valid_mask] = 0.0

    durations = _coerce_transition_durations(
        transition_durations_s,
        n_time=emissions.n_time,
        fallback_dt=float(emissions.dt),
    )
    reference_dt = float(np.median(durations)) if durations.size else float(emissions.dt)
    max_step_sigma = float(getattr(config, "max_step_sigma", 4.0))
    if not np.isfinite(max_step_sigma) or max_step_sigma <= 0.0:
        raise ValueError("max_step_sigma must be finite and positive")

    diffusion_sigmas = _per_transition_sigmas(
        float(getattr(config, "diffusion_sigma_cm_sqrt_s", 85.0)),
        durations,
    )
    momentum_sigmas = _per_transition_sigmas(
        float(getattr(config, "momentum_sigma_cm_sqrt_s", 85.0)),
        durations,
    )
    initial_sigmas = _per_transition_sigmas(
        float(getattr(config, "momentum_initial_sigma_cm_sqrt_s", 85.0)),
        durations,
    )
    initial_sigma = _median_or_fallback(
        initial_sigmas,
        _per_bin_sigma(
            float(getattr(config, "momentum_initial_sigma_cm_sqrt_s", 85.0)),
            reference_dt,
        ),
    )
    decays = _duration_adjusted_decays(config, durations, float(emissions.dt))
    time_scales = _time_scales(durations)
    tree = cKDTree(centers[valid])
    position_prior = _uniform_position_prior(emissions.n_bins, valid_mask)
    trajectory_imm_mode_stickiness = _trajectory_imm_mode_stickiness(config)
    mode_prior = _trajectory_imm_mode_prior(config)
    mode_transitions = _trajectory_imm_mode_transition_matrices(
        config,
        trajectory_imm_mode_stickiness,
        durations,
    )
    mode_stickiness_per_step = np.asarray(
        [matrix[0, 0] for matrix in mode_transitions], dtype=float
    )
    stationary_transition = _gaussian_transition_matrix(
        centers,
        float(getattr(config, "stationary_sigma_cm", 2.0)),
        max_step_sigma,
        valid_bin_mask=valid_mask,
    )

    first0 = np.zeros((_FIRST_ORDER_MODE_COUNT, emissions.n_bins), dtype=float)
    for mode_index in range(_FIRST_ORDER_MODE_COUNT):
        first0[mode_index] = (
            position_prior
            * scaled[0]
            * mode_prior[mode_index]
        )
    momentum_position0 = position_prior * scaled[0] * mode_prior[_MOMENTUM_MODE_INDEX]
    scale0 = float(first0.sum() + momentum_position0.sum())
    if scale0 <= 0.0 or not np.isfinite(scale0):
        raise ValueError("first emission row has no finite likelihood mass")
    first0 /= scale0
    momentum_position0 /= scale0
    initial_state = _ForwardState(
        first_order=first0,
        momentum_prev=np.empty(0, dtype=int),
        momentum_curr=np.arange(emissions.n_bins, dtype=int),
        momentum_alpha=momentum_position0,
    )

    states: list[_ForwardState] = [initial_state]
    scales = np.zeros(emissions.n_time, dtype=float)
    scales[0] = scale0
    logp = float(np.log(scale0) + offsets[0])
    entry_edge_counts: list[int] = []
    momentum_edge_counts: list[int] = []

    for time_index in range(1, emissions.n_time):
        transition_index = time_index - 1
        diffusion_transition = _gaussian_transition_matrix(
            centers,
            float(diffusion_sigmas[transition_index]),
            max_step_sigma,
            valid_bin_mask=valid_mask,
        )
        state, entry_counts, momentum_counts = _advance_state(
            states[-1],
            source_time_index=time_index - 1,
            centers=centers,
            valid_indices=valid,
            tree=tree,
            emission=scaled[time_index],
            stationary_transition=stationary_transition,
            diffusion_transition=diffusion_transition,
            position_prior=position_prior,
            mode_transition=mode_transitions[transition_index],
            entry_sigma_cm=float(initial_sigmas[transition_index]),
            momentum_sigma_cm=float(momentum_sigmas[transition_index]),
            velocity_decay=float(decays[transition_index]) * float(time_scales[transition_index]),
            max_step_sigma=max_step_sigma,
        )
        scale = float(state.first_order.sum() + state.momentum_alpha.sum())
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        state.first_order /= scale
        state.momentum_alpha /= scale
        states.append(state)
        scales[time_index] = scale
        logp += float(np.log(scale) + offsets[time_index])
        entry_edge_counts.extend(entry_counts)
        momentum_edge_counts.extend(momentum_counts)

    if return_trajectory:
        betas = _backward_states(
            states,
            scaled,
            scales,
            centers=centers,
            valid_indices=valid,
            tree=tree,
            stationary_sigma_cm=float(getattr(config, "stationary_sigma_cm", 2.0)),
            diffusion_sigmas=diffusion_sigmas,
            position_prior=position_prior,
            mode_transitions=mode_transitions,
            entry_sigmas=initial_sigmas,
            momentum_sigmas=momentum_sigmas,
            decays=decays,
            time_scales=time_scales,
            max_step_sigma=max_step_sigma,
            valid_bin_mask=valid_mask,
        )
        trajectory, mode_posterior = _smoothed_posteriors(states, betas, emissions.n_bins)
        trajectory_log_posterior = _as_log_probs(trajectory)
        terminal = trajectory_log_posterior[-1]
        mode_posterior_out: np.ndarray | None = mode_posterior
        mode_posterior_label = "smoothed_heterogeneous_state"
        posterior_entropy = _mean_entropy(trajectory_log_posterior)
    else:
        terminal_position, terminal_mode = _filtered_terminal_posteriors(states[-1], emissions.n_bins)
        terminal = terminal_position
        trajectory_log_posterior = None
        mode_posterior = terminal_mode[None, :]
        mode_posterior_out = None
        mode_posterior_label = "not_returned_evidence_only"
        posterior_entropy = float("nan")

    median_diffusion_sigma = _median_or_fallback(
        diffusion_sigmas,
        _per_bin_sigma(float(getattr(config, "diffusion_sigma_cm_sqrt_s", 85.0)), reference_dt),
    )
    median_momentum_sigma = _median_or_fallback(
        momentum_sigmas,
        _per_bin_sigma(float(getattr(config, "momentum_sigma_cm_sqrt_s", 85.0)), reference_dt),
    )
    median_decay = _median_or_fallback(
        decays,
        float(getattr(config, "momentum_velocity_decay", 0.95)),
    )
    pair_counts = np.asarray(
        [state.momentum_alpha.shape[0] for state in states[1:]],
        dtype=float,
    )
    final_mode_posterior = mode_posterior[-1]
    event_mode_mass = mode_posterior.mean(axis=0)
    trajectory_columns = [1, 2, 3]
    diagnostics: dict[str, float | int | str] = {
        "state_space_trajectory_imm_evidence_support": "exact_full_grid",
        "state_space_trajectory_imm_state_support": "exact_first_order_plus_finite_radius_pair_grid",
        "state_space_trajectory_imm_transition_support": "finite_radius_gaussian",
        "state_space_trajectory_imm_modes": ",".join(_TRAJECTORY_IMM_MODES),
        "state_space_trajectory_imm_mode_count": int(len(_TRAJECTORY_IMM_MODES)),
        "state_space_trajectory_imm_mode_posterior": mode_posterior_label,
        "state_space_trajectory_imm_mode_stickiness": float(
            trajectory_imm_mode_stickiness
        ),
        "state_space_trajectory_imm_switch_tau_s": float(
            getattr(config, "imm_switch_tau_s", 0.0)
        ),
        "state_space_trajectory_imm_mode_stickiness_per_step": _format_float_series(mode_stickiness_per_step),
        "state_space_trajectory_imm_momentum_initial_probability": float(
            mode_prior[_MOMENTUM_MODE_INDEX]
        ),
        "state_space_trajectory_imm_momentum_switch_probability": (
            _optional_float_diagnostic(
                getattr(config, "trajectory_imm_momentum_switch_probability", None)
            )
        ),
        "state_space_trajectory_imm_mean_mode_entropy": _mean_entropy(_as_log_probs(mode_posterior)),
        "state_space_trajectory_imm_mean_posterior_entropy": float(posterior_entropy),
        "state_space_trajectory_imm_terminal_pair_count": (
            int(pair_counts[-1]) if pair_counts.size else 0
        ),
        "state_space_trajectory_imm_mean_pair_count": (
            float(np.mean(pair_counts)) if pair_counts.size else 0.0
        ),
        "state_space_trajectory_imm_max_pair_count": (
            int(np.max(pair_counts)) if pair_counts.size else 0
        ),
        "state_space_trajectory_imm_mean_entry_outgoing_count": (
            float(np.mean(entry_edge_counts)) if entry_edge_counts else 0.0
        ),
        "state_space_trajectory_imm_mean_momentum_outgoing_count": (
            float(np.mean(momentum_edge_counts)) if momentum_edge_counts else 0.0
        ),
        "state_space_trajectory_family_terminal_probability": float(
            final_mode_posterior[trajectory_columns].sum()
        ),
        "state_space_trajectory_family_event_probability": float(
            event_mode_mass[trajectory_columns].sum()
        ),
        "state_space_trajectory_imm_diffusion_transition_sigma_cm": float(median_diffusion_sigma),
        "state_space_momentum_transition_sigma_cm": float(median_momentum_sigma),
        "state_space_momentum_initial_transition_sigma_cm": float(initial_sigma),
        "state_space_momentum_velocity_decay_effective": float(median_decay),
        "state_space_trajectory_imm_diffusion_transition_sigma_cm_per_step": _format_float_series(
            diffusion_sigmas
        ),
        "state_space_momentum_initial_transition_sigma_cm_per_step": _format_float_series(
            initial_sigmas
        ),
        "state_space_momentum_transition_sigma_cm_per_step": _format_float_series(momentum_sigmas),
        "state_space_momentum_velocity_decay_per_step": _format_float_series(decays),
        "state_space_momentum_evidence_support": "exact_full_grid",
        "state_space_momentum_candidate_support": "not_used_exact_sparse_trajectory_imm",
        "state_space_momentum_candidate_selection": "none_exact_sparse_trajectory_imm",
    }
    for mode_index, mode_name in enumerate(_TRAJECTORY_IMM_MODES):
        key = mode_name.replace("-", "_")
        diagnostics[f"state_space_mode_{key}_terminal_probability"] = float(
            final_mode_posterior[mode_index]
        )
        diagnostics[f"state_space_mode_{key}_event_probability"] = float(
            event_mode_mass[mode_index]
        )
        if mode_posterior_out is not None:
            diagnostics[f"state_space_mode_{key}_posterior_over_time"] = _format_float_series(
                mode_posterior_out[:, mode_index]
            )

    return float(logp), trajectory_log_posterior, terminal, mode_posterior_out, diagnostics


def _trajectory_imm_mode_stickiness(config: object) -> float:
    value = getattr(config, "trajectory_imm_mode_stickiness", None)
    if value is None:
        value = getattr(config, "imm_mode_stickiness", 0.95)
    out = float(value)
    if not np.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError("trajectory_imm_mode_stickiness must be in [0, 1]")
    return out


def _trajectory_imm_mode_prior(config: object) -> np.ndarray:
    value = getattr(config, "trajectory_imm_momentum_initial_probability", None)
    if value is None:
        return np.full(len(_TRAJECTORY_IMM_MODES), 1.0 / len(_TRAJECTORY_IMM_MODES), dtype=float)
    momentum_probability = float(value)
    if not np.isfinite(momentum_probability) or not 0.0 <= momentum_probability <= 1.0:
        raise ValueError("trajectory_imm_momentum_initial_probability must be in [0, 1]")
    prior = np.empty(len(_TRAJECTORY_IMM_MODES), dtype=float)
    prior[:_FIRST_ORDER_MODE_COUNT] = (1.0 - momentum_probability) / _FIRST_ORDER_MODE_COUNT
    prior[_MOMENTUM_MODE_INDEX] = momentum_probability
    return prior


def _trajectory_imm_mode_transition_matrix(
    config: object,
    stickiness: float,
) -> np.ndarray:
    momentum_switch = getattr(config, "trajectory_imm_momentum_switch_probability", None)
    if momentum_switch is None:
        return _mode_transition_matrix(len(_TRAJECTORY_IMM_MODES), stickiness)
    momentum_probability = float(momentum_switch)
    if not np.isfinite(momentum_probability) or momentum_probability < 0.0:
        raise ValueError("trajectory_imm_momentum_switch_probability must be finite and nonnegative")
    remaining = 1.0 - float(stickiness)
    if momentum_probability > remaining + 1e-12:
        raise ValueError("trajectory_imm_momentum_switch_probability cannot exceed 1 - stickiness")
    matrix = np.zeros((len(_TRAJECTORY_IMM_MODES), len(_TRAJECTORY_IMM_MODES)), dtype=float)
    np.fill_diagonal(matrix, float(stickiness))
    first_order_other = (remaining - momentum_probability) / (_FIRST_ORDER_MODE_COUNT - 1)
    for src in range(_FIRST_ORDER_MODE_COUNT):
        for dst in range(_FIRST_ORDER_MODE_COUNT):
            if src != dst:
                matrix[src, dst] = first_order_other
        matrix[src, _MOMENTUM_MODE_INDEX] = momentum_probability
    matrix[_MOMENTUM_MODE_INDEX, :_FIRST_ORDER_MODE_COUNT] = remaining / _FIRST_ORDER_MODE_COUNT
    return matrix


def _trajectory_imm_mode_transition_matrices(
    config: object,
    stickiness: float,
    durations: np.ndarray,
) -> list[np.ndarray]:
    """Return one trajectory-IMM mode-transition matrix per transition.

    ``stickiness`` is the legacy per-transition probability.  When
    ``imm_switch_tau_s`` is positive, derive the stickiness for each adjacent
    replay-bin transition from the actual center-to-center duration, matching
    the duration-aware behavior of the other IMM state-space decoders.
    """

    durations = np.asarray(durations, dtype=float)
    tau_s = float(getattr(config, "imm_switch_tau_s", 0.0))
    if not np.isfinite(tau_s) or tau_s < 0.0:
        raise ValueError("imm_switch_tau_s must be finite and nonnegative")
    if tau_s == 0.0:
        transition = _trajectory_imm_mode_transition_matrix(config, stickiness)
        return [transition for _ in range(int(durations.size))]
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("transition durations must be finite and positive")
    return [
        _trajectory_imm_mode_transition_matrix(config, float(np.exp(-float(duration) / tau_s)))
        for duration in durations
    ]


def _optional_float_diagnostic(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _advance_state(
    previous: _ForwardState,
    *,
    source_time_index: int,
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    emission: np.ndarray,
    stationary_transition: csr_matrix,
    diffusion_transition: csr_matrix,
    position_prior: np.ndarray,
    mode_transition: np.ndarray,
    entry_sigma_cm: float,
    momentum_sigma_cm: float,
    velocity_decay: float,
    max_step_sigma: float,
) -> tuple[_ForwardState, list[int], list[int]]:
    n_bins = centers.shape[0]
    source_positions = _source_position_masses(previous, source_time_index, n_bins)
    first_pred = np.zeros((_FIRST_ORDER_MODE_COUNT, n_bins), dtype=float)

    for dst_index, dst_mode in enumerate(_TRAJECTORY_IMM_MODES[:_FIRST_ORDER_MODE_COUNT]):
        combined = np.zeros(n_bins, dtype=float)
        for src_index, source in enumerate(source_positions):
            combined += mode_transition[src_index, dst_index] * _apply_first_order_transition(
                source,
                mode=dst_mode,
                stationary_transition=stationary_transition,
                diffusion_transition=diffusion_transition,
                position_prior=position_prior,
            )
        first_pred[dst_index] = combined * emission

    prev_parts: list[np.ndarray] = []
    curr_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    entry_counts: list[int] = []
    momentum_counts: list[int] = []
    dst_index = _MOMENTUM_MODE_INDEX
    for src_index, source in enumerate(source_positions[:_FIRST_ORDER_MODE_COUNT]):
        prev, curr, values, counts = _advance_position_to_momentum(
            source * mode_transition[src_index, dst_index],
            centers,
            valid_indices,
            tree,
            emission,
            sigma_cm=entry_sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        if values.size:
            prev_parts.append(prev)
            curr_parts.append(curr)
            value_parts.append(values)
        entry_counts.extend(counts)

    if source_time_index == 0:
        prev, curr, values, counts = _advance_position_to_momentum(
            source_positions[_MOMENTUM_MODE_INDEX] * mode_transition[_MOMENTUM_MODE_INDEX, dst_index],
            centers,
            valid_indices,
            tree,
            emission,
            sigma_cm=entry_sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        if values.size:
            prev_parts.append(prev)
            curr_parts.append(curr)
            value_parts.append(values)
        entry_counts.extend(counts)
    else:
        prev, curr, values, counts = _advance_momentum_to_momentum(
            previous.momentum_prev,
            previous.momentum_curr,
            previous.momentum_alpha * mode_transition[_MOMENTUM_MODE_INDEX, dst_index],
            centers,
            valid_indices,
            tree,
            emission,
            sigma_cm=momentum_sigma_cm,
            velocity_decay=velocity_decay,
            max_step_sigma=max_step_sigma,
        )
        if values.size:
            prev_parts.append(prev)
            curr_parts.append(curr)
            value_parts.append(values)
        momentum_counts.extend(counts)

    momentum_prev, momentum_curr, momentum_alpha = _coalesce_pairs(
        prev_parts,
        curr_parts,
        value_parts,
        n_bins,
    )
    return (
        _ForwardState(
            first_order=first_pred,
            momentum_prev=momentum_prev,
            momentum_curr=momentum_curr,
            momentum_alpha=momentum_alpha,
        ),
        entry_counts,
        momentum_counts,
    )


def _source_position_masses(
    previous: _ForwardState,
    source_time_index: int,
    n_bins: int,
) -> list[np.ndarray]:
    sources = [previous.first_order[index] for index in range(_FIRST_ORDER_MODE_COUNT)]
    if source_time_index == 0:
        sources.append(previous.momentum_alpha)
        return sources
    momentum = np.zeros(n_bins, dtype=float)
    np.add.at(momentum, previous.momentum_curr, previous.momentum_alpha)
    sources.append(momentum)
    return sources


def _apply_first_order_transition(
    source: np.ndarray,
    *,
    mode: str,
    stationary_transition: csr_matrix,
    diffusion_transition: csr_matrix,
    position_prior: np.ndarray,
) -> np.ndarray:
    if mode == "stationary":
        return np.asarray(stationary_transition @ source, dtype=float)
    if mode == "diffusion":
        return np.asarray(diffusion_transition @ source, dtype=float)
    if mode == "fragmented":
        return position_prior * float(source.sum())
    raise ValueError(f"unsupported first-order destination mode: {mode}")


def _advance_position_to_momentum(
    source_position: np.ndarray,
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    emission: np.ndarray,
    *,
    sigma_cm: float,
    max_step_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    prev_parts: list[np.ndarray] = []
    curr_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_counts: list[int] = []
    for src in valid_indices:
        source_mass = float(source_position[int(src)])
        if source_mass <= 0.0:
            continue
        dst, weights = _finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            centers[int(src)],
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        values = source_mass * weights * emission[dst]
        keep = values > 0.0
        edge_counts.append(int(dst.shape[0]))
        if not np.any(keep):
            continue
        prev_parts.append(np.full(int(np.sum(keep)), int(src), dtype=int))
        curr_parts.append(dst[keep])
        value_parts.append(values[keep])
    return (*_coalesce_pairs(prev_parts, curr_parts, value_parts, centers.shape[0]), edge_counts)


def _advance_momentum_to_momentum(
    prev: np.ndarray,
    curr: np.ndarray,
    alpha: np.ndarray,
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    emission: np.ndarray,
    *,
    sigma_cm: float,
    velocity_decay: float,
    max_step_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    prev_parts: list[np.ndarray] = []
    curr_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_counts: list[int] = []
    for src_prev, src_curr, source_mass in zip(prev, curr, alpha, strict=True):
        if source_mass <= 0.0:
            continue
        prediction = centers[int(src_curr)] + float(velocity_decay) * (
            centers[int(src_curr)] - centers[int(src_prev)]
        )
        dst, weights = _finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            prediction,
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        values = float(source_mass) * weights * emission[dst]
        keep = values > 0.0
        edge_counts.append(int(dst.shape[0]))
        if not np.any(keep):
            continue
        prev_parts.append(np.full(int(np.sum(keep)), int(src_curr), dtype=int))
        curr_parts.append(dst[keep])
        value_parts.append(values[keep])
    return (*_coalesce_pairs(prev_parts, curr_parts, value_parts, centers.shape[0]), edge_counts)


def _backward_states(
    states: list[_ForwardState],
    scaled: np.ndarray,
    scales: np.ndarray,
    *,
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    stationary_sigma_cm: float,
    diffusion_sigmas: np.ndarray,
    position_prior: np.ndarray,
    mode_transitions: list[np.ndarray],
    entry_sigmas: np.ndarray,
    momentum_sigmas: np.ndarray,
    decays: np.ndarray,
    time_scales: np.ndarray,
    max_step_sigma: float,
    valid_bin_mask: np.ndarray | None,
) -> list[_BackwardState]:
    n_time = len(states)
    n_bins = centers.shape[0]
    betas: list[_BackwardState] = [
        _BackwardState(
            first_order=np.zeros((_FIRST_ORDER_MODE_COUNT, n_bins), dtype=float),
            momentum_beta=np.zeros_like(state.momentum_alpha, dtype=float),
        )
        for state in states
    ]
    betas[-1].first_order.fill(1.0)
    betas[-1].momentum_beta.fill(1.0)

    stationary_transition = _gaussian_transition_matrix(
        centers,
        stationary_sigma_cm,
        max_step_sigma,
        valid_bin_mask=valid_bin_mask,
    )
    for time_index in range(n_time - 1, 0, -1):
        destination = states[time_index]
        destination_beta = betas[time_index]
        source = states[time_index - 1]
        transition_index = time_index - 1
        mode_transition = mode_transitions[transition_index]
        diffusion_transition = _gaussian_transition_matrix(
            centers,
            float(diffusion_sigmas[transition_index]),
            max_step_sigma,
            valid_bin_mask=valid_bin_mask,
        )
        dest_first_values = [
            scaled[time_index] * destination_beta.first_order[index]
            for index in range(_FIRST_ORDER_MODE_COUNT)
        ]
        dest_momentum_values = (
            scaled[time_index][destination.momentum_curr] * destination_beta.momentum_beta
        )
        source_first = np.zeros_like(source.first_order)
        source_momentum_position = np.zeros(n_bins, dtype=float)
        for src_index in range(_FIRST_ORDER_MODE_COUNT):
            total = np.zeros(n_bins, dtype=float)
            for dst_index, dst_mode in enumerate(_TRAJECTORY_IMM_MODES[:_FIRST_ORDER_MODE_COUNT]):
                total += mode_transition[src_index, dst_index] * _backward_first_order_transition(
                    dest_first_values[dst_index],
                    mode=dst_mode,
                    stationary_transition=stationary_transition,
                    diffusion_transition=diffusion_transition,
                    position_prior=position_prior,
                    valid_bin_mask=valid_bin_mask,
                )
            total += mode_transition[src_index, _MOMENTUM_MODE_INDEX] * _backward_position_to_momentum(
                destination.momentum_prev,
                destination.momentum_curr,
                dest_momentum_values,
                centers,
                valid_indices,
                tree,
                sigma_cm=float(entry_sigmas[transition_index]),
                max_step_sigma=max_step_sigma,
            )
            source_first[src_index] = total / scales[time_index]

        for dst_index, dst_mode in enumerate(_TRAJECTORY_IMM_MODES[:_FIRST_ORDER_MODE_COUNT]):
            source_momentum_position += mode_transition[_MOMENTUM_MODE_INDEX, dst_index] * _backward_first_order_transition(
                dest_first_values[dst_index],
                mode=dst_mode,
                stationary_transition=stationary_transition,
                diffusion_transition=diffusion_transition,
                position_prior=position_prior,
                valid_bin_mask=valid_bin_mask,
            )

        if time_index - 1 == 0:
            source_momentum_position += mode_transition[_MOMENTUM_MODE_INDEX, _MOMENTUM_MODE_INDEX] * _backward_position_to_momentum(
                destination.momentum_prev,
                destination.momentum_curr,
                dest_momentum_values,
                centers,
                valid_indices,
                tree,
                sigma_cm=float(entry_sigmas[transition_index]),
                max_step_sigma=max_step_sigma,
            )
            betas[time_index - 1] = _BackwardState(
                first_order=source_first,
                momentum_beta=source_momentum_position / scales[time_index],
            )
            continue

        first_order_from_pair = np.zeros(n_bins, dtype=float)
        if source.momentum_alpha.size:
            for dst_index, dst_mode in enumerate(_TRAJECTORY_IMM_MODES[:_FIRST_ORDER_MODE_COUNT]):
                first_order_from_pair += mode_transition[_MOMENTUM_MODE_INDEX, dst_index] * _backward_first_order_transition(
                    dest_first_values[dst_index],
                    mode=dst_mode,
                    stationary_transition=stationary_transition,
                    diffusion_transition=diffusion_transition,
                    position_prior=position_prior,
                    valid_bin_mask=valid_bin_mask,
                )
            momentum_to_momentum = _backward_momentum_to_momentum(
                source.momentum_prev,
                source.momentum_curr,
                destination.momentum_prev,
                destination.momentum_curr,
                dest_momentum_values,
                centers,
                valid_indices,
                tree,
                sigma_cm=float(momentum_sigmas[transition_index]),
                velocity_decay=float(decays[transition_index]) * float(time_scales[transition_index]),
                max_step_sigma=max_step_sigma,
            )
            source_momentum_beta = (
                first_order_from_pair[source.momentum_curr]
                + mode_transition[_MOMENTUM_MODE_INDEX, _MOMENTUM_MODE_INDEX] * momentum_to_momentum
            ) / scales[time_index]
        else:
            source_momentum_beta = np.empty(0, dtype=float)
        betas[time_index - 1] = _BackwardState(
            first_order=source_first,
            momentum_beta=source_momentum_beta,
        )
    return betas


def _backward_first_order_transition(
    values: np.ndarray,
    *,
    mode: str,
    stationary_transition: csr_matrix,
    diffusion_transition: csr_matrix,
    position_prior: np.ndarray,
    valid_bin_mask: np.ndarray | None,
) -> np.ndarray:
    if mode == "stationary":
        return np.asarray(stationary_transition.T @ values, dtype=float)
    if mode == "diffusion":
        return np.asarray(diffusion_transition.T @ values, dtype=float)
    if mode == "fragmented":
        out = np.zeros_like(values, dtype=float)
        if valid_bin_mask is None:
            out.fill(float(values.sum()) / values.shape[0])
        else:
            out[valid_bin_mask] = float(values[valid_bin_mask].sum()) / int(np.sum(valid_bin_mask))
        return out
    raise ValueError(f"unsupported first-order destination mode: {mode}")


def _backward_position_to_momentum(
    dest_prev: np.ndarray,
    dest_curr: np.ndarray,
    dest_values: np.ndarray,
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    *,
    sigma_cm: float,
    max_step_sigma: float,
) -> np.ndarray:
    out = np.zeros(centers.shape[0], dtype=float)
    if dest_values.size == 0:
        return out
    lookup = _pair_value_lookup(dest_prev, dest_curr, dest_values, centers.shape[0])
    for src in valid_indices:
        dst, weights = _finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            centers[int(src)],
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        out[int(src)] = float(np.sum(weights * _lookup_pair_values(lookup, int(src), dst, centers.shape[0])))
    return out


def _backward_momentum_to_momentum(
    source_prev: np.ndarray,
    source_curr: np.ndarray,
    dest_prev: np.ndarray,
    dest_curr: np.ndarray,
    dest_values: np.ndarray,
    centers: np.ndarray,
    valid_indices: np.ndarray,
    tree: cKDTree,
    *,
    sigma_cm: float,
    velocity_decay: float,
    max_step_sigma: float,
) -> np.ndarray:
    out = np.zeros(source_prev.shape[0], dtype=float)
    if source_prev.size == 0 or dest_values.size == 0:
        return out
    lookup = _pair_value_lookup(dest_prev, dest_curr, dest_values, centers.shape[0])
    for row_index, (prev, curr) in enumerate(zip(source_prev, source_curr, strict=True)):
        prediction = centers[int(curr)] + float(velocity_decay) * (
            centers[int(curr)] - centers[int(prev)]
        )
        dst, weights = _finite_gaussian_row(
            centers,
            valid_indices,
            tree,
            prediction,
            sigma_cm=sigma_cm,
            max_step_sigma=max_step_sigma,
        )
        out[row_index] = float(
            np.sum(weights * _lookup_pair_values(lookup, int(curr), dst, centers.shape[0]))
        )
    return out


def _pair_value_lookup(
    prev: np.ndarray,
    curr: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(prev, dtype=np.int64) * int(n_bins) + np.asarray(curr, dtype=np.int64)
    order = np.argsort(keys, kind="stable")
    return keys[order], np.asarray(values, dtype=float)[order]


def _lookup_pair_values(
    lookup: tuple[np.ndarray, np.ndarray],
    prev: int,
    curr_values: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    keys, values = lookup
    query = int(prev) * int(n_bins) + np.asarray(curr_values, dtype=np.int64)
    positions = np.searchsorted(keys, query)
    out = np.zeros(query.shape[0], dtype=float)
    in_bounds = positions < keys.shape[0]
    if np.any(in_bounds):
        rows = np.flatnonzero(in_bounds)
        matched = keys[positions[rows]] == query[rows]
        out[rows[matched]] = values[positions[rows[matched]]]
    return out


def _smoothed_posteriors(
    states: list[_ForwardState],
    betas: list[_BackwardState],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros((len(states), n_bins), dtype=float)
    mode = np.zeros((len(states), len(_TRAJECTORY_IMM_MODES)), dtype=float)
    for time_index, (state, beta) in enumerate(zip(states, betas, strict=True)):
        first_gamma = state.first_order * beta.first_order
        momentum_gamma = state.momentum_alpha * beta.momentum_beta
        total = float(first_gamma.sum() + momentum_gamma.sum())
        if total <= 0.0 or not np.isfinite(total):
            first_gamma = state.first_order
            momentum_gamma = state.momentum_alpha
            total = float(first_gamma.sum() + momentum_gamma.sum())
        first_gamma = first_gamma / total
        momentum_gamma = momentum_gamma / total
        position[time_index] = first_gamma.sum(axis=0)
        mode[time_index, :_FIRST_ORDER_MODE_COUNT] = first_gamma.sum(axis=1)
        if time_index == 0:
            position[time_index] += momentum_gamma
            mode[time_index, _MOMENTUM_MODE_INDEX] = float(momentum_gamma.sum())
        else:
            np.add.at(position[time_index], state.momentum_curr, momentum_gamma)
            mode[time_index, _MOMENTUM_MODE_INDEX] = float(momentum_gamma.sum())
        position_total = float(position[time_index].sum())
        mode_total = float(mode[time_index].sum())
        if position_total > 0.0:
            position[time_index] /= position_total
        if mode_total > 0.0:
            mode[time_index] /= mode_total
    return position, mode


def _filtered_terminal_posteriors(
    state: _ForwardState,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    position = state.first_order.sum(axis=0)
    mode = np.zeros(len(_TRAJECTORY_IMM_MODES), dtype=float)
    mode[:_FIRST_ORDER_MODE_COUNT] = state.first_order.sum(axis=1)
    # Later states can have no momentum pair rows when momentum prior/switch mass is zero.
    # Only the initial position-only momentum state stores one dense value per spatial bin.
    if state.momentum_alpha.size:
        if state.momentum_prev.size == 0:
            if state.momentum_alpha.shape != (n_bins,):
                raise ValueError(
                    "position-only momentum posterior must contain one value per spatial bin"
                )
            position += state.momentum_alpha
        else:
            np.add.at(position, state.momentum_curr, state.momentum_alpha)
    mode[_MOMENTUM_MODE_INDEX] = float(state.momentum_alpha.sum())
    position_total = float(position.sum())
    mode_total = float(mode.sum())
    if position_total <= 0.0 or mode_total <= 0.0:
        raise ValueError("terminal posterior has no finite mass")
    return _as_log_probs((position / position_total)[None, :])[0], mode / mode_total


def _median_or_fallback(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(fallback) if arr.size == 0 else float(np.median(arr))

"""Expose full-event smoothed trajectories for candidate kinematic models."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
from scipy.special import logsumexp

_STATIC_PATCHED_ATTR = "_candidate_static_smoothed_trajectory_applied"
_IMM_PATCHED_ATTR = "_candidate_imm_smoothed_trajectory_applied"
_MODES = ("stationary", "diffusion", "momentum", "jump")


def apply_candidate_kinematic_smoothing_patch() -> None:
    """Install forward-backward trajectory smoothing for candidate models."""

    from . import models

    current_static = models.CandidateKinematicModel._score_static_pair
    if not getattr(current_static, _STATIC_PATCHED_ATTR, False):

        @wraps(current_static)
        def score_static_pair_with_smoothing(
            self: Any,
            emissions: Any,
            bin_centers: np.ndarray,
            candidates: list[np.ndarray],
            mode: str,
        ):
            result = current_static(self, emissions, bin_centers, candidates, mode)
            logp, masses, terminal, _trajectory = result
            if not np.isfinite(float(logp)):
                return result
            trajectory = _smoothed_static_trajectory(
                self,
                emissions,
                bin_centers,
                candidates,
                mode,
            )
            trajectory[-1] = np.asarray(terminal, dtype=float)
            return logp, masses, terminal, trajectory

        setattr(score_static_pair_with_smoothing, _STATIC_PATCHED_ATTR, True)
        setattr(
            score_static_pair_with_smoothing,
            "__hipporeplayimm_original__",
            current_static,
        )
        models.CandidateKinematicModel._score_static_pair = score_static_pair_with_smoothing

    current_imm = models.CandidateKinematicModel._score_imm
    if not getattr(current_imm, _IMM_PATCHED_ATTR, False):

        @wraps(current_imm)
        def score_imm_with_smoothing(
            self: Any,
            emissions: Any,
            bin_centers: np.ndarray,
            candidates: list[np.ndarray],
        ):
            result = current_imm(self, emissions, bin_centers, candidates)
            logp, masses, terminal, _trajectory = result
            if not np.isfinite(float(logp)):
                return result
            trajectory = _smoothed_imm_trajectory(
                self,
                emissions,
                bin_centers,
                candidates,
            )
            trajectory[-1] = np.asarray(terminal, dtype=float)
            return logp, masses, terminal, trajectory

        setattr(score_imm_with_smoothing, _IMM_PATCHED_ATTR, True)
        setattr(
            score_imm_with_smoothing,
            "__hipporeplayimm_original__",
            current_imm,
        )
        models.CandidateKinematicModel._score_imm = score_imm_with_smoothing


def _smoothed_static_trajectory(
    model: Any,
    emissions: Any,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
    mode: str,
) -> np.ndarray:
    from . import models

    forward = _static_forward_messages(
        model,
        emissions,
        bin_centers,
        candidates,
        mode,
        models,
    )
    backward: list[np.ndarray] = [np.empty(0)] * len(forward)
    backward[-1] = np.zeros_like(forward[-1])
    for pair_index in range(len(forward) - 2, -1, -1):
        backward[pair_index] = _static_backward_message(
            model,
            mode,
            emissions.log_likelihood[pair_index + 2],
            bin_centers,
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[pair_index + 2],
            backward[pair_index + 1],
            models,
        )
    return _trajectory_from_pair_messages(
        forward,
        backward,
        candidates,
        int(emissions.n_bins),
        models,
    )


def _static_forward_messages(
    model: Any,
    emissions: Any,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
    mode: str,
    models: Any,
) -> list[np.ndarray]:
    alpha = models._init_pair_log_alpha(
        emissions,
        candidates[0],
        candidates[1],
        bin_centers,
        mode=mode,
        stationary_sigma_cm=model.stationary_sigma_cm,
        diffusion_sigma_cm=model.diffusion_sigma_cm,
        momentum_sigma_cm=model.momentum_sigma_cm,
    )
    forward = [alpha]
    for time_index in range(2, int(emissions.n_time)):
        alpha = models._advance_pair_log_alpha(
            alpha,
            candidates[time_index - 2],
            candidates[time_index - 1],
            candidates[time_index],
            emissions.log_likelihood[time_index, candidates[time_index]],
            bin_centers,
            mode=mode,
            stationary_sigma_cm=model.stationary_sigma_cm,
            diffusion_sigma_cm=model.diffusion_sigma_cm,
            momentum_sigma_cm=model.momentum_sigma_cm,
            velocity_decay=model.velocity_decay,
        )
        forward.append(alpha)
    return forward


def _static_backward_message(
    model: Any,
    mode: str,
    next_emission_row: np.ndarray,
    bin_centers: np.ndarray,
    previous_previous: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    next_beta: np.ndarray,
    models: Any,
) -> np.ndarray:
    beta = np.empty((len(previous_previous), len(previous)), dtype=float)
    current_emission = np.asarray(next_emission_row, dtype=float)[current]
    for previous_column in range(len(previous)):
        log_kernel = _next_transition_log_kernel(
            model,
            mode,
            bin_centers,
            previous_previous,
            previous,
            previous_column,
            current,
            models,
        )
        future = current_emission[None, :] + next_beta[previous_column][None, :]
        beta[:, previous_column] = logsumexp(log_kernel + future, axis=1)
    return beta


def _smoothed_imm_trajectory(
    model: Any,
    emissions: Any,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
) -> np.ndarray:
    from . import models

    transition_modes = models._mode_transition_matrix(len(_MODES), model.mode_stickiness)
    log_transition_modes = np.full(transition_modes.shape, -np.inf, dtype=float)
    positive = transition_modes > 0.0
    log_transition_modes[positive] = np.log(transition_modes[positive])

    forward = _imm_forward_messages(
        model,
        emissions,
        bin_centers,
        candidates,
        log_transition_modes,
        models,
    )
    backward: list[np.ndarray] = [np.empty(0)] * len(forward)
    backward[-1] = np.zeros_like(forward[-1])
    for pair_index in range(len(forward) - 2, -1, -1):
        backward[pair_index] = _imm_backward_message(
            model,
            emissions.log_likelihood[pair_index + 2],
            bin_centers,
            candidates[pair_index],
            candidates[pair_index + 1],
            candidates[pair_index + 2],
            backward[pair_index + 1],
            log_transition_modes,
            models,
        )
    return _trajectory_from_pair_messages(
        forward,
        backward,
        candidates,
        int(emissions.n_bins),
        models,
    )


def _imm_forward_messages(
    model: Any,
    emissions: Any,
    bin_centers: np.ndarray,
    candidates: list[np.ndarray],
    log_transition_modes: np.ndarray,
    models: Any,
) -> list[np.ndarray]:
    by_mode = [
        models._init_pair_log_alpha(
            emissions,
            candidates[0],
            candidates[1],
            bin_centers,
            mode=mode,
            stationary_sigma_cm=model.stationary_sigma_cm,
            diffusion_sigma_cm=model.diffusion_sigma_cm,
            momentum_sigma_cm=model.momentum_sigma_cm,
        )
        for mode in _MODES
    ]
    alpha = np.stack(by_mode, axis=0) - np.log(len(_MODES))
    forward = [alpha]
    for time_index in range(2, int(emissions.n_time)):
        next_alpha = []
        for destination_mode_index, destination_mode in enumerate(_MODES):
            mixed_previous = logsumexp(
                alpha + log_transition_modes[:, destination_mode_index, None, None],
                axis=0,
            )
            next_alpha.append(
                models._advance_pair_log_alpha(
                    mixed_previous,
                    candidates[time_index - 2],
                    candidates[time_index - 1],
                    candidates[time_index],
                    emissions.log_likelihood[time_index, candidates[time_index]],
                    bin_centers,
                    mode=destination_mode,
                    stationary_sigma_cm=model.stationary_sigma_cm,
                    diffusion_sigma_cm=model.diffusion_sigma_cm,
                    momentum_sigma_cm=model.momentum_sigma_cm,
                    velocity_decay=model.velocity_decay,
                )
            )
        alpha = np.stack(next_alpha, axis=0)
        forward.append(alpha)
    return forward


def _imm_backward_message(
    model: Any,
    next_emission_row: np.ndarray,
    bin_centers: np.ndarray,
    previous_previous: np.ndarray,
    previous: np.ndarray,
    current: np.ndarray,
    next_beta: np.ndarray,
    log_transition_modes: np.ndarray,
    models: Any,
) -> np.ndarray:
    beta = np.full(
        (len(_MODES), len(previous_previous), len(previous)),
        -np.inf,
        dtype=float,
    )
    current_emission = np.asarray(next_emission_row, dtype=float)[current]
    for source_mode_index in range(len(_MODES)):
        for previous_column in range(len(previous)):
            destination_masses = []
            for destination_mode_index, destination_mode in enumerate(_MODES):
                log_kernel = _next_transition_log_kernel(
                    model,
                    destination_mode,
                    bin_centers,
                    previous_previous,
                    previous,
                    previous_column,
                    current,
                    models,
                )
                future = (
                    current_emission[None, :]
                    + next_beta[destination_mode_index, previous_column][None, :]
                )
                destination_masses.append(
                    log_transition_modes[source_mode_index, destination_mode_index]
                    + logsumexp(log_kernel + future, axis=1)
                )
            beta[source_mode_index, :, previous_column] = logsumexp(
                np.stack(destination_masses, axis=0),
                axis=0,
            )
    return beta


def _next_transition_log_kernel(
    model: Any,
    mode: str,
    bin_centers: np.ndarray,
    previous_previous: np.ndarray,
    previous: np.ndarray,
    previous_column: int,
    current: np.ndarray,
    models: Any,
) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    if mode == "jump":
        return np.full(
            (len(previous_previous), len(current)),
            -np.log(centers.shape[0]),
            dtype=float,
        )

    previous_center = centers[previous[previous_column]]
    if mode == "momentum":
        predictions = previous_center[None, :] + model.velocity_decay * (
            previous_center[None, :] - centers[previous_previous]
        )
        sigma = model.momentum_sigma_cm
    elif mode == "stationary":
        predictions = np.repeat(
            previous_center[None, :],
            len(previous_previous),
            axis=0,
        )
        sigma = model.stationary_sigma_cm
    elif mode == "diffusion":
        predictions = np.repeat(
            previous_center[None, :],
            len(previous_previous),
            axis=0,
        )
        sigma = model.diffusion_sigma_cm
    else:
        raise ValueError(f"Unknown kinematic mode: {mode}")

    return models._full_grid_normalized_pairwise_gaussian_log_prob(
        predictions,
        centers[current],
        centers,
        sigma,
    )


def _trajectory_from_pair_messages(
    forward: list[np.ndarray],
    backward: list[np.ndarray],
    candidates: list[np.ndarray],
    n_bins: int,
    models: Any,
) -> np.ndarray:
    first_joint = forward[0] + backward[0]
    trajectory = [
        models._pair_previous_posterior(first_joint, candidates[0], n_bins)
    ]
    trajectory.extend(
        models._pair_terminal_posterior(alpha + beta, candidates[index + 1], n_bins)
        for index, (alpha, beta) in enumerate(zip(forward, backward, strict=True))
    )
    return np.stack(trajectory, axis=0)


__all__ = ["apply_candidate_kinematic_smoothing_patch"]

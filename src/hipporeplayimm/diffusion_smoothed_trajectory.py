"""Return full-event smoothed trajectories for the core diffusion model."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np
from scipy.special import logsumexp

_PATCHED_ATTR = "_diffusion_smoothed_trajectory_posterior_applied"


def apply_diffusion_smoothed_trajectory_patch() -> None:
    """Install forward-backward trajectory smoothing on ``DiffusionModel``."""

    from . import models

    current = models.DiffusionModel.score
    if getattr(current, _PATCHED_ATTR, False):
        return

    @wraps(current)
    def score_with_smoothed_trajectory(
        self: Any,
        emissions: Any,
        bin_centers: Any,
    ):
        result = current(self, emissions, bin_centers)
        trajectory = result.trajectory_log_posterior
        terminal = result.terminal_log_posterior
        if trajectory is None or terminal is None or int(result.n_time) <= 1:
            return result

        centers = models._validate_score_inputs(emissions, bin_centers)
        transition = models._log_transition_matrix(
            centers,
            sigma_cm=self.sigma_cm,
            max_step_sigma=self.max_step_sigma,
        )
        result.trajectory_log_posterior = _smooth_diffusion_trajectory(
            np.asarray(trajectory, dtype=float),
            np.asarray(emissions.log_likelihood, dtype=float),
            transition,
            terminal_log_posterior=np.asarray(terminal, dtype=float),
        )
        return result

    setattr(score_with_smoothed_trajectory, _PATCHED_ATTR, True)
    setattr(
        score_with_smoothed_trajectory,
        "__hipporeplayimm_original__",
        current,
    )
    models.DiffusionModel.score = score_with_smoothed_trajectory


def _smooth_diffusion_trajectory(
    filtering_log_posterior: np.ndarray,
    log_likelihood: np.ndarray,
    transition: list[tuple[np.ndarray, np.ndarray]],
    *,
    terminal_log_posterior: np.ndarray,
) -> np.ndarray:
    """Return ``p(x_t | y_0:T)`` from filtering marginals and transitions."""

    filtered = np.asarray(filtering_log_posterior, dtype=float)
    emissions = np.asarray(log_likelihood, dtype=float)
    terminal = np.asarray(terminal_log_posterior, dtype=float)
    if filtered.ndim != 2 or emissions.shape != filtered.shape:
        raise ValueError("diffusion trajectory and emissions must have matching 2D shapes")
    if terminal.shape != (filtered.shape[1],):
        raise ValueError("diffusion terminal posterior must match the spatial grid")
    if len(transition) != filtered.shape[1]:
        raise ValueError("diffusion transition must contain one row per spatial bin")

    smoothed = np.empty_like(filtered)
    smoothed[-1] = terminal
    log_beta = np.zeros(filtered.shape[1], dtype=float)

    for time_index in range(filtered.shape[0] - 2, -1, -1):
        next_log_factor = emissions[time_index + 1] + log_beta
        log_beta = _backward_log_matvec(next_log_factor, transition)
        row = filtered[time_index] + log_beta
        normalizer = float(logsumexp(row))
        if not np.isfinite(normalizer):
            raise ValueError("diffusion model has no finite smoothed path mass")
        smoothed[time_index] = row - normalizer

    return smoothed


def _backward_log_matvec(
    next_log_factor: np.ndarray,
    transition: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Apply the transpose transition in log space for a backward message."""

    values = np.asarray(next_log_factor, dtype=float)
    output = np.full(len(transition), -np.inf, dtype=float)
    for source, (destination_indices, log_weights) in enumerate(transition):
        destinations = np.asarray(destination_indices, dtype=int)
        weights = np.asarray(log_weights, dtype=float)
        output[source] = float(logsumexp(weights + values[destinations]))
    return output


__all__ = ["apply_diffusion_smoothed_trajectory_patch"]

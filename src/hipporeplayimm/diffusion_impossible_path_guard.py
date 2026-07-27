"""Preserve exact support in the core sparse diffusion recursion."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATH_GUARD_ATTR = "_diffusion_impossible_path_guard_applied"
_SPARSE_MATVEC_ATTR = "_diffusion_exact_sparse_matvec_applied"


def apply_diffusion_impossible_path_guard_patch() -> None:
    """Reject impossible paths and prevent finite sentinel mass from leaking."""

    from . import models

    _patch_exact_sparse_log_matvec(models)

    current = models.DiffusionModel.score
    if getattr(current, _PATH_GUARD_ATTR, False):
        return

    @wraps(current)
    def score_with_path_guard(
        self: Any,
        emissions: Any,
        bin_centers: Any,
    ):
        centers = models._validate_score_inputs(emissions, bin_centers)
        transition = models._log_transition_matrix(
            centers,
            sigma_cm=self.sigma_cm,
            max_step_sigma=self.max_step_sigma,
        )
        _validate_finite_diffusion_path(emissions.log_likelihood, transition)
        return current(self, emissions, centers)

    setattr(score_with_path_guard, _PATH_GUARD_ATTR, True)
    setattr(score_with_path_guard, "__hipporeplayimm_original__", current)
    models.DiffusionModel.score = score_with_path_guard


def _patch_exact_sparse_log_matvec(models: Any) -> None:
    """Use negative infinity, not a finite floor, for unreachable states."""

    current = models._log_sparse_matvec
    if getattr(current, _SPARSE_MATVEC_ATTR, False):
        return

    @wraps(current)
    def exact_sparse_log_matvec(
        log_alpha: Any,
        transition: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        alpha = np.asarray(log_alpha, dtype=float)
        result = np.full(alpha.shape, -np.inf, dtype=float)
        for source, (destinations, log_weights) in enumerate(transition):
            values = alpha[source] + np.asarray(log_weights, dtype=float)
            for destination, value in zip(destinations, values):
                index = int(destination)
                result[index] = np.logaddexp(result[index], value)
        return result

    setattr(exact_sparse_log_matvec, _SPARSE_MATVEC_ATTR, True)
    setattr(exact_sparse_log_matvec, "__hipporeplayimm_original__", current)
    models._log_sparse_matvec = exact_sparse_log_matvec


def _validate_finite_diffusion_path(
    log_likelihood: Any,
    transition: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Require one finite-support path through every emission time bin."""

    values = np.asarray(log_likelihood, dtype=float)
    reachable = np.isfinite(values[0])
    for time_index in range(1, values.shape[0]):
        next_reachable = np.zeros(reachable.shape, dtype=bool)
        for source in np.flatnonzero(reachable):
            destinations, log_weights = transition[int(source)]
            finite_destinations = np.asarray(destinations)[np.isfinite(log_weights)]
            next_reachable[finite_destinations] = True
        reachable = next_reachable & np.isfinite(values[time_index])
        if not np.any(reachable):
            raise ValueError("diffusion model has no finite path mass")


__all__ = ["apply_diffusion_impossible_path_guard_patch"]

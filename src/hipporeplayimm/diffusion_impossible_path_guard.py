"""Keep core replay-model posterior outputs consistent with model support."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCHED_ATTR = "_diffusion_impossible_posterior_guard_applied"
_CANDIDATE_PATCHED_ATTR = "_candidate_impossible_posterior_guard_applied"
_STATIONARY_PATCHED_ATTR = "_stationary_smoothed_trajectory_posterior_applied"
_SPARSE_MATVEC_PATCHED_ATTR = "_diffusion_exact_sparse_matvec_support_applied"
_PAIR_POSTERIOR_PATCHED_ATTR = "_candidate_exact_pair_posterior_support_applied"
_NO_FINITE_PATH = "no_finite_path"
_POSTERIOR_DIAGNOSTIC_KEYS = (
    "decoded_endpoint_x",
    "decoded_endpoint_y",
    "decoded_map_x",
    "decoded_map_y",
    "decoded_map_bin",
    "terminal_posterior_entropy",
)


def apply_diffusion_impossible_path_guard_patch() -> None:
    """Install exact sparse support and posterior guards for core replay models."""

    from . import models

    _patch_exact_sparse_log_matvec(models)
    _patch_exact_pair_posteriors(models)
    _patch_stationary_trajectory_posterior(models.StationaryModel)
    _patch_impossible_posterior_guard(
        models.DiffusionModel,
        marker=_PATCHED_ATTR,
        diagnostic_key="diffusion_path_support",
    )
    _patch_impossible_posterior_guard(
        models.CandidateKinematicModel,
        marker=_CANDIDATE_PATCHED_ATTR,
        diagnostic_key="candidate_path_support",
    )


def _patch_exact_sparse_log_matvec(models: Any) -> None:
    """Keep unreachable diffusion states at exact negative-infinite log mass.

    The legacy sparse recursion initialized every destination with ``LOG_ZERO``
    (``-1e300``).  A destination with no reachable incoming transition therefore
    retained a finite sentinel.  If its emission was finite, that sentinel could
    become the event's apparent evidence and prevent the impossible-path guard
    from recognizing a disconnected trajectory.
    """

    current = models._log_sparse_matvec
    if getattr(current, _SPARSE_MATVEC_PATCHED_ATTR, False):
        return

    @wraps(current)
    def exact_sparse_log_matvec(
        log_alpha: Any,
        transition: Any,
    ) -> np.ndarray:
        alpha = np.asarray(log_alpha, dtype=float)
        result = np.full(alpha.shape, -np.inf, dtype=float)
        for source, (destination_indices, log_weights) in enumerate(transition):
            destinations = np.asarray(destination_indices, dtype=int)
            weights = np.asarray(log_weights, dtype=float)
            contributions = alpha[source] + weights
            for destination, contribution in zip(destinations, contributions):
                destination_index = int(destination)
                result[destination_index] = np.logaddexp(
                    result[destination_index],
                    contribution,
                )
        return result

    setattr(exact_sparse_log_matvec, _SPARSE_MATVEC_PATCHED_ATTR, True)
    setattr(exact_sparse_log_matvec, "__hipporeplayimm_original__", current)
    models._log_sparse_matvec = exact_sparse_log_matvec


def _patch_exact_pair_posteriors(models: Any) -> None:
    """Keep bins outside candidate support at exact zero posterior probability.

    ``models.LOG_ZERO`` is a finite legacy sentinel.  Using it to prefill a
    posterior vector is unsafe because the vector is subsequently normalized:
    sufficiently small but valid candidate log mass can fall below the sentinel,
    making an impossible non-candidate bin dominate the normalized posterior.
    Restrict exact ``-inf`` to these structural posterior zeros instead of
    changing the process-wide sentinel and its compatibility semantics.
    """

    current_terminal = models._pair_terminal_posterior
    current_previous = models._pair_previous_posterior
    if getattr(current_terminal, _PAIR_POSTERIOR_PATCHED_ATTR, False) and getattr(
        current_previous,
        _PAIR_POSTERIOR_PATCHED_ATTR,
        False,
    ):
        return

    @wraps(current_terminal)
    def exact_pair_terminal_posterior(
        log_pair_or_modes: Any,
        current_indices: Any,
        n_bins: int,
    ) -> np.ndarray:
        values = np.asarray(log_pair_or_modes, dtype=float)
        if values.ndim == 3:
            collapsed = models.logsumexp(values, axis=(0, 1))
        else:
            collapsed = models.logsumexp(values, axis=0)
        log_posterior = np.full(int(n_bins), -np.inf, dtype=float)
        log_posterior[np.asarray(current_indices, dtype=int)] = collapsed
        return models._normalize_log_weights(log_posterior)

    @wraps(current_previous)
    def exact_pair_previous_posterior(
        log_pair_or_modes: Any,
        previous_indices: Any,
        n_bins: int,
    ) -> np.ndarray:
        values = np.asarray(log_pair_or_modes, dtype=float)
        if values.ndim == 3:
            collapsed = models.logsumexp(values, axis=(0, 2))
        else:
            collapsed = models.logsumexp(values, axis=1)
        log_posterior = np.full(int(n_bins), -np.inf, dtype=float)
        log_posterior[np.asarray(previous_indices, dtype=int)] = collapsed
        return models._normalize_log_weights(log_posterior)

    setattr(
        exact_pair_terminal_posterior,
        _PAIR_POSTERIOR_PATCHED_ATTR,
        True,
    )
    setattr(
        exact_pair_terminal_posterior,
        "__hipporeplayimm_original__",
        current_terminal,
    )
    setattr(
        exact_pair_previous_posterior,
        _PAIR_POSTERIOR_PATCHED_ATTR,
        True,
    )
    setattr(
        exact_pair_previous_posterior,
        "__hipporeplayimm_original__",
        current_previous,
    )
    models._pair_terminal_posterior = exact_pair_terminal_posterior
    models._pair_previous_posterior = exact_pair_previous_posterior


def _patch_stationary_trajectory_posterior(model_type: type[Any]) -> None:
    """Repeat the full-event posterior for every stationary trajectory bin.

    A stationary model has one latent position shared by the whole event. Its
    trajectory posterior must therefore be the same smoothed posterior at every
    time bin, rather than the sequence of online filtering posteriors produced
    while accumulating emissions.
    """

    current = model_type.score
    if getattr(current, _STATIONARY_PATCHED_ATTR, False):
        return

    @wraps(current)
    def score_with_smoothed_stationary_trajectory(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ):
        result = current(self, *args, **kwargs)
        terminal = result.terminal_log_posterior
        if terminal is None:
            return result
        terminal_values = np.asarray(terminal, dtype=float)
        result.trajectory_log_posterior = np.repeat(
            terminal_values[None, :],
            int(result.n_time),
            axis=0,
        )
        return result

    setattr(
        score_with_smoothed_stationary_trajectory,
        _STATIONARY_PATCHED_ATTR,
        True,
    )
    setattr(
        score_with_smoothed_stationary_trajectory,
        "__hipporeplayimm_original__",
        current,
    )
    model_type.score = score_with_smoothed_stationary_trajectory


def _patch_impossible_posterior_guard(
    model_type: type[Any],
    *,
    marker: str,
    diagnostic_key: str,
) -> None:
    """Clear posterior-derived outputs when a dynamic scorer has no finite path."""

    current = model_type.score
    if getattr(current, marker, False):
        return

    @wraps(current)
    def score_with_impossible_posterior_guard(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ):
        with np.errstate(invalid="ignore"):
            result = current(self, *args, **kwargs)
        return _clear_impossible_posterior(result, diagnostic_key)

    setattr(score_with_impossible_posterior_guard, marker, True)
    setattr(
        score_with_impossible_posterior_guard,
        "__hipporeplayimm_original__",
        current,
    )
    setattr(model_type, "score", score_with_impossible_posterior_guard)


def _clear_impossible_posterior(result: Any, diagnostic_key: str) -> Any:
    """Remove undefined posterior fields for an exact negative-infinite score."""

    if not np.isneginf(float(result.log_likelihood)):
        return result

    result.terminal_log_posterior = None
    result.trajectory_log_posterior = None
    result.diagnostics = dict(getattr(result, "diagnostics", {}) or {})
    for key in _POSTERIOR_DIAGNOSTIC_KEYS:
        result.diagnostics.pop(key, None)
    result.diagnostics[diagnostic_key] = _NO_FINITE_PATH
    return result


__all__ = ["apply_diffusion_impossible_path_guard_patch"]

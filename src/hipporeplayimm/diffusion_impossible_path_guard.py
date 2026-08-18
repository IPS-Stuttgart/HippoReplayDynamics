"""Keep core replay-model posterior outputs consistent with model support."""

from __future__ import annotations

import sys
from functools import wraps
from typing import Any

import numpy as np

_PATCHED_ATTR = "_diffusion_impossible_posterior_guard_applied"
_CANDIDATE_PATCHED_ATTR = "_candidate_impossible_posterior_guard_applied"
_STATIONARY_PATCHED_ATTR = "_stationary_smoothed_trajectory_posterior_applied"
_SPARSE_MATVEC_PATCHED_ATTR = "_diffusion_exact_sparse_matvec_support_applied"
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
    """Install exact log-zero support and posterior guards for core replay models."""

    from . import models

    _install_exact_log_zero(models)
    _patch_exact_sparse_log_matvec(models)
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


def _install_exact_log_zero(models: Any) -> None:
    """Keep impossible and pruned states at true zero probability in log space."""

    models.LOG_ZERO = -np.inf
    for module in list(sys.modules.values()):
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("hipporeplayimm"):
            continue
        if hasattr(module, "LOG_ZERO"):
            setattr(module, "LOG_ZERO", -np.inf)
        if hasattr(module, "_LOG_ZERO_ROW_THRESHOLD"):
            setattr(module, "_LOG_ZERO_ROW_THRESHOLD", -np.inf)


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

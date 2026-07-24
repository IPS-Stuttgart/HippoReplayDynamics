"""Make KD evidence recursions and grid summaries robust to impossible rows."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

_PATCHED_FLAG = "_kd_impossible_emission_patch_applied"
_KD_POISSON_WRAPPER_FLAG = "_kd_zero_rate_exact_support_wrapper"
_KD_POISSON_ORIGINAL_ATTR = "__hipporeplayimm_kd_zero_rate_original__"
_KD_POISSON_WRAPPER_VERSION = 1
_KD_DURATION_DIFFUSION_WRAPPER_FLAG = "_kd_duration_impossible_diffusion_wrapper"
_KD_DURATION_DIFFUSION_ORIGINAL_ATTR = "__hipporeplayimm_kd_duration_diffusion_original__"
_KD_DURATION_MOMENTUM_WRAPPER_FLAG = "_kd_duration_impossible_momentum_wrapper"
_KD_DURATION_MOMENTUM_ORIGINAL_ATTR = "__hipporeplayimm_kd_duration_momentum_original__"
_KD_DURATION_WRAPPER_VERSION = 1


def apply_kd_impossible_emission_patch() -> None:
    """Install guards for all-impossible KD emission rows and grid summaries."""

    from . import kd_reference as kd

    _patch_kd_poisson_exact_support(kd)
    kd._scaled_emission = _scaled_emission
    kd._first_order_separable_log_evidence = _first_order_separable_log_evidence
    kd._second_order_separable_log_evidence = _second_order_separable_log_evidence
    kd.kd_stationary_gaussian_log_evidence_from_transitions = _stationary_gaussian_log_evidence_from_transitions
    _patch_duration_aware_impossible_rows(kd)
    kd.empirical_grid_prior = _empirical_grid_prior
    kd.best_grid_params = _best_grid_params
    setattr(kd, _PATCHED_FLAG, True)


def _patch_kd_poisson_exact_support(kd) -> None:
    """Preserve mathematically exact support in the KD Poisson likelihood."""

    current = kd.poisson_log_emissions
    if getattr(current, _KD_POISSON_WRAPPER_FLAG, None) == _KD_POISSON_WRAPPER_VERSION:
        return
    original = getattr(current, _KD_POISSON_ORIGINAL_ATTR, current)

    @wraps(original)
    def poisson_log_emissions(
        spike_counts,
        rates_hz,
        dt,
        *,
        spike_rate_scale=1.0,
    ):
        from .poisson_input_boolean_validation import _restore_exact_zero_rate_support

        log_likelihood = original(
            spike_counts,
            rates_hz,
            dt,
            spike_rate_scale=spike_rate_scale,
        )
        counts = np.asarray(spike_counts, dtype=float)
        rates = np.asarray(rates_hz, dtype=float)
        return _restore_exact_zero_rate_support(
            log_likelihood,
            spike_counts=counts,
            rates_hz=rates,
            dt=dt,
            spike_rate_scale=float(spike_rate_scale),
            cell_weights=np.ones(counts.shape[1], dtype=float),
            likelihood_temperature=1.0,
            negative_binomial_overdispersion=0.0,
        )

    setattr(
        poisson_log_emissions,
        _KD_POISSON_WRAPPER_FLAG,
        _KD_POISSON_WRAPPER_VERSION,
    )
    setattr(poisson_log_emissions, _KD_POISSON_ORIGINAL_ATTR, original)
    kd.poisson_log_emissions = poisson_log_emissions


def _patch_duration_aware_impossible_rows(kd) -> None:
    """Guard the list-transition recursions installed by the duration patch."""

    current_diffusion = kd.kd_diffusion_log_evidence_from_transition
    if getattr(current_diffusion, _KD_DURATION_DIFFUSION_WRAPPER_FLAG, None) != _KD_DURATION_WRAPPER_VERSION:
        original_diffusion = getattr(
            current_diffusion,
            _KD_DURATION_DIFFUSION_ORIGINAL_ATTR,
            current_diffusion,
        )

        @wraps(original_diffusion)
        def diffusion_log_evidence(log_emissions, n_bins_x, n_bins_y, transition):
            if isinstance(transition, (list, tuple)):
                return _first_order_duration_log_evidence(
                    kd,
                    log_emissions,
                    n_bins_x,
                    n_bins_y,
                    transition,
                )
            return original_diffusion(log_emissions, n_bins_x, n_bins_y, transition)

        setattr(
            diffusion_log_evidence,
            _KD_DURATION_DIFFUSION_WRAPPER_FLAG,
            _KD_DURATION_WRAPPER_VERSION,
        )
        setattr(
            diffusion_log_evidence,
            _KD_DURATION_DIFFUSION_ORIGINAL_ATTR,
            original_diffusion,
        )
        kd.kd_diffusion_log_evidence_from_transition = diffusion_log_evidence

    current_momentum = kd.kd_momentum_log_evidence_from_transitions
    if getattr(current_momentum, _KD_DURATION_MOMENTUM_WRAPPER_FLAG, None) != _KD_DURATION_WRAPPER_VERSION:
        original_momentum = getattr(
            current_momentum,
            _KD_DURATION_MOMENTUM_ORIGINAL_ATTR,
            current_momentum,
        )

        @wraps(original_momentum)
        def momentum_log_evidence(log_emissions, n_bins, initial, transition):
            if isinstance(transition, (list, tuple)):
                return _second_order_duration_log_evidence(
                    kd,
                    log_emissions,
                    n_bins,
                    initial,
                    transition,
                )
            return original_momentum(log_emissions, n_bins, initial, transition)

        setattr(
            momentum_log_evidence,
            _KD_DURATION_MOMENTUM_WRAPPER_FLAG,
            _KD_DURATION_WRAPPER_VERSION,
        )
        setattr(
            momentum_log_evidence,
            _KD_DURATION_MOMENTUM_ORIGINAL_ATTR,
            original_momentum,
        )
        kd.kd_momentum_log_evidence_from_transitions = momentum_log_evidence


def _first_order_duration_log_evidence(kd, log_emissions, n_bins_x, n_bins_y, transitions) -> float:
    emission, offset = kd._scaled_emission(log_emissions, 0)
    alpha = emission.reshape(n_bins_x, n_bins_y) / log_emissions.shape[1]
    conditional = float(alpha.sum())
    if conditional <= 0.0:
        return float("-inf")
    logp = np.log(conditional) + offset
    alpha /= conditional

    for time_index in range(1, log_emissions.shape[0]):
        emission, offset = kd._scaled_emission(log_emissions, time_index)
        transition = transitions[time_index - 1]
        predicted = transition @ alpha @ transition.T
        alpha = predicted * emission.reshape(n_bins_x, n_bins_y)
        conditional = float(alpha.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha /= conditional
    return float(logp)


def _second_order_duration_log_evidence(kd, log_emissions, n_bins, initial, transitions) -> float:
    if log_emissions.shape[0] == 1:
        return kd.kd_random_log_evidence(log_emissions)

    emission0, offset0 = kd._scaled_emission(log_emissions, 0)
    alpha0 = emission0.reshape(n_bins, n_bins) / log_emissions.shape[1]
    conditional0 = float(alpha0.sum())
    if conditional0 <= 0.0:
        return float("-inf")
    logp = np.log(conditional0) + offset0
    alpha0 /= conditional0

    emission1, offset1 = kd._scaled_emission(log_emissions, 1)
    emission1_grid = emission1.reshape(n_bins, n_bins)
    alpha_t = np.einsum("ip,jq,pq,ij->ijpq", initial, initial, alpha0, emission1_grid, optimize=True)
    conditional1 = float(alpha_t.sum())
    if conditional1 <= 0.0:
        return float("-inf")
    logp += np.log(conditional1) + offset1
    alpha_t /= conditional1

    for time_index in range(2, log_emissions.shape[0]):
        transition = transitions[time_index - 2]
        emission, offset = kd._scaled_emission(log_emissions, time_index)
        emission_grid = emission.reshape(n_bins, n_bins)
        y_sum = np.einsum("jbq,abpq->abpj", transition, alpha_t, optimize=True)
        predicted = np.einsum("iap,abpj->ijab", transition, y_sum, optimize=True)
        alpha_t = predicted * emission_grid[:, :, None, None]
        conditional = float(alpha_t.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha_t /= conditional
    return float(logp)


def _scaled_emission(log_emissions: np.ndarray, time_index: int) -> tuple[np.ndarray, float]:
    row = np.asarray(log_emissions[time_index], dtype=float)
    if np.any(np.isnan(row)) or np.any(np.isposinf(row)):
        raise ValueError("log_emissions cannot contain NaN or +inf")
    offset = float(np.max(row))
    if np.isneginf(offset):
        return np.zeros_like(row, dtype=float), offset
    return np.exp(row - offset), offset


def _stationary_gaussian_log_evidence_from_transitions(
    log_emissions: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    transition_x: np.ndarray,
    transition_y: np.ndarray | None = None,
) -> float:
    """Score the KD stationary-Gaussian model without inventing support.

    Zero predictive mass is mathematically impossible for the corresponding
    latent center.  Keeping those entries at ``-inf`` is essential when
    different time bins have disjoint latent-center support.
    """

    from scipy.special import logsumexp

    transition_y = transition_x if transition_y is None else transition_y
    log_terms = np.zeros((n_bins_x, n_bins_y), dtype=float)
    for time_index in range(log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        weighted = transition_x.T @ emission.reshape(n_bins_x, n_bins_y) @ transition_y
        positive = weighted > 0.0
        if not np.any(positive):
            return float("-inf")
        log_weighted = np.full(weighted.shape, float("-inf"), dtype=float)
        log_weighted[positive] = np.log(weighted[positive])
        log_terms += log_weighted + offset
    return float(logsumexp(log_terms.reshape(-1) - np.log(log_emissions.shape[1])))


def _first_order_separable_log_evidence(log_emissions: np.ndarray, n_bins_x: int, n_bins_y: int, transition: np.ndarray) -> float:
    emission, offset = _scaled_emission(log_emissions, 0)
    alpha = emission.reshape(n_bins_x, n_bins_y) / log_emissions.shape[1]
    conditional = float(alpha.sum())
    if conditional <= 0.0:
        return float("-inf")
    logp = np.log(conditional) + offset
    alpha /= conditional
    for time_index in range(1, log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        predicted = transition @ alpha @ transition.T
        alpha = predicted * emission.reshape(n_bins_x, n_bins_y)
        conditional = float(alpha.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha /= conditional
    return float(logp)


def _second_order_separable_log_evidence(log_emissions: np.ndarray, n_bins: int, initial: np.ndarray, transition: np.ndarray) -> float:
    emission0, offset0 = _scaled_emission(log_emissions, 0)
    alpha0 = emission0.reshape(n_bins, n_bins) / log_emissions.shape[1]
    conditional0 = float(alpha0.sum())
    if conditional0 <= 0.0:
        return float("-inf")
    logp = np.log(conditional0) + offset0
    alpha0 /= conditional0
    if log_emissions.shape[0] == 1:
        return float(logp)

    emission1, offset1 = _scaled_emission(log_emissions, 1)
    emission1_grid = emission1.reshape(n_bins, n_bins)
    alpha_t = np.einsum("ip,jq,pq,ij->ijpq", initial, initial, alpha0, emission1_grid, optimize=True)
    conditional1 = float(alpha_t.sum())
    if conditional1 <= 0.0:
        return float("-inf")
    logp += np.log(conditional1) + offset1
    alpha_t /= conditional1

    for time_index in range(2, log_emissions.shape[0]):
        emission, offset = _scaled_emission(log_emissions, time_index)
        emission_grid = emission.reshape(n_bins, n_bins)
        y_sum = np.einsum("jbq,abpq->abpj", transition, alpha_t, optimize=True)
        predicted = np.einsum("iap,abpj->ijab", transition, y_sum, optimize=True)
        alpha_t = predicted * emission_grid[:, :, None, None]
        conditional = float(alpha_t.sum())
        if conditional <= 0.0:
            return float("-inf")
        logp += np.log(conditional) + offset
        alpha_t /= conditional
    return float(logp)


def _empirical_grid_prior(grid_params: dict[str, np.ndarray], grid: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit an empirical KD grid prior using only event rows with finite scores."""

    from . import kd_reference as kd

    values = np.asarray(grid, dtype=float)
    finite_rows = _finite_event_grid_rows(values)
    if not np.any(finite_rows):
        raise ValueError("grid must contain at least one finite event/parameter score")
    finite_event_count = int(np.sum(finite_rows))

    if values.ndim == 2:
        sd_meters = np.asarray(grid_params["sd_meters"], dtype=float)
        if values.shape[1] != sd_meters.shape[0]:
            raise ValueError("sd_meters length must match the grid parameter axis")
        finite_grid = np.where(np.isfinite(values[finite_rows]), values[finite_rows], np.nan)
        best = sd_meters[np.nanargmax(finite_grid, axis=1)]
        prior = kd._fit_invgamma_prior(best, sd_meters)
        return prior, {"sd_prior_mass": float(prior.sum()), "finite_event_rows": finite_event_count}

    sd_meters = np.asarray(grid_params["sd_meters"], dtype=float)
    decay = np.asarray(grid_params["decay"], dtype=float)
    if values.shape[1:] != (sd_meters.shape[0], decay.shape[0]):
        raise ValueError("sd_meters and decay lengths must match the grid parameter axes")
    finite_grid = np.where(np.isfinite(values[finite_rows]), values[finite_rows], np.nan)
    flat = np.nanargmax(finite_grid.reshape(finite_grid.shape[0], -1), axis=1)
    sd_idx, decay_idx = np.unravel_index(flat, values.shape[1:])
    prior = kd._fit_lognormal2d_prior(sd_meters[sd_idx], decay[decay_idx], sd_meters, decay)
    return prior, {"joint_prior_mass": float(prior.sum()), "finite_event_rows": finite_event_count}


def _best_grid_params(model: str, event_indices: Iterable[int], grid_params: dict[str, np.ndarray], grid: np.ndarray) -> list[dict[str, float | int | str]]:
    """Return per-event KD grid optima without inventing parameters for failed rows."""

    values = np.asarray(grid, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("grid must have 2 or 3 dimensions")
    event_list = list(event_indices)
    if len(event_list) != values.shape[0]:
        raise ValueError("event_indices must contain one entry per grid event row")

    sd_meters = np.asarray(grid_params["sd_meters"], dtype=float)
    if values.shape[1] != sd_meters.shape[0]:
        raise ValueError("sd_meters length must match the grid parameter axis")
    decay = None
    if values.ndim == 3:
        decay = np.asarray(grid_params["decay"], dtype=float)
        if values.shape[2] != decay.shape[0]:
            raise ValueError("decay length must match the grid parameter axis")

    rows: list[dict[str, float | int | str]] = []
    for row_index, event_index in enumerate(event_list):
        if values.ndim == 2:
            row = values[row_index]
            if not np.any(np.isfinite(row)):
                rows.append(
                    {
                        "event_index": int(event_index),
                        "model": model,
                        "best_sd_meters": float("nan"),
                        "best_log_evidence": float("nan"),
                    }
                )
                continue
            best = int(np.nanargmax(np.where(np.isfinite(row), row, np.nan)))
            rows.append(
                {
                    "event_index": int(event_index),
                    "model": model,
                    "best_sd_meters": float(sd_meters[best]),
                    "best_log_evidence": float(row[best]),
                }
            )
        else:
            assert decay is not None
            event_grid = values[row_index]
            if not np.any(np.isfinite(event_grid)):
                rows.append(
                    {
                        "event_index": int(event_index),
                        "model": model,
                        "best_sd_meters": float("nan"),
                        "best_decay": float("nan"),
                        "best_log_evidence": float("nan"),
                    }
                )
                continue
            best = int(np.nanargmax(np.where(np.isfinite(event_grid), event_grid, np.nan).reshape(-1)))
            sd_idx, decay_idx = np.unravel_index(best, event_grid.shape)
            rows.append(
                {
                    "event_index": int(event_index),
                    "model": model,
                    "best_sd_meters": float(sd_meters[sd_idx]),
                    "best_decay": float(decay[decay_idx]),
                    "best_log_evidence": float(event_grid[sd_idx, decay_idx]),
                }
            )
    return rows


def _finite_event_grid_rows(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    if values.ndim not in (2, 3):
        raise ValueError("grid must have 2 or 3 dimensions")
    return np.any(np.isfinite(values), axis=tuple(range(1, values.ndim)))


__all__ = ["apply_kd_impossible_emission_patch"]

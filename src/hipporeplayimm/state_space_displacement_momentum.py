"""Exact finite-displacement momentum state-space decoder.

The candidate-supported pairwise momentum decoder is exact only when the
candidate support is the full spatial grid.  This module provides a separate,
paper-facing approximation that is exact over a declared finite state space
``(position, displacement)``.  It therefore gives comparable evidence rows for
testing second-order/momentum structure without treating a candidate-pruned
lower bound as a model evidence.

The displacement state is a small lattice of physical displacements derived
from the spatial grid spacing.  Dynamics are first order in the augmented state:

    p(d_t | d_{t-1}) p(x_t | x_{t-1}, d_t) p(y_t | x_t)

This is intentionally not the same model as the full pairwise momentum decoder;
it is an exact finite-state surrogate designed for recovery diagnostics and
paper-level sensitivity analyses.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

from .encoding import LogEmissionTensor
from .state_space_utils import _as_log_probs, _mean_entropy, _scaled_emissions


def _score_displacement_momentum_exact(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    config: object,
    transition_durations_s: Iterable[float],
    *,
    valid_bin_mask: np.ndarray | None = None,
    return_trajectory: bool = True,
) -> tuple[float, np.ndarray | None, np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Return exact evidence and posteriors for the finite-displacement model.

    Parameters
    ----------
    emissions:
        Replay-bin log likelihoods with shape ``(time, position)``.
    bin_centers:
        Spatial-bin centers in centimetres.
    config:
        ``StateSpaceDecoderConfig`` or a duck-typed object with the displacement
        and momentum attributes used below.
    transition_durations_s:
        One duration per transition.  Uniform-bin workflows pass a constant
        sequence; partial-bin workflows can pass variable durations.
    valid_bin_mask:
        Optional occupancy-derived spatial support.  Evidence is exact over the
        valid spatial bins crossed with the displacement lattice.
    return_trajectory:
        When false, skip the backward smoother and return only the terminal
        position posterior needed by evidence-only callers.
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
    vectors = _displacement_lattice(
        centers,
        radius_bins=int(getattr(config, "displacement_radius_bins", 2)),
    )
    n_bins = emissions.n_bins
    n_displacements = vectors.shape[0]
    valid_count = _valid_count(valid_mask, n_bins)

    if n_displacements <= 0:
        raise ValueError("displacement lattice must contain at least one state")

    reference_dt = float(np.median(durations)) if durations.size else float(emissions.dt)
    transition_sigmas = _transition_sigmas_cm(config, durations)
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

    position_prior = _uniform_position_prior(n_bins, valid_mask)
    displacement_prior = _zero_centered_displacement_prior(vectors, prior_sigma)
    alpha = position_prior[:, None] * displacement_prior[None, :] * scaled[0, :, None]
    scales = np.zeros(emissions.n_time, dtype=float)
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError("first emission row has no finite likelihood mass")
    alpha /= scales[0]
    filtered = np.zeros((emissions.n_time, n_bins, n_displacements), dtype=float)
    filtered[0] = alpha
    logp = float(np.log(scales[0]) + offsets[0])

    displacement_transitions: list[np.ndarray] = []
    spatial_transitions: list[list[csr_matrix]] = []
    for time_index in range(1, emissions.n_time):
        transition_index = time_index - 1
        duration_scale = _duration_scale_at(durations, transition_index, reference_dt)
        displacement_transition = _displacement_transition_matrix(
            vectors,
            sigma_cm=float(transition_sigmas[transition_index]),
            decay=float(decays[transition_index]) * float(time_scales[transition_index]),
        )
        transitions = [
            _shifted_gaussian_transition_matrix(
                centers,
                displacement=duration_scale * vectors[disp_index],
                sigma_cm=position_sigma,
                max_step_sigma=float(getattr(config, "max_step_sigma", 4.0)),
                valid_bin_mask=valid_mask,
            )
            for disp_index in range(n_displacements)
        ]
        mixed = alpha @ displacement_transition.T
        predicted = np.zeros_like(alpha)
        for disp_index, transition in enumerate(transitions):
            predicted[:, disp_index] = np.asarray(
                transition @ mixed[:, disp_index],
                dtype=float,
            )
        alpha = predicted * scaled[time_index, :, None]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        logp += float(np.log(scales[time_index]) + offsets[time_index])
        displacement_transitions.append(displacement_transition)
        spatial_transitions.append(transitions)

    if return_trajectory:
        smoothed = np.zeros_like(filtered)
        beta = np.ones((n_bins, n_displacements), dtype=float)
        smoothed[-1] = filtered[-1]
        for time_index in range(emissions.n_time - 1, 0, -1):
            transition_index = time_index - 1
            spatial_backward = np.zeros_like(beta)
            for disp_index, transition in enumerate(spatial_transitions[transition_index]):
                values = scaled[time_index] * beta[:, disp_index]
                spatial_backward[:, disp_index] = np.asarray(
                    transition.T @ values,
                    dtype=float,
                )
            beta = (spatial_backward @ displacement_transitions[transition_index]) / scales[time_index]
            gamma = filtered[time_index - 1] * beta
            total = float(gamma.sum())
            smoothed[time_index - 1] = gamma / total if total > 0.0 else filtered[time_index - 1]

        position_posterior = smoothed.sum(axis=2)
        displacement_posterior = smoothed.sum(axis=1)
        trajectory = _as_log_probs(position_posterior)
        terminal_log_posterior = trajectory[-1]
    else:
        trajectory = None
        position_posterior = alpha.sum(axis=1)[None, :]
        displacement_posterior = alpha.sum(axis=0)[None, :]
        terminal_log_posterior = _as_log_probs(position_posterior)[0]
    displacement_log_posterior = _as_log_probs(displacement_posterior)
    median_transition_sigma = _median_or_fallback(
        transition_sigmas,
        _per_bin_sigma(_displacement_transition_sigma_cm_sqrt_s(config), reference_dt),
    )
    median_decay = _median_or_fallback(decays, float(getattr(config, "momentum_velocity_decay", 0.95)))
    diagnostics: dict[str, float | int | str] = {
        "state_space_displacement_momentum_evidence_support": "exact_full_grid",
        "state_space_displacement_momentum_state_support": "finite_displacement_grid",
        "state_space_displacement_radius_bins": int(getattr(config, "displacement_radius_bins", 2)),
        "state_space_displacement_state_count": int(n_displacements),
        "state_space_displacement_joint_state_count": int(valid_count * n_displacements),
        "state_space_displacement_position_sigma_cm": float(position_sigma),
        "state_space_displacement_prior_sigma_cm": float(prior_sigma),
        "state_space_displacement_transition_sigma_cm": float(median_transition_sigma),
        "state_space_displacement_transition_sigma_cm_per_step": _format_float_series(transition_sigmas),
        "state_space_displacement_velocity_decay_effective": float(median_decay),
        "state_space_displacement_velocity_decay_per_step": _format_float_series(decays),
        "state_space_displacement_mean_posterior_entropy": _mean_entropy(displacement_log_posterior),
        "state_space_displacement_vectors_cm": _format_vector_series(vectors),
    }
    return (
        float(logp),
        trajectory,
        terminal_log_posterior,
        displacement_log_posterior,
        diagnostics,
    )


def _displacement_lattice(bin_centers: np.ndarray, *, radius_bins: int) -> np.ndarray:
    """Return a rectangular displacement lattice in centimetres.

    ``radius_bins=2`` in a 2-D grid yields 25 displacement states.  The zero
    vector is always present.  Coordinates with no measurable spacing fall back
    to unit spacing, which keeps tiny synthetic test grids well-defined.
    """

    if int(radius_bins) < 0:
        raise ValueError("displacement_radius_bins must be nonnegative")
    centers = _as_2d_centers(bin_centers)
    spacing = _grid_spacings(centers)
    offsets = range(-int(radius_bins), int(radius_bins) + 1)
    vectors = np.asarray(
        [np.asarray(index, dtype=float) * spacing for index in _offset_product(offsets, centers.shape[1])],
        dtype=float,
    )
    if vectors.size == 0:
        vectors = np.zeros((1, centers.shape[1]), dtype=float)
    zero = np.zeros(centers.shape[1], dtype=float)
    if not np.any(np.all(np.isclose(vectors, zero[None, :]), axis=1)):
        vectors = np.vstack([zero, vectors])
    order = np.argsort(np.sum(vectors * vectors, axis=1), kind="stable")
    return vectors[order]


def _offset_product(values: range, ndim: int):
    if ndim <= 1:
        for value in values:
            yield (value,)
        return
    import itertools

    yield from itertools.product(values, repeat=ndim)


def _shifted_gaussian_transition_matrix(
    bin_centers: np.ndarray,
    *,
    displacement: np.ndarray,
    sigma_cm: float,
    max_step_sigma: float,
    valid_bin_mask: np.ndarray | None = None,
) -> csr_matrix:
    if sigma_cm <= 0.0 or not np.isfinite(sigma_cm):
        raise ValueError("sigma_cm must be finite and positive")
    centers = _as_2d_centers(bin_centers)
    n_bins = centers.shape[0]
    valid_mask = _coerce_valid_bin_mask(valid_bin_mask, n_bins)
    allowed = np.arange(n_bins, dtype=int) if valid_mask is None else np.flatnonzero(valid_mask)
    radius2 = (float(sigma_cm) * float(max_step_sigma)) ** 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    displacement = np.asarray(displacement, dtype=float).reshape(1, centers.shape[1])
    for src, center in enumerate(centers):
        predicted = center[None, :] + displacement
        dist2 = np.sum((centers - predicted) ** 2, axis=1)
        keep = dist2 <= radius2
        if valid_mask is not None:
            keep &= valid_mask
        if not np.any(keep):
            keep[int(allowed[int(np.argmin(dist2[allowed]))])] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (float(sigma_cm) * float(sigma_cm)))
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            weights = np.ones(dst.shape[0], dtype=float) / max(int(dst.shape[0]), 1)
        else:
            weights /= weights_sum
        rows.extend(int(idx) for idx in dst)
        cols.extend([int(src)] * len(dst))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(n_bins, n_bins))


def _displacement_transition_matrix(
    vectors: np.ndarray,
    *,
    sigma_cm: float,
    decay: float,
) -> np.ndarray:
    if sigma_cm <= 0.0 or not np.isfinite(sigma_cm):
        raise ValueError("displacement transition sigma must be finite and positive")
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("displacement velocity decay must be finite and nonnegative")
    predicted = float(decay) * vectors
    delta = vectors[:, None, :] - predicted[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    weights = np.exp(-0.5 * dist2 / (float(sigma_cm) * float(sigma_cm)))
    weights_sum = weights.sum(axis=0, keepdims=True)
    weights_sum[weights_sum <= 0.0] = 1.0
    return weights / weights_sum


def _zero_centered_displacement_prior(vectors: np.ndarray, sigma_cm: float) -> np.ndarray:
    dist2 = np.sum(vectors * vectors, axis=1)
    weights = np.exp(-0.5 * dist2 / (float(sigma_cm) * float(sigma_cm)))
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        weights = np.ones(vectors.shape[0], dtype=float)
        total = float(weights.sum())
    return weights / total


def _uniform_position_prior(n_bins: int, valid_bin_mask: np.ndarray | None) -> np.ndarray:
    prior = np.zeros(n_bins, dtype=float)
    if valid_bin_mask is None:
        prior.fill(1.0 / n_bins)
    else:
        prior[valid_bin_mask] = 1.0 / int(np.sum(valid_bin_mask))
    return prior


def _coerce_valid_bin_mask(mask: np.ndarray | None, n_bins: int) -> np.ndarray | None:
    if mask is None:
        return None
    out = np.asarray(mask, dtype=bool)
    if out.shape != (n_bins,):
        raise ValueError("valid_bin_mask must contain one boolean value per spatial bin")
    if not np.any(out):
        raise ValueError("valid_bin_mask must contain at least one valid spatial bin")
    return out


def _valid_count(mask: np.ndarray | None, n_bins: int) -> int:
    return int(n_bins) if mask is None else int(np.sum(mask))


def _as_2d_centers(bin_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim == 1:
        centers = centers[:, None]
    if centers.ndim != 2:
        raise ValueError("bin_centers must be a one- or two-dimensional array")
    return centers


def _grid_spacings(centers: np.ndarray) -> np.ndarray:
    spacings = []
    for dim in range(centers.shape[1]):
        values = np.unique(np.asarray(centers[:, dim], dtype=float))
        diffs = np.diff(np.sort(values))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        spacings.append(float(np.median(diffs)) if diffs.size else 1.0)
    return np.asarray(spacings, dtype=float)


def _default_position_sigma_cm(centers: np.ndarray) -> float:
    spacing = _grid_spacings(centers)
    return max(float(np.median(spacing)) * 0.75, np.finfo(float).eps)


def _positive_config_value(config: object, name: str, *, default: float) -> float:
    value = float(getattr(config, name, 0.0))
    if np.isfinite(value) and value > 0.0:
        return value
    return float(default)


def _transition_sigmas_cm(config: object, durations: np.ndarray) -> np.ndarray:
    sigma_cm_sqrt_s = _displacement_transition_sigma_cm_sqrt_s(config)
    return np.asarray([_per_bin_sigma(sigma_cm_sqrt_s, duration) for duration in durations], dtype=float)


def _displacement_transition_sigma_cm_sqrt_s(config: object) -> float:
    value = float(getattr(config, "displacement_transition_sigma_cm_sqrt_s", 0.0))
    if np.isfinite(value) and value > 0.0:
        return value
    return float(getattr(config, "momentum_sigma_cm_sqrt_s", 85.0))


def _per_bin_sigma(sigma_cm_sqrt_s: float, dt_s: float) -> float:
    sigma = float(sigma_cm_sqrt_s)
    dt = float(dt_s)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_cm_sqrt_s must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    return max(sigma * np.sqrt(dt), np.finfo(float).eps)


def _coerce_transition_durations(values: Iterable[float], *, n_time: int, fallback_dt: float) -> np.ndarray:
    expected = max(int(n_time) - 1, 0)
    out = np.asarray(list(values), dtype=float)
    if out.shape != (expected,) or not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        return np.full(expected, float(fallback_dt), dtype=float)
    return out


def _duration_adjusted_decays(config: object, durations: np.ndarray, reference_dt: float) -> np.ndarray:
    if durations.size == 0:
        return np.empty(0, dtype=float)
    tau_s = float(getattr(config, "momentum_velocity_decay_tau_s", 0.0))
    if tau_s > 0.0:
        if not np.isfinite(tau_s):
            raise ValueError("momentum_velocity_decay_tau_s must be finite when positive")
        return np.exp(-durations / tau_s)
    decay = float(getattr(config, "momentum_velocity_decay", 0.95))
    if not np.isfinite(decay) or decay < 0.0:
        raise ValueError("momentum_velocity_decay must be finite and nonnegative")
    return np.asarray([decay ** (float(duration) / reference_dt) for duration in durations], dtype=float)


def _time_scales(durations: np.ndarray) -> np.ndarray:
    scales = np.ones_like(durations, dtype=float)
    if durations.size > 1:
        scales[1:] = durations[1:] / durations[:-1]
    return scales


def _duration_scale_at(durations: np.ndarray, transition_index: int, reference_dt: float) -> float:
    if durations.size == 0:
        return 1.0
    return float(durations[transition_index]) / float(reference_dt)


def _median_or_fallback(values: np.ndarray, fallback: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(fallback) if arr.size == 0 else float(np.median(arr))


def _format_float_series(values: np.ndarray) -> str:
    return ",".join(f"{float(value):.12g}" for value in np.asarray(values, dtype=float))


def _format_vector_series(vectors: np.ndarray) -> str:
    return ";".join(
        ",".join(f"{float(value):.6g}" for value in row)
        for row in np.asarray(vectors, dtype=float)
    )

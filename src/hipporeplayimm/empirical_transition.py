"""Empirical behavioral-transition state-space replay model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .data import ReplaySession
from .encoding import EncodingModel, LogEmissionTensor, _clean_position, _speed_cm_s, _times_in_intervals
from .models import EventScore, _posterior_diagnostics
from .state_space_first_order import _forward_backward_first_order
from .state_space_utils import _mean_entropy


@dataclass
class EmpiricalTransitionStateSpaceReplayModel:
    """Exact first-order decoder using a transition matrix fit from behavior."""

    transition: csr_matrix
    name: str = "sorted-spike-state-space-empirical-transition"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        if emissions.n_time == 0:
            raise ValueError("emissions must contain at least one time bin")
        transition = _validated_transition_matrix(self.transition, emissions.n_bins)
        logp, trajectory = _forward_backward_first_order(emissions.log_likelihood, transition)
        terminal = trajectory[-1]
        diagnostics = {
            "state_space_mode": "empirical-transition",
            "state_space_observation_model": "sorted-spike-poisson",
            "state_space_trajectory_posterior": 1,
            "state_space_trajectory_time_bins": int(emissions.n_time),
            "state_space_empirical_transition_nonzeros": int(transition.nnz),
            "state_space_empirical_transition_density": float(transition.nnz / max(1, transition.shape[0] ** 2)),
            "state_space_empirical_evidence_support": "exact_full_grid",
            "mean_trajectory_posterior_entropy": _mean_entropy(trajectory),
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def _validated_transition_matrix(transition: csr_matrix, n_bins: int) -> csr_matrix:
    """Return a sparse column-stochastic transition matrix or reject it."""

    matrix = csr_matrix(transition)
    if matrix.shape != (n_bins, n_bins):
        raise ValueError("transition matrix shape must match emissions.n_bins")

    data = np.asarray(matrix.data, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("transition matrix entries must be finite")
    if np.any(data < 0.0):
        raise ValueError("transition matrix entries must be nonnegative")

    column_sums = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    if not np.all(np.isfinite(column_sums)):
        raise ValueError("transition matrix column sums must be finite")
    if not np.allclose(column_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("transition matrix columns must sum to 1")
    return matrix


def _adjacent_times_share_interval(times: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    """Return whether each adjacent timestamp pair lies in one common interval."""

    shared = np.zeros(max(times.shape[0] - 1, 0), dtype=bool)
    if shared.size == 0:
        return shared
    for start, end in intervals:
        shared |= (
            (times[:-1] >= start)
            & (times[:-1] <= end)
            & (times[1:] >= start)
            & (times[1:] <= end)
        )
    return shared


def _speed_cm_s_within_intervals(
    times: np.ndarray,
    xy: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    """Estimate speed independently inside each interval.

    A global finite difference lets samples from the next run bout influence the
    speed at the end of the current bout (and vice versa). Long inter-bout gaps
    can therefore make genuinely fast boundary samples look slow and remove
    otherwise valid within-bout transition counts.
    """

    speed = np.zeros(times.shape, dtype=float)
    for start, end in intervals:
        in_interval = (times >= start) & (times <= end)
        if np.any(in_interval):
            speed[in_interval] = _speed_cm_s(times[in_interval], xy[in_interval])
    return speed


def _finite_real_scalar(name: str, value: object) -> float:
    """Return a finite real scalar without silently coercing booleans or text."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real numeric scalar, not boolean")
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a real numeric scalar, not text")
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be a real numeric scalar, not complex")

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if raw.ndim != 0:
        raise ValueError(f"{name} must be a real numeric scalar")
    if raw.dtype.kind == "b":
        raise TypeError(f"{name} must be a real numeric scalar, not boolean")
    if raw.dtype.kind in {"S", "U"}:
        raise TypeError(f"{name} must be a real numeric scalar, not text")
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must be a real numeric scalar, not complex")
    if raw.dtype.kind == "O":
        item = raw.item()
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"{name} must be a real numeric scalar, not boolean")
        if isinstance(item, (str, bytes, bytearray)):
            raise TypeError(f"{name} must be a real numeric scalar, not text")
        if isinstance(item, (complex, np.complexfloating)):
            raise TypeError(f"{name} must be a real numeric scalar, not complex")

    try:
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def fit_empirical_transition_matrix(
    session: ReplaySession,
    encoding: EncodingModel,
    *,
    min_speed_cm_s: float | None = None,
    add_self_loop_count: float = 1.0,
    teleport_probability: float = 1e-6,
) -> csr_matrix:
    """Fit a column-stochastic transition matrix from consecutive run frames."""

    teleport = _finite_real_scalar("teleport_probability", teleport_probability)
    if not 0.0 <= teleport < 1.0:
        raise ValueError("teleport_probability must lie in [0, 1)")
    self_loop_count = _finite_real_scalar("add_self_loop_count", add_self_loop_count)
    if self_loop_count < 0.0:
        raise ValueError("add_self_loop_count must be finite and nonnegative")
    min_speed_value = encoding.config.min_speed_cm_s if min_speed_cm_s is None else min_speed_cm_s
    min_speed = _finite_real_scalar("min_speed_cm_s", min_speed_value)
    if min_speed < 0.0:
        raise ValueError("min_speed_cm_s must be finite and nonnegative")
    position = _clean_position(session.position)
    times = position[:, 0]
    if times.shape[0] > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("position times must be strictly increasing to fit empirical transitions")
    xy = position[:, 1:3]
    speed = _speed_cm_s_within_intervals(times, xy, session.run_times)
    in_run = _times_in_intervals(times, session.run_times)
    same_run_interval = _adjacent_times_share_interval(times, session.run_times)
    bins = encoding.positions_to_flat_bins(xy)
    valid = in_run & (speed >= min_speed) & (bins >= 0)
    n_bins = encoding.n_bins
    counts = np.zeros((n_bins, n_bins), dtype=float)
    if self_loop_count > 0.0:
        counts += np.eye(n_bins) * self_loop_count
    for idx in range(len(bins) - 1):
        if valid[idx] and valid[idx + 1] and same_run_interval[idx]:
            src = int(bins[idx])
            dst = int(bins[idx + 1])
            counts[dst, src] += 1.0
    empty_cols = counts.sum(axis=0) <= 0.0
    counts[empty_cols, empty_cols] = 1.0
    probs = counts / np.maximum(counts.sum(axis=0, keepdims=True), np.finfo(float).tiny)
    if teleport > 0.0:
        probs = (1.0 - teleport) * probs + teleport / n_bins
    return csr_matrix(probs)


def fit_empirical_transition_model(
    session: ReplaySession,
    encoding: EncodingModel,
    **kwargs,
) -> EmpiricalTransitionStateSpaceReplayModel:
    return EmpiricalTransitionStateSpaceReplayModel(
        transition=fit_empirical_transition_matrix(session, encoding, **kwargs)
    )

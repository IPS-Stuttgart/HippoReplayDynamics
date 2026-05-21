"""Opt-in accuracy upgrades for replay model evaluation.

This module intentionally avoids changing the default benchmark path.  It
collects utilities and lightweight model wrappers for higher-fidelity replay
analysis: arena-valid latent states, continuous-time emissions, replay-rate
calibration, empirical/contextual transition priors, reverse/bidirectional
hypotheses, posterior calibration diagnostics, adaptive event windows, and
observation-model ensembles.

The implementations are designed as reviewable building blocks.  They should be
validated on held-out behavior and synthetic recovery before they are used for
scientific claims on real replay events.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from types import SimpleNamespace
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.sparse import csr_matrix
from scipy.special import gammaln, logsumexp

from .data import ReplaySession, RippleEvent
from .encoding import (
    EmissionConfig,
    EncodingModel,
    LogEmissionTensor,
    _clean_position,
    _poisson_log_emissions,
    _positions_to_flat_bins,
    _speed_cm_s,
    _times_in_intervals,
    build_emissions,
)
from .models import EventScore, LOG_ZERO, _normalize_log_weights, _posterior_diagnostics
from .state_space_first_order import _forward_backward_first_order
from .state_space_utils import _as_log_probs, _mean_entropy, _pairwise_gaussian_log_prob


# ---------------------------------------------------------------------------
# 1. Valid-state masks and topology-aware transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidStateConfig:
    """Configuration for selecting physically/behaviorally plausible bins."""

    min_occupancy_s: float = 0.02
    keep_top_occupancy_fraction: float | None = None
    require_finite_rates: bool = True


def valid_state_mask_from_encoding(
    encoding: EncodingModel,
    config: ValidStateConfig | None = None,
) -> np.ndarray:
    """Return a boolean latent-state mask from occupancy and rate diagnostics."""

    config = ValidStateConfig() if config is None else config
    occupancy = np.asarray(encoding.occupancy_s, dtype=float)
    mask = np.isfinite(occupancy) & (occupancy >= float(config.min_occupancy_s))
    if config.keep_top_occupancy_fraction is not None:
        fraction = float(config.keep_top_occupancy_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("keep_top_occupancy_fraction must lie in (0, 1]")
        positive = occupancy[np.isfinite(occupancy) & (occupancy > 0.0)]
        if positive.size:
            threshold = float(np.quantile(positive, max(0.0, 1.0 - fraction)))
            mask &= occupancy >= threshold
    if config.require_finite_rates and encoding.rates_hz.size:
        mask &= np.all(np.isfinite(encoding.rates_hz), axis=0)
    if not np.any(mask):
        # Keep the single most occupied bin instead of returning an unusable
        # empty state space.
        mask[int(np.nanargmax(np.where(np.isfinite(occupancy), occupancy, -np.inf)))] = True
    return mask.astype(bool)


def restrict_encoding_to_mask(encoding: EncodingModel, valid_mask: np.ndarray) -> EncodingModel:
    """Return a shallow EncodingModel view restricted to valid spatial bins."""

    mask = _coerce_mask(valid_mask, encoding.n_bins)
    return EncodingModel(
        x_edges=encoding.x_edges,
        y_edges=encoding.y_edges,
        bin_centers=encoding.bin_centers[mask],
        rates_hz=encoding.rates_hz[:, mask],
        occupancy_s=encoding.occupancy_s[mask],
        cell_ids=encoding.cell_ids,
        config=encoding.config,
    )


def restrict_emissions_to_mask(emissions: LogEmissionTensor, valid_mask: np.ndarray) -> LogEmissionTensor:
    """Return a LogEmissionTensor restricted to valid spatial bins."""

    mask = _coerce_mask(valid_mask, emissions.n_bins)
    out = LogEmissionTensor(
        log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[:, mask],
        spike_counts=np.asarray(emissions.spike_counts).copy(),
        times=np.asarray(emissions.times, dtype=float).copy(),
        dt=emissions.dt,
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(emissions.n_spikes),
    )
    for name in ("metadata", "transition_durations"):
        if hasattr(emissions, name):
            setattr(out, name, getattr(emissions, name))
    return out


def masked_gaussian_transition_matrix(
    bin_centers: np.ndarray,
    sigma_cm: float,
    max_step_sigma: float = 4.0,
) -> csr_matrix:
    """Column-stochastic Gaussian transition over the supplied valid centers."""

    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[0] == 0:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    sigma_cm = float(sigma_cm)
    if sigma_cm <= 0.0:
        raise ValueError("sigma_cm must be positive")
    radius2 = (sigma_cm * float(max_step_sigma)) ** 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(centers):
        dist2 = np.sum((centers - center[None, :]) ** 2, axis=1)
        keep = dist2 <= radius2
        if not np.any(keep):
            keep[int(np.argmin(dist2))] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma_cm * sigma_cm))
        weights /= float(weights.sum())
        rows.extend(int(idx) for idx in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)
    return csr_matrix((data, (rows, cols)), shape=(centers.shape[0], centers.shape[0]))


def valid_grid_graph_transition(
    grid_shape: tuple[int, int],
    valid_mask: np.ndarray,
    *,
    diagonal_neighbors: bool = True,
    stay_probability: float = 0.0,
) -> csr_matrix:
    """Build a topology-aware transition over valid grid cells.

    Columns are source states and rows are destination states.  The returned
    matrix is defined in the compact valid-state index space.
    """

    nx, ny = (int(grid_shape[0]), int(grid_shape[1]))
    mask = _coerce_mask(valid_mask, nx * ny)
    valid_flat = np.flatnonzero(mask)
    compact_index = {int(flat): idx for idx, flat in enumerate(valid_flat)}
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal_neighbors:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    stay_probability = float(stay_probability)
    if not 0.0 <= stay_probability < 1.0:
        raise ValueError("stay_probability must lie in [0, 1)")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src_compact, flat in enumerate(valid_flat):
        x = int(flat // ny)
        y = int(flat % ny)
        neighbors: list[int] = []
        for dx, dy in offsets:
            xx, yy = x + dx, y + dy
            if 0 <= xx < nx and 0 <= yy < ny:
                neighbor_flat = xx * ny + yy
                if mask[neighbor_flat]:
                    neighbors.append(compact_index[int(neighbor_flat)])
        if not neighbors:
            neighbors = [src_compact]
        move_probability = 1.0 - stay_probability
        for dst in neighbors:
            rows.append(dst)
            cols.append(src_compact)
            data.append(move_probability / len(neighbors))
        if stay_probability > 0.0:
            rows.append(src_compact)
            cols.append(src_compact)
            data.append(stay_probability)
    return csr_matrix((data, (rows, cols)), shape=(len(valid_flat), len(valid_flat)))


@dataclass
class ValidStateDiffusionReplayModel:
    """Exact first-order diffusion model restricted to a valid-state mask."""

    valid_mask: np.ndarray
    sigma_cm: float = 5.0
    max_step_sigma: float = 4.0
    name: str = "valid-state-diffusion"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        mask = _coerce_mask(self.valid_mask, emissions.n_bins)
        restricted = restrict_emissions_to_mask(emissions, mask)
        centers = np.asarray(bin_centers, dtype=float)[mask]
        transition = masked_gaussian_transition_matrix(centers, self.sigma_cm, self.max_step_sigma)
        logp, trajectory = _forward_backward_first_order(restricted.log_likelihood, transition)
        full_trajectory = _expand_log_trajectory(trajectory, mask, emissions.n_bins)
        terminal = full_trajectory[-1]
        diagnostics = {
            "valid_state_bins": int(np.sum(mask)),
            "valid_state_fraction": float(np.mean(mask)),
            "valid_state_transition": "gaussian_masked",
            "valid_state_sigma_cm": float(self.sigma_cm),
            "mean_trajectory_posterior_entropy": _mean_entropy(full_trajectory),
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=full_trajectory,
        )


@dataclass
class ValidStateGridReplayModel:
    """Exact first-order graph-walk model over behaviorally valid grid states."""

    valid_mask: np.ndarray
    grid_shape: tuple[int, int]
    diagonal_neighbors: bool = True
    stay_probability: float = 0.0
    name: str = "valid-state-grid"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        mask = _coerce_mask(self.valid_mask, emissions.n_bins)
        restricted = restrict_emissions_to_mask(emissions, mask)
        transition = valid_grid_graph_transition(
            self.grid_shape,
            mask,
            diagonal_neighbors=bool(self.diagonal_neighbors),
            stay_probability=float(self.stay_probability),
        )
        logp, trajectory = _forward_backward_first_order(restricted.log_likelihood, transition)
        full_trajectory = _expand_log_trajectory(trajectory, mask, emissions.n_bins)
        terminal = full_trajectory[-1]
        diagnostics = {
            "valid_state_bins": int(np.sum(mask)),
            "valid_state_fraction": float(np.mean(mask)),
            "valid_state_transition": "grid_graph",
            "valid_state_grid_shape_x": int(self.grid_shape[0]),
            "valid_state_grid_shape_y": int(self.grid_shape[1]),
            "valid_state_grid_diagonal_neighbors": int(bool(self.diagonal_neighbors)),
            "valid_state_grid_stay_probability": float(self.stay_probability),
            "mean_trajectory_posterior_entropy": _mean_entropy(full_trajectory),
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name,
            float(logp),
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=full_trajectory,
        )


# ---------------------------------------------------------------------------
# 2. Continuous-time point-process emissions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousTimeEmissionConfig:
    """Build emissions from spike-time intervals instead of fixed bins."""

    spike_rate_scale: float = 1.0
    min_interval_s: float = 1e-6
    include_terminal_no_spike_interval: bool = True


def build_continuous_time_emissions(
    session: ReplaySession,
    encoding: EncodingModel,
    ripple: RippleEvent | int,
    config: ContinuousTimeEmissionConfig | None = None,
) -> LogEmissionTensor:
    """Construct point-process emissions from inter-spike/no-spike intervals.

    Each row represents the likelihood contribution of one interval.  If one or
    more spikes occur at the interval endpoint, their cell counts are included in
    that row along with the survival term over the elapsed interval.
    """

    config = ContinuousTimeEmissionConfig() if config is None else config
    if config.spike_rate_scale <= 0.0:
        raise ValueError("spike_rate_scale must be positive")
    if config.min_interval_s <= 0.0:
        raise ValueError("min_interval_s must be positive")
    ripple_event = session.ripple(ripple) if isinstance(ripple, int) else ripple
    start = float(ripple_event.start)
    end = float(ripple_event.end)
    if end <= start:
        raise ValueError("ripple end must be greater than ripple start")

    spikes = np.asarray(session.spikes, dtype=float)
    if spikes.size:
        keep = (
            (spikes[:, 0] >= start)
            & (spikes[:, 0] < end)
            & np.isin(spikes[:, 1].astype(int), encoding.cell_ids)
        )
        event_spikes = spikes[keep]
        order = np.argsort(event_spikes[:, 0], kind="mergesort")
        event_spikes = event_spikes[order]
    else:
        event_spikes = np.empty((0, 2), dtype=float)

    cell_to_col = {int(cell_id): idx for idx, cell_id in enumerate(encoding.cell_ids)}
    rows: list[np.ndarray] = []
    durations: list[float] = []
    times: list[float] = []
    cursor = start
    spike_index = 0
    while spike_index < event_spikes.shape[0]:
        spike_time = float(event_spikes[spike_index, 0])
        dt = max(spike_time - cursor, float(config.min_interval_s))
        counts = np.zeros(encoding.n_cells, dtype=int)
        while spike_index < event_spikes.shape[0] and np.isclose(
            float(event_spikes[spike_index, 0]), spike_time, rtol=0.0, atol=1e-12
        ):
            col = cell_to_col.get(int(event_spikes[spike_index, 1]))
            if col is not None:
                counts[col] += 1
            spike_index += 1
        rows.append(counts)
        durations.append(dt)
        times.append(spike_time)
        cursor = spike_time

    terminal_dt = end - cursor
    if config.include_terminal_no_spike_interval or not rows:
        rows.append(np.zeros(encoding.n_cells, dtype=int))
        durations.append(max(terminal_dt, float(config.min_interval_s)))
        times.append(end)

    spike_counts = np.vstack(rows) if rows else np.zeros((0, encoding.n_cells), dtype=int)
    durations_arr = np.asarray(durations, dtype=float)
    log_likelihood = _poisson_log_emissions(
        spike_counts,
        encoding.rates_hz,
        durations_arr,
        spike_rate_scale=float(config.spike_rate_scale),
    )
    times_arr = np.asarray(times, dtype=float)
    emissions = LogEmissionTensor(
        log_likelihood=log_likelihood,
        spike_counts=spike_counts,
        times=times_arr,
        dt=float(np.median(durations_arr)) if durations_arr.size else float(config.min_interval_s),
        cell_ids=np.asarray(encoding.cell_ids, dtype=int).copy(),
        n_spikes=int(spike_counts.sum()),
    )
    if emissions.n_time > 1:
        emissions.transition_durations = np.maximum(np.diff(times_arr), float(config.min_interval_s))
    emissions.metadata = {
        "emission_model": "continuous-time-binned-at-spikes",
        "continuous_time_intervals": int(emissions.n_time),
        "continuous_time_min_interval_s": float(config.min_interval_s),
    }
    return emissions


# ---------------------------------------------------------------------------
# 3 and 12. Replay gain, overdispersion, and Bayesian place-field uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayGainConfig:
    prior_observed_spikes: float = 1.0
    prior_expected_spikes: float = 1.0
    min_gain: float = 0.1
    max_gain: float = 10.0


def estimate_replay_cell_gains(
    session: ReplaySession,
    encoding: EncodingModel,
    ripple_indices: Iterable[int],
    config: ReplayGainConfig | None = None,
) -> np.ndarray:
    """Estimate per-cell replay gain against a uniform spatial prior."""

    config = ReplayGainConfig() if config is None else config
    observed = np.zeros(encoding.n_cells, dtype=float)
    total_duration = 0.0
    for event_index in ripple_indices:
        event = session.ripple(int(event_index))
        total_duration += max(float(event.end) - float(event.start), 0.0)
        spikes = np.asarray(session.spikes, dtype=float)
        if spikes.size == 0:
            continue
        keep = (
            (spikes[:, 0] >= event.start)
            & (spikes[:, 0] < event.end)
            & np.isin(spikes[:, 1].astype(int), encoding.cell_ids)
        )
        rows = np.searchsorted(encoding.cell_ids, spikes[keep, 1].astype(int))
        valid = (rows >= 0) & (rows < encoding.n_cells)
        np.add.at(observed, rows[valid], 1.0)
    spatial_mean_rate = np.mean(np.asarray(encoding.rates_hz, dtype=float), axis=1)
    expected = spatial_mean_rate * max(total_duration, np.finfo(float).tiny)
    gains = (observed + config.prior_observed_spikes) / (expected + config.prior_expected_spikes)
    return np.clip(gains, float(config.min_gain), float(config.max_gain))


def apply_cell_gains(encoding: EncodingModel, gains: np.ndarray) -> EncodingModel:
    """Return an encoding with per-cell multiplicative gains applied."""

    gains = np.asarray(gains, dtype=float)
    if gains.shape != (encoding.n_cells,):
        raise ValueError(f"gains must have shape {(encoding.n_cells,)}, got {gains.shape}")
    return replace(encoding, rates_hz=encoding.rates_hz * gains[:, None])


def negative_binomial_log_emissions(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    dt: float | np.ndarray,
    *,
    overdispersion: float,
    spike_rate_scale: float = 1.0,
) -> np.ndarray:
    """Negative-binomial predictive log emissions with mean rate * dt.

    ``overdispersion`` is alpha in Var[count] = mu + alpha * mu^2.  The alpha=0
    limit is the Poisson emission used by the base package.
    """

    alpha = float(overdispersion)
    if alpha < 0.0:
        raise ValueError("overdispersion must be non-negative")
    if alpha <= np.finfo(float).eps:
        return _poisson_log_emissions(spike_counts, rates_hz, dt, spike_rate_scale=spike_rate_scale)
    counts = np.asarray(spike_counts, dtype=float)
    dt_array = np.asarray(dt, dtype=float)
    if dt_array.ndim == 0:
        mu = rates_hz[None, :, :] * float(dt_array) * float(spike_rate_scale)
    else:
        if dt_array.shape != (counts.shape[0],):
            raise ValueError("dt must be scalar or one duration per time bin")
        mu = dt_array[:, None, None] * rates_hz[None, :, :] * float(spike_rate_scale)
    mu = np.maximum(mu, np.finfo(float).tiny)
    size = 1.0 / alpha
    logp_success = np.log(size / (size + mu))
    logp_failure = np.log(mu / (size + mu))
    return np.sum(
        gammaln(counts[:, :, None] + size)
        - gammaln(size)
        - gammaln(counts[:, :, None] + 1.0)
        + size * logp_success
        + counts[:, :, None] * logp_failure,
        axis=1,
    )


def gamma_poisson_predictive_log_emissions(
    spike_counts: np.ndarray,
    rate_shape: np.ndarray,
    rate_exposure_s: np.ndarray,
    dt: float | np.ndarray,
    *,
    spike_rate_scale: float = 1.0,
) -> np.ndarray:
    """Gamma-Poisson predictive emissions integrating place-field uncertainty."""

    counts = np.asarray(spike_counts, dtype=float)
    shape = np.asarray(rate_shape, dtype=float)
    exposure = np.asarray(rate_exposure_s, dtype=float)
    if shape.shape != exposure.shape:
        raise ValueError("rate_shape and rate_exposure_s must have matching shapes")
    if shape.ndim != 2 or shape.shape[0] != counts.shape[1]:
        raise ValueError("rate prior arrays must have shape (n_cells, n_bins)")
    dt_array = np.asarray(dt, dtype=float)
    if dt_array.ndim == 0:
        trial_exposure = np.full(counts.shape[0], float(dt_array) * float(spike_rate_scale))
    else:
        trial_exposure = dt_array * float(spike_rate_scale)
    out = np.zeros((counts.shape[0], shape.shape[1]), dtype=float)
    beta = np.maximum(exposure, np.finfo(float).tiny)
    for time_index, trial_dt in enumerate(trial_exposure):
        k = counts[time_index][:, None]
        out[time_index] = np.sum(
            gammaln(shape + k)
            - gammaln(shape)
            - gammaln(k + 1.0)
            + shape * np.log(beta / (beta + trial_dt))
            + k * np.log(trial_dt / (beta + trial_dt)),
            axis=0,
        )
    return out


def gamma_rate_prior_from_encoding(
    encoding: EncodingModel,
    *,
    prior_count: float = 1.0,
    prior_exposure_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate Gamma posterior parameters from an EncodingModel."""

    occupancy = np.maximum(np.asarray(encoding.occupancy_s, dtype=float), 0.0)
    counts = np.asarray(encoding.rates_hz, dtype=float) * occupancy[None, :]
    shape = counts + float(prior_count)
    exposure = occupancy[None, :] + float(prior_exposure_s)
    return shape, np.broadcast_to(exposure, shape.shape).copy()


# ---------------------------------------------------------------------------
# 4 and 13. Calibration diagnostics
# ---------------------------------------------------------------------------


def summarize_behavior_position_calibration(samples: pd.DataFrame) -> pd.DataFrame:
    """Summarize posterior calibration from position-validation sample rows."""

    if samples.empty:
        return pd.DataFrame()
    frame = samples.copy()
    frame["true_bin_log_probability"] = np.log(
        np.maximum(frame.get("true_bin_probability", np.nan).astype(float), np.finfo(float).tiny)
    )
    grouped = frame.groupby("session", as_index=False)
    return grouped.agg(
        windows=("window_index", "count"),
        median_posterior_mean_error_cm=("posterior_mean_error_cm", "median"),
        median_map_error_cm=("map_error_cm", "median"),
        mean_true_bin_probability=("true_bin_probability", "mean"),
        median_true_bin_rank=("true_bin_rank", "median"),
        mean_true_bin_log_probability=("true_bin_log_probability", "mean"),
        mean_posterior_entropy=("posterior_entropy", "mean"),
    )


def model_probability_diagnostics(
    scores: pd.DataFrame,
    *,
    evidence_column: str = "log_evidence",
    group_columns: Sequence[str] = ("session", "event_index"),
) -> pd.DataFrame:
    """Return per-event model-probability entropy and evidence margins."""

    if scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in scores.groupby(list(group_columns), sort=False):
        ok = group.copy()
        if "status" in ok:
            ok = ok[ok["status"].eq("success")]
        if "evidence_comparable" in ok:
            ok = ok[ok["evidence_comparable"].fillna(False).astype(bool)]
        values = ok[evidence_column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        ordered = np.sort(values)[::-1]
        probs = np.exp(values - logsumexp(values))
        best_idx = int(np.argmax(ok[evidence_column].to_numpy(dtype=float)))
        row = _group_key_dict(group_columns, key)
        row.update(
            {
                "models": int(values.size),
                "best_model": str(ok.iloc[best_idx]["model"]),
                "best_log_evidence": float(ordered[0]),
                "evidence_margin_to_second_best": float(ordered[0] - ordered[1]) if values.size > 1 else np.nan,
                "model_probability_entropy": float(-np.sum(probs * np.log(np.maximum(probs, np.finfo(float).tiny)))),
                "best_model_probability": float(np.max(probs)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_model_win_probabilities(
    scores: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    random_seed: int = 1,
    evidence_column: str = "log_evidence",
    group_columns: Sequence[str] = ("session", "event_index"),
) -> pd.DataFrame:
    """Bootstrap session/event rows to estimate model win uncertainty."""

    if scores.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_seed)
    pivot = scores.pivot_table(index=list(group_columns), columns="model", values=evidence_column, aggfunc="first")
    pivot = pivot.dropna(axis=0, how="all")
    if pivot.empty:
        return pd.DataFrame()
    models = list(pivot.columns)
    win_counts = dict.fromkeys(models, 0)
    values = pivot.to_numpy(dtype=float)
    for _ in range(int(n_bootstrap)):
        sample_indices = rng.integers(0, values.shape[0], size=values.shape[0])
        sample = values[sample_indices]
        means = np.nanmean(sample, axis=0)
        win_counts[models[int(np.nanargmax(means))]] += 1
    return pd.DataFrame(
        {"model": model, "bootstrap_win_probability": count / float(n_bootstrap)}
        for model, count in win_counts.items()
    ).sort_values("bootstrap_win_probability", ascending=False)


# ---------------------------------------------------------------------------
# 5 and 6. Leave-rat/session splits and empirical/contextual dynamics
# ---------------------------------------------------------------------------


def rat_id_from_session_id(session_id: str) -> str:
    return str(session_id).replace("\\", "/").split("/", 1)[0]


def leave_one_rat_splits(session_ids: Iterable[str]) -> list[dict[str, object]]:
    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids))
    rats = sorted({rat_id_from_session_id(session_id) for session_id in ids})
    return [
        {
            "heldout_rat": rat,
            "train_sessions": tuple(session_id for session_id in ids if rat_id_from_session_id(session_id) != rat),
            "test_sessions": tuple(session_id for session_id in ids if rat_id_from_session_id(session_id) == rat),
        }
        for rat in rats
    ]


def leave_one_session_splits(session_ids: Iterable[str]) -> list[dict[str, object]]:
    ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids))
    return [
        {
            "heldout_session": session_id,
            "train_sessions": tuple(other for other in ids if other != session_id),
            "test_sessions": (session_id,),
        }
        for session_id in ids
    ]


def fit_empirical_transition_matrix(
    session: ReplaySession,
    encoding: EncodingModel,
    *,
    min_speed_cm_s: float | None = None,
    lag_frames: int = 1,
    smoothing_count: float = 1e-3,
    valid_mask: np.ndarray | None = None,
) -> csr_matrix:
    """Fit a column-stochastic transition prior from run behavior."""

    if lag_frames < 1:
        raise ValueError("lag_frames must be at least one")
    position = _clean_position(session.position)
    times = position[:, 0]
    xy = position[:, 1:3]
    speed = _speed_cm_s(times, xy)
    min_speed = encoding.config.min_speed_cm_s if min_speed_cm_s is None else float(min_speed_cm_s)
    movement = _times_in_intervals(times, session.run_times) & (speed >= min_speed)
    bins = _positions_to_flat_bins(xy, encoding.x_edges, encoding.y_edges)
    mask = np.ones(encoding.n_bins, dtype=bool) if valid_mask is None else _coerce_mask(valid_mask, encoding.n_bins)
    counts = np.zeros((encoding.n_bins, encoding.n_bins), dtype=float)
    source = bins[:-lag_frames]
    dest = bins[lag_frames:]
    keep = movement[:-lag_frames] & movement[lag_frames:] & (source >= 0) & (dest >= 0)
    keep &= mask[source] & mask[dest]
    np.add.at(counts, (dest[keep].astype(int), source[keep].astype(int)), 1.0)
    if smoothing_count > 0.0:
        valid = np.flatnonzero(mask)
        counts[np.ix_(valid, valid)] += float(smoothing_count)
    column_sums = counts.sum(axis=0)
    empty = column_sums <= 0.0
    if np.any(empty):
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("valid_mask excludes all states")
        counts[np.ix_(valid, np.flatnonzero(empty))] = 1.0 / valid.size
        column_sums = counts.sum(axis=0)
    probabilities = counts / np.maximum(column_sums[None, :], np.finfo(float).tiny)
    return csr_matrix(probabilities)


@dataclass
class EmpiricalTransitionReplayModel:
    """Exact first-order state-space model using a fitted behavioral transition."""

    transition: csr_matrix
    name: str = "empirical-transition"

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        if self.transition.shape != (emissions.n_bins, emissions.n_bins):
            raise ValueError("transition shape must match emissions.n_bins")
        logp, trajectory = _forward_backward_first_order(emissions.log_likelihood, self.transition)
        terminal = trajectory[-1]
        diagnostics = {
            "state_space_mode": "empirical-transition",
            "state_space_transition_nonzeros": int(self.transition.nnz),
            "state_space_trajectory_posterior": 1,
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


# ---------------------------------------------------------------------------
# 7. Explicit reverse and bidirectional hypotheses
# ---------------------------------------------------------------------------


class ReplayScorer(Protocol):
    name: str

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        ...


@dataclass
class TimeReversedReplayModel:
    """Score a base model under reversed emission order."""

    base_model: ReplayScorer
    name: str | None = None

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        reversed_emissions = reverse_emissions(emissions)
        base_score = self.base_model.score(reversed_emissions, bin_centers)
        trajectory = None
        terminal = base_score.terminal_log_posterior
        if base_score.trajectory_log_posterior is not None:
            trajectory = base_score.trajectory_log_posterior[::-1].copy()
            terminal = trajectory[-1]
        diagnostics = dict(base_score.diagnostics)
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        diagnostics["direction_hypothesis"] = "reverse"
        diagnostics["base_model"] = str(base_score.model_name)
        return EventScore(
            self.name or f"reverse-{base_score.model_name}",
            base_score.log_likelihood,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


@dataclass
class BidirectionalMixtureReplayModel:
    """Equal-prior mixture of forward and reverse base-model hypotheses."""

    base_model: ReplayScorer
    name: str | None = None

    def score(self, emissions: LogEmissionTensor, bin_centers: np.ndarray) -> EventScore:
        forward = self.base_model.score(emissions, bin_centers)
        reverse = TimeReversedReplayModel(self.base_model).score(emissions, bin_centers)
        weights = np.asarray([forward.log_likelihood, reverse.log_likelihood], dtype=float)
        logp = float(logsumexp(weights - np.log(2.0)))
        posterior = np.exp(weights - logsumexp(weights))
        terminal = _mix_log_posteriors(
            [forward.terminal_log_posterior, reverse.terminal_log_posterior],
            posterior,
        )
        trajectory = None
        if forward.trajectory_log_posterior is not None and reverse.trajectory_log_posterior is not None:
            trajectory = _mix_log_trajectories(
                [forward.trajectory_log_posterior, reverse.trajectory_log_posterior],
                posterior,
            )
        diagnostics = {
            "direction_hypothesis": "bidirectional-mixture",
            "base_model": str(forward.model_name),
            "forward_log_evidence": float(forward.log_likelihood),
            "reverse_log_evidence": float(reverse.log_likelihood),
            "forward_direction_probability": float(posterior[0]),
            "reverse_direction_probability": float(posterior[1]),
        }
        diagnostics.update(_posterior_diagnostics(terminal, bin_centers))
        return EventScore(
            self.name or f"bidirectional-{forward.model_name}",
            logp,
            emissions.n_time,
            emissions.n_spikes,
            diagnostics=diagnostics,
            terminal_log_posterior=terminal,
            trajectory_log_posterior=trajectory,
        )


def reverse_emissions(emissions: LogEmissionTensor) -> LogEmissionTensor:
    out = LogEmissionTensor(
        log_likelihood=np.asarray(emissions.log_likelihood, dtype=float)[::-1].copy(),
        spike_counts=np.asarray(emissions.spike_counts)[::-1].copy(),
        times=np.asarray(emissions.times, dtype=float)[::-1].copy(),
        dt=emissions.dt,
        cell_ids=np.asarray(emissions.cell_ids).copy(),
        n_spikes=int(emissions.n_spikes),
    )
    if hasattr(emissions, "metadata"):
        out.metadata = dict(getattr(emissions, "metadata"))
    if hasattr(emissions, "transition_durations"):
        out.transition_durations = np.asarray(getattr(emissions, "transition_durations"), dtype=float)[::-1].copy()
    return out


# ---------------------------------------------------------------------------
# 8. Improved behavioral context proxy labels
# ---------------------------------------------------------------------------


def behavioral_well_context(
    session: ReplaySession,
    event_index: int,
    *,
    visit_radius_cm: float = 10.0,
    pre_window_s: float = 5.0,
    post_window_s: float = 30.0,
) -> dict[str, object]:
    """Return pre/post well and route-context labels for one ripple event."""

    from .ground_truth import infer_well_locations

    wells = infer_well_locations(session)
    event = session.ripple(int(event_index))
    if wells.empty:
        return {
            "session": session.session_id,
            "event_index": int(event_index),
            "pre_well": "",
            "post_well": "",
            "route_label": "",
            "context_valid": False,
        }
    position = _clean_position(session.position)
    pre = _nearest_well_visit(
        position,
        wells,
        start=float(event.start) - float(pre_window_s),
        end=float(event.start),
        visit_radius_cm=visit_radius_cm,
        prefer="latest",
    )
    post = _nearest_well_visit(
        position,
        wells,
        start=float(event.end),
        end=float(event.end) + float(post_window_s),
        visit_radius_cm=visit_radius_cm,
        prefer="earliest",
    )
    pre_label = "" if pre is None else str(pre["well_id"])
    post_label = "" if post is None else str(post["well_id"])
    route_label = f"{pre_label}->{post_label}" if pre_label and post_label else ""
    return {
        "session": session.session_id,
        "event_index": int(event_index),
        "pre_well": pre_label,
        "post_well": post_label,
        "route_label": route_label,
        "pre_well_time_s": np.nan if pre is None else float(pre["time"]),
        "post_well_time_s": np.nan if post is None else float(post["time"]),
        "time_since_pre_well_s": np.nan if pre is None else float(event.start) - float(pre["time"]),
        "time_to_post_well_s": np.nan if post is None else float(post["time"]) - float(event.end),
        "context_valid": bool(pre_label or post_label),
    }


# ---------------------------------------------------------------------------
# 9. Position tracking uncertainty and robust filtering
# ---------------------------------------------------------------------------


def robust_position_filter(
    position: np.ndarray,
    *,
    median_window: int = 5,
    max_speed_cm_s: float = 300.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Median-filter position and flag implausible speed jumps."""

    arr = _clean_position(position)
    if arr.size == 0:
        return arr, pd.DataFrame()
    window = max(1, int(median_window))
    if window % 2 == 0:
        window += 1
    filtered = arr.copy()
    if window > 1:
        filtered[:, 1] = median_filter(filtered[:, 1], size=window, mode="nearest")
        filtered[:, 2] = median_filter(filtered[:, 2], size=window, mode="nearest")
    speed = _speed_cm_s(filtered[:, 0], filtered[:, 1:3])
    high_speed = speed > float(max_speed_cm_s)
    diagnostics = pd.DataFrame(
        {
            "time": filtered[:, 0],
            "speed_cm_s": speed,
            "position_high_speed_flag": high_speed,
            "position_quality_weight": np.where(high_speed, 0.0, 1.0),
        }
    )
    return filtered, diagnostics


# ---------------------------------------------------------------------------
# 10. Tetrode/channel-aware clusterless helpers
# ---------------------------------------------------------------------------


def infer_tetrode_ids_from_feature_names(feature_names: Sequence[str]) -> np.ndarray:
    """Infer tetrode IDs from mark feature names when naming conventions allow."""

    ids: list[int] = []
    for index, name in enumerate(feature_names):
        text = str(name).lower().replace("-", "_")
        tetrode = None
        for token in text.split("_"):
            if token.startswith("tt") and token[2:].isdigit():
                tetrode = int(token[2:])
                break
            if token.startswith("tetrode") and token.replace("tetrode", "").isdigit():
                tetrode = int(token.replace("tetrode", ""))
                break
        ids.append(index if tetrode is None else tetrode)
    return np.asarray(ids, dtype=int)


def summarize_tetrode_mark_partitions(marks) -> pd.DataFrame:
    """Summarize available mark features by inferred tetrode ID."""

    feature_names = tuple(getattr(marks, "feature_names", ()) or ())
    if not feature_names:
        return pd.DataFrame(columns=["tetrode_id", "features"])
    tetrode_ids = infer_tetrode_ids_from_feature_names(feature_names)
    rows = []
    for tetrode_id in sorted(np.unique(tetrode_ids)):
        feature_idx = np.flatnonzero(tetrode_ids == tetrode_id)
        rows.append(
            {
                "tetrode_id": int(tetrode_id),
                "features": ",".join(str(feature_names[i]) for i in feature_idx),
                "n_features": int(feature_idx.size),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11. Exact small-grid second-order evidence and pruning-gap checks
# ---------------------------------------------------------------------------


def exact_second_order_momentum_log_evidence(
    emissions: LogEmissionTensor,
    bin_centers: np.ndarray,
    *,
    sigma_cm: float,
    initial_sigma_cm: float | None = None,
    velocity_decay: float = 0.95,
    max_bins: int = 192,
) -> float:
    """Exact full-grid second-order momentum evidence for small grids."""

    centers = np.asarray(bin_centers, dtype=float)
    n_bins = centers.shape[0]
    if n_bins > int(max_bins):
        raise ValueError(f"Exact second-order evidence is limited to {max_bins} bins; got {n_bins}")
    if emissions.n_time == 1:
        return float(logsumexp(emissions.log_likelihood[0]) - np.log(n_bins))
    initial_sigma = float(initial_sigma_cm if initial_sigma_cm is not None else sigma_cm)
    first_kernel = _full_grid_pairwise_gaussian_log_prob(centers, centers, initial_sigma)
    pair = (
        emissions.log_likelihood[0, :, None]
        - np.log(n_bins)
        + first_kernel
        + emissions.log_likelihood[1, None, :]
    )
    for time_index in range(2, emissions.n_time):
        next_pair = np.full((n_bins, n_bins), LOG_ZERO, dtype=float)
        curr_emission = emissions.log_likelihood[time_index]
        for prev in range(n_bins):
            predictions = centers[prev][None, :] + float(velocity_decay) * (centers[prev][None, :] - centers)
            kernel = _full_grid_pairwise_gaussian_log_prob(predictions, centers, float(sigma_cm))
            next_pair[prev] = logsumexp(pair[:, prev][:, None] + kernel, axis=0) + curr_emission
        pair = next_pair
    return float(logsumexp(pair))


def pruning_gap_report(
    exact_log_evidence: float,
    pruned_log_evidence: float,
) -> dict[str, float]:
    """Return a compact exact-vs-pruned evidence gap report."""

    exact = float(exact_log_evidence)
    pruned = float(pruned_log_evidence)
    return {
        "exact_log_evidence": exact,
        "pruned_log_evidence": pruned,
        "pruning_gap_log_units": exact - pruned,
        "pruned_evidence_fraction": float(np.exp(min(0.0, pruned - exact))) if isfinite(exact) and isfinite(pruned) else np.nan,
    }


# ---------------------------------------------------------------------------
# 14. Event-window sensitivity and adaptive segmentation
# ---------------------------------------------------------------------------


def candidate_event_windows(
    ripple_event: RippleEvent,
    *,
    pre_pads_s: Sequence[float] = (0.0, 0.005, 0.010),
    post_pads_s: Sequence[float] = (0.0, 0.005, 0.010),
    min_duration_s: float = 0.005,
) -> pd.DataFrame:
    """Generate candidate replay windows around a ripple event."""

    rows = []
    for pre in pre_pads_s:
        for post in post_pads_s:
            start = float(ripple_event.start) - float(pre)
            end = float(ripple_event.end) + float(post)
            if end - start >= float(min_duration_s):
                rows.append(
                    {
                        "pre_pad_s": float(pre),
                        "post_pad_s": float(post),
                        "window_start_s": start,
                        "window_end_s": end,
                        "window_duration_s": end - start,
                    }
                )
    return pd.DataFrame(rows)


def build_emissions_for_window(
    session: ReplaySession,
    encoding: EncodingModel,
    start: float,
    end: float,
    config: EmissionConfig | None = None,
) -> LogEmissionTensor:
    """Build ordinary binned emissions for an arbitrary start/end window."""

    event = SimpleNamespace(start=float(start), end=float(end))
    return build_emissions(session, encoding, event, config)


# ---------------------------------------------------------------------------
# 15. Observation-model ensembles
# ---------------------------------------------------------------------------


def weighted_ensemble_emissions(
    left: LogEmissionTensor,
    right: LogEmissionTensor,
    *,
    alpha: float = 0.5,
) -> LogEmissionTensor:
    """Combine two aligned emission tensors by a weighted product of experts."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if left.log_likelihood.shape != right.log_likelihood.shape:
        raise ValueError("emission tensors must have matching log_likelihood shapes")
    out = LogEmissionTensor(
        log_likelihood=alpha * left.log_likelihood + (1.0 - alpha) * right.log_likelihood,
        spike_counts=np.asarray(left.spike_counts).copy(),
        times=np.asarray(left.times, dtype=float).copy(),
        dt=left.dt,
        cell_ids=np.asarray(left.cell_ids).copy(),
        n_spikes=int(left.n_spikes + right.n_spikes),
    )
    out.metadata = {
        "emission_model": "weighted-product-ensemble",
        "ensemble_alpha_left": alpha,
    }
    if hasattr(left, "transition_durations"):
        out.transition_durations = getattr(left, "transition_durations")
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def event_reliability_flags(
    row: Mapping[str, object],
    *,
    min_spikes: int = 5,
    min_time_bins: int = 2,
    max_entropy: float | None = None,
    min_candidate_log_mass: float | None = None,
) -> dict[str, object]:
    """Convert event/model diagnostics into reliability flags."""

    reasons: list[str] = []
    n_spikes = _as_int(row.get("n_spikes", row.get("test_spikes", 0)))
    n_time = _as_int(row.get("n_time", 0))
    if n_spikes < int(min_spikes):
        reasons.append("low_spike_count")
    if n_time < int(min_time_bins):
        reasons.append("too_few_time_bins")
    if max_entropy is not None:
        entropy = _as_float(row.get("diagnostic_terminal_posterior_entropy", np.nan))
        if np.isfinite(entropy) and entropy > float(max_entropy):
            reasons.append("high_terminal_entropy")
    if min_candidate_log_mass is not None:
        mass = _as_float(row.get("diagnostic_mean_candidate_log_mass", np.nan))
        if np.isfinite(mass) and mass < float(min_candidate_log_mass):
            reasons.append("low_candidate_mass")
    return {
        "event_reliable": not reasons,
        "event_reliability_reasons": ",".join(reasons),
    }


def _coerce_mask(mask: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (int(n_bins),):
        raise ValueError(f"mask must have shape {(int(n_bins),)}, got {arr.shape}")
    if not np.any(arr):
        raise ValueError("mask must keep at least one bin")
    return arr


def _expand_log_trajectory(trajectory: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full((trajectory.shape[0], int(n_bins)), LOG_ZERO, dtype=float)
    out[:, mask] = trajectory
    return out


def _full_grid_pairwise_gaussian_log_prob(predicted: np.ndarray, observed: np.ndarray, sigma_cm: float) -> np.ndarray:
    log_kernel = _pairwise_gaussian_log_prob(predicted, observed, float(sigma_cm))
    return log_kernel - logsumexp(log_kernel, axis=1, keepdims=True)


def _mix_log_posteriors(posteriors: Sequence[np.ndarray | None], weights: np.ndarray) -> np.ndarray:
    valid = [(posterior, float(weight)) for posterior, weight in zip(posteriors, weights, strict=False) if posterior is not None]
    if not valid:
        raise ValueError("at least one posterior is required")
    stacked = np.stack([posterior + np.log(weight) for posterior, weight in valid], axis=0)
    return _normalize_log_weights(logsumexp(stacked, axis=0))


def _mix_log_trajectories(trajectories: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack([trajectory + np.log(float(weight)) for trajectory, weight in zip(trajectories, weights, strict=True)], axis=0)
    mixed = logsumexp(stacked, axis=0)
    return _as_log_probs(np.exp(mixed))


def _nearest_well_visit(
    position: np.ndarray,
    wells: pd.DataFrame,
    *,
    start: float,
    end: float,
    visit_radius_cm: float,
    prefer: str,
) -> dict[str, object] | None:
    keep = (position[:, 0] >= float(start)) & (position[:, 0] <= float(end))
    if not np.any(keep):
        return None
    samples = position[keep]
    well_xy = wells[["well_x", "well_y"]].to_numpy(dtype=float)
    labels = wells.get("well_id", pd.Series(np.arange(len(wells)))).to_numpy()
    distances = np.sqrt(np.sum((samples[:, None, 1:3] - well_xy[None, :, :]) ** 2, axis=2))
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(samples.shape[0]), nearest]
    close = nearest_distance <= float(visit_radius_cm)
    if not np.any(close):
        return None
    idx = int(np.flatnonzero(close)[-1] if prefer == "latest" else np.flatnonzero(close)[0])
    return {
        "time": float(samples[idx, 0]),
        "well_id": labels[int(nearest[idx])],
        "well_x": float(well_xy[int(nearest[idx]), 0]),
        "well_y": float(well_xy[int(nearest[idx]), 1]),
        "distance_cm": float(nearest_distance[idx]),
    }


def _group_key_dict(columns: Sequence[str], key: object) -> dict[str, object]:
    if len(columns) == 1:
        return {str(columns[0]): key}
    if not isinstance(key, tuple):
        key = (key,)
    return {str(column): value for column, value in zip(columns, key, strict=True)}


def _as_int(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")

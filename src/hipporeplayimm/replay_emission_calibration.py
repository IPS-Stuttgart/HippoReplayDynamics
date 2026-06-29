"""Replay-specific emission calibration helpers.

These utilities keep the standard run-period place-field encoder intact but add
an optional replay-rate gain layer.  They are intentionally lightweight: gains
are estimated from selected replay events under a uniform spatial prior and can
then be applied to an :class:`~hipporeplayimm.encoding.EncodingModel` before
building replay emissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .data import ReplaySession
from .encoding import EmissionConfig, EncodingModel, build_emissions


@dataclass(frozen=True)
class ReplayEmissionCalibration:
    """Per-cell replay gain calibration relative to run-period place fields."""

    cell_ids: np.ndarray
    gains: np.ndarray
    observed_spikes: np.ndarray
    expected_spikes: np.ndarray
    prior_count: float
    prior_gain: float
    event_count: int

    def as_metadata(self, prefix: str = "replay_emission") -> dict[str, object]:
        return {
            f"{prefix}_calibrated": True,
            f"{prefix}_event_count": int(self.event_count),
            f"{prefix}_prior_count": float(self.prior_count),
            f"{prefix}_prior_gain": float(self.prior_gain),
            f"{prefix}_median_gain": float(np.median(self.gains)) if self.gains.size else np.nan,
            f"{prefix}_min_gain": float(np.min(self.gains)) if self.gains.size else np.nan,
            f"{prefix}_max_gain": float(np.max(self.gains)) if self.gains.size else np.nan,
        }


def _finite_float(name: str, value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _event_index_int(event_index: object) -> int:
    if isinstance(event_index, (bool, np.bool_)):
        raise TypeError("event index must be an integer, not boolean")
    return int(event_index)


def fit_replay_cell_gains(
    session: ReplaySession,
    encoding: EncodingModel,
    event_indices: Sequence[int],
    config: EmissionConfig | None = None,
    *,
    prior_count: float = 5.0,
    prior_gain: float = 1.0,
    min_gain: float = 0.05,
    max_gain: float = 20.0,
) -> ReplayEmissionCalibration:
    """Estimate per-cell replay gains from selected events.

    The expected spike count uses the run-period rate map averaged over spatial
    bins, multiplied by ripple duration.  This deliberately avoids using decoded
    trajectories from any candidate replay model while still correcting gross
    replay-vs-run firing-rate mismatch.
    """

    config = EmissionConfig() if config is None else config
    prior_count = _finite_float("prior_count", prior_count)
    prior_gain = _finite_float("prior_gain", prior_gain)
    min_gain = _finite_float("min_gain", min_gain)
    max_gain = _finite_float("max_gain", max_gain)
    if prior_count < 0.0:
        raise ValueError("prior_count must be non-negative")
    if prior_gain <= 0.0:
        raise ValueError("prior_gain must be positive")
    if min_gain <= 0.0 or max_gain <= 0.0 or max_gain < min_gain:
        raise ValueError("gain bounds must be positive and ordered")

    observed = np.zeros(encoding.n_cells, dtype=float)
    expected = np.zeros(encoding.n_cells, dtype=float)
    event_count = 0
    mean_rates = np.mean(encoding.rates_hz, axis=1) if encoding.n_cells else np.empty(0)
    for event_index in event_indices:
        event_index_int = _event_index_int(event_index)
        event = session.ripple(event_index_int)
        duration = max(float(event.end - event.start), np.finfo(float).eps)
        emissions = build_emissions(session, encoding, event_index_int, config)
        if emissions.n_time == 0:
            continue
        observed += emissions.spike_counts.sum(axis=0).astype(float)
        expected += mean_rates * duration * float(config.spike_rate_scale)
        event_count += 1

    if event_count == 0:
        gains = np.full(encoding.n_cells, np.clip(prior_gain, min_gain, max_gain), dtype=float)
    else:
        numerator = observed + prior_count * prior_gain
        denominator = np.maximum(expected + prior_count, np.finfo(float).tiny)
        gains = np.clip(numerator / denominator, min_gain, max_gain)
    return ReplayEmissionCalibration(
        cell_ids=np.asarray(encoding.cell_ids, dtype=int).copy(),
        gains=np.asarray(gains, dtype=float),
        observed_spikes=observed,
        expected_spikes=expected,
        prior_count=float(prior_count),
        prior_gain=float(prior_gain),
        event_count=int(event_count),
    )


def apply_replay_cell_gains(
    encoding: EncodingModel,
    gains: ReplayEmissionCalibration | Mapping[int, float] | np.ndarray,
) -> EncodingModel:
    """Return a copy of ``encoding`` with rates multiplied by replay gains."""

    gain_vector = _gain_vector_for_encoding(encoding, gains)
    return EncodingModel(
        x_edges=encoding.x_edges.copy(),
        y_edges=encoding.y_edges.copy(),
        bin_centers=encoding.bin_centers.copy(),
        rates_hz=encoding.rates_hz * gain_vector[:, None],
        occupancy_s=encoding.occupancy_s.copy(),
        cell_ids=encoding.cell_ids.copy(),
        config=encoding.config,
    )


def _gain_vector_for_encoding(
    encoding: EncodingModel,
    gains: ReplayEmissionCalibration | Mapping[int, float] | np.ndarray,
) -> np.ndarray:
    if isinstance(gains, ReplayEmissionCalibration):
        mapping = {int(cell): float(gain) for cell, gain in zip(gains.cell_ids, gains.gains, strict=True)}
        gain_vector = np.asarray([mapping.get(int(cell), 1.0) for cell in encoding.cell_ids], dtype=float)
    elif isinstance(gains, Mapping):
        gain_vector = np.asarray([float(gains.get(int(cell), 1.0)) for cell in encoding.cell_ids], dtype=float)
    else:
        gain_vector = np.asarray(gains, dtype=float)
    return _validated_gain_vector(encoding, gain_vector)


def _validated_gain_vector(encoding: EncodingModel, gain_vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(gain_vector, dtype=float)
    if arr.shape != (encoding.n_cells,):
        raise ValueError(f"gain array must have shape {(encoding.n_cells,)}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("replay gains must be finite")
    if np.any(arr <= 0.0):
        raise ValueError("replay gains must be positive")
    return arr

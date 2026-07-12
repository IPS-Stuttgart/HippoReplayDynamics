"""Replay-specific emission calibration helpers.

These utilities keep the standard run-period place-field encoder intact but add
an optional replay-rate gain layer.  They are intentionally lightweight: gains
are estimated from selected replay events under a uniform spatial prior and can
then be applied to an :class:`~hipporeplayimm.encoding.EncodingModel` before
building replay emissions.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
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


def _is_boolean_scalar(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if arr.ndim != 0:
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        try:
            return isinstance(arr.item(), (bool, np.bool_))
        except ValueError:
            return False
    return False


def _array_contains_boolean(value: object) -> bool:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(arr.dtype, np.bool_):
        return True
    if arr.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in arr.flat)
    return False


def _array_contains_text(value: object) -> bool:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.issubdtype(arr.dtype, np.str_) or np.issubdtype(arr.dtype, np.bytes_):
        return True
    if arr.dtype == object:
        return any(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in arr.flat)
    return False


def _finite_float(name: str, value: object) -> float:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if arr.ndim != 0:
        raise TypeError(f"{name} must be a finite numeric scalar")
    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must be a numeric scalar, not boolean")
    try:
        out = float(arr)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _finite_gain_scalar(value: object) -> float:
    if _is_boolean_scalar(value):
        raise TypeError("replay gains must be numeric, not boolean")
    if _array_contains_text(value):
        raise TypeError("replay gains must be numeric, not text")
    try:
        out = float(np.asarray(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("replay gains must be finite") from exc
    if not np.isfinite(out):
        raise ValueError("replay gains must be finite")
    return out


def _event_index_int(event_index: object) -> int:
    if isinstance(event_index, (bool, np.bool_)):
        raise TypeError("event index must be an integer, not boolean")
    try:
        return int(operator.index(event_index))
    except TypeError as exc:
        raise TypeError("event index must be an integer") from exc


def _integral_cell_id(value: object, name: str) -> int:
    """Coerce one cell ID without truncating boolean or fractional values."""

    if _is_boolean_scalar(value):
        raise TypeError(f"{name} must contain integer identifiers, not boolean values")
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite integer identifiers") from exc
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be one-dimensional")
    item = scalar.item()
    if isinstance(item, (bool, np.bool_)):
        raise TypeError(f"{name} must contain integer identifiers, not boolean values")
    if isinstance(item, (str, bytes, np.str_, np.bytes_)):
        if isinstance(item, (bytes, np.bytes_)):
            try:
                text = bytes(item).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{name} must contain finite integer identifiers") from exc
        else:
            text = str(item)
        try:
            return int(text.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"{name} must contain finite integer identifiers") from exc
    if isinstance(item, (complex, np.complexfloating)):
        raise TypeError(f"{name} must contain real integer identifiers")
    try:
        return int(operator.index(item))
    except TypeError:
        pass
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite integer identifiers") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must contain finite integer identifiers")
    return int(numeric)


def _coerce_integral_cell_ids(
    values: object,
    name: str,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be one-dimensional") from exc
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_size is not None and raw.shape != (int(expected_size),):
        raise ValueError(f"{name} must contain one ID per encoding cell")
    identifiers = [_integral_cell_id(value, name) for value in raw]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must contain unique identifiers")
    integer_info = np.iinfo(np.dtype(int))
    if any(identifier < int(integer_info.min) or identifier > int(integer_info.max) for identifier in identifiers):
        raise ValueError(f"{name} must fit into integer identifier range")
    return np.asarray(identifiers, dtype=int)


def _gain_mapping_from_mapping(gains: Mapping[object, object]) -> dict[int, float]:
    mapping: dict[int, float] = {}
    for raw_cell_id, raw_gain in gains.items():
        cell_id = _integral_cell_id(raw_cell_id, "replay gain mapping cell IDs")
        if cell_id in mapping:
            raise ValueError("replay gain mapping cell IDs must contain unique identifiers")
        integer_info = np.iinfo(np.dtype(int))
        if cell_id < int(integer_info.min) or cell_id > int(integer_info.max):
            raise ValueError("replay gain mapping cell IDs must fit into integer identifier range")
        mapping[cell_id] = _finite_gain_scalar(raw_gain)
    return mapping


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
    cell_ids = _coerce_integral_cell_ids(
        encoding.cell_ids,
        "encoding.cell_ids",
        expected_size=encoding.n_cells,
    )

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
        cell_ids=cell_ids.copy(),
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
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_rates_hz = encoding.rates_hz * gain_vector[:, None]
    if not np.all(np.isfinite(scaled_rates_hz)):
        raise ValueError("replay gain scaling must produce finite rates")
    return EncodingModel(
        x_edges=encoding.x_edges.copy(),
        y_edges=encoding.y_edges.copy(),
        bin_centers=encoding.bin_centers.copy(),
        rates_hz=scaled_rates_hz,
        occupancy_s=encoding.occupancy_s.copy(),
        cell_ids=encoding.cell_ids.copy(),
        config=encoding.config,
    )


def _gain_vector_for_encoding(
    encoding: EncodingModel,
    gains: ReplayEmissionCalibration | Mapping[int, float] | np.ndarray,
) -> np.ndarray:
    encoding_cell_ids = _coerce_integral_cell_ids(
        encoding.cell_ids,
        "encoding.cell_ids",
        expected_size=encoding.n_cells,
    )
    if isinstance(gains, ReplayEmissionCalibration):
        calibration_cell_ids = _coerce_integral_cell_ids(
            gains.cell_ids,
            "calibration cell IDs",
        )
        calibration_gains = np.asarray(gains.gains, dtype=object)
        if calibration_gains.ndim != 1 or calibration_gains.shape != calibration_cell_ids.shape:
            raise ValueError("calibration gains must contain one value per calibration cell ID")
        mapping = {
            int(cell_id): _finite_gain_scalar(gain)
            for cell_id, gain in zip(calibration_cell_ids, calibration_gains, strict=True)
        }
        gain_vector = np.asarray(
            [mapping.get(int(cell_id), 1.0) for cell_id in encoding_cell_ids],
            dtype=float,
        )
    elif isinstance(gains, Mapping):
        mapping = _gain_mapping_from_mapping(gains)
        gain_vector = np.asarray(
            [mapping.get(int(cell_id), 1.0) for cell_id in encoding_cell_ids],
            dtype=float,
        )
    else:
        gain_vector = np.asarray(gains)
    return _validated_gain_vector(encoding, gain_vector)


def _validated_gain_vector(encoding: EncodingModel, gain_vector: np.ndarray) -> np.ndarray:
    if _array_contains_boolean(gain_vector):
        raise TypeError("replay gains must be numeric, not boolean")
    if _array_contains_text(gain_vector):
        raise TypeError("replay gains must be numeric, not text")
    arr = np.asarray(gain_vector, dtype=float)
    if arr.shape != (encoding.n_cells,):
        raise ValueError(f"gain array must have shape {(encoding.n_cells,)}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("replay gains must be finite")
    if np.any(arr <= 0.0):
        raise ValueError("replay gains must be positive")
    return arr

"""Spatial and cell-identity shuffle controls for replay evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .data import ReplaySession
from .encoding import EmissionConfig, EncodingModel, build_emissions
from .state_space_model import StateSpaceReplayModel

SUPPORTED_SHUFFLE_MODES = (
    "cell-permutation",
    "spatial-roll",
    "spatial-permutation",
    "independent-spatial-permutation",
)
SHUFFLE_CONTROL_SCORE_COLUMNS = (
    "session",
    "event_index",
    "requested_model",
    "model",
    "control_type",
    "control_index",
    "log_evidence",
    "n_time",
    "n_spikes",
)


@dataclass(frozen=True)
class ShuffleControlConfig:
    mode: str = "spatial-roll"
    n_shuffles: int = 100
    random_seed: int = 1


def shuffled_encoding(
    encoding: EncodingModel,
    *,
    mode: str = "spatial-roll",
    random_seed: int = 1,
) -> EncodingModel:
    """Return a rate-map control encoding.

    Supported modes are ``cell-permutation``, ``spatial-roll``,
    ``spatial-permutation``, and ``independent-spatial-permutation``.
    """

    mode = _validated_shuffle_mode(mode)
    random_seed = _nonnegative_integer_value("random_seed", random_seed)
    rng = np.random.default_rng(random_seed)
    rates = np.asarray(encoding.rates_hz, dtype=float).copy()
    if mode == "cell-permutation":
        if rates.shape[0] > 1:
            rates = rates[rng.permutation(rates.shape[0])]
    elif mode == "spatial-roll":
        rates = _spatial_roll_rates(rates, encoding.grid_shape, rng)
    elif mode == "spatial-permutation":
        permutation = rng.permutation(encoding.n_bins)
        rates = rates[:, permutation]
    elif mode == "independent-spatial-permutation":
        if rates.shape[0] > 0:
            rates = np.vstack([row[rng.permutation(encoding.n_bins)] for row in rates])
    else:  # pragma: no cover - guarded by _validated_shuffle_mode.
        raise AssertionError(f"Unhandled shuffle mode: {mode!r}")
    return EncodingModel(
        x_edges=encoding.x_edges.copy(),
        y_edges=encoding.y_edges.copy(),
        bin_centers=encoding.bin_centers.copy(),
        rates_hz=rates,
        occupancy_s=encoding.occupancy_s.copy(),
        cell_ids=encoding.cell_ids.copy(),
        config=encoding.config,
    )


def _validated_shuffle_mode(mode: object) -> str:
    if isinstance(mode, str) and mode in SUPPORTED_SHUFFLE_MODES:
        return mode
    supported = ", ".join(SUPPORTED_SHUFFLE_MODES)
    raise ValueError(f"mode must be one of: {supported}")


def _nonnegative_integer_value(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be a finite integer")
    integer = int(round(numeric))
    if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
        raise ValueError(f"{name} must be an integer")
    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _spatial_roll_rates(rates: np.ndarray, grid_shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    if rates.size == 0:
        return rates.copy()
    n_x, n_y = _validate_grid_shape(grid_shape)
    if rates.shape[1] != n_x * n_y:
        raise ValueError("rates must contain one column per spatial grid bin")
    out = np.empty_like(rates)
    for cell_index, row in enumerate(rates):
        grid = row.reshape((n_x, n_y))
        dx, dy = _nonidentity_roll_shift((n_x, n_y), rng)
        out[cell_index] = np.roll(np.roll(grid, dx, axis=0), dy, axis=1).reshape(-1)
    return out


def _validate_grid_shape(grid_shape: tuple[int, int]) -> tuple[int, int]:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly two dimensions")
    n_x, n_y = (int(grid_shape[0]), int(grid_shape[1]))
    if n_x <= 0 or n_y <= 0:
        raise ValueError("grid_shape dimensions must be positive")
    return n_x, n_y


def _nonidentity_roll_shift(grid_shape: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    """Draw a spatial-roll shift that is not the identity when possible."""

    n_x, n_y = grid_shape
    n_bins = n_x * n_y
    if n_bins <= 1:
        return 0, 0
    flat_shift = int(rng.integers(1, n_bins))
    return divmod(flat_shift, n_y)


def score_shuffle_controls(
    session: ReplaySession,
    encoding: EncodingModel,
    event_indices: Iterable[int],
    models: Mapping[str, object],
    emission_config: EmissionConfig | None = None,
    control_config: ShuffleControlConfig | None = None,
) -> pd.DataFrame:
    """Score models on shuffled encodings and return event/model/control rows."""

    emission_config = EmissionConfig() if emission_config is None else emission_config
    control_config = ShuffleControlConfig() if control_config is None else control_config
    mode = _validated_shuffle_mode(control_config.mode)
    n_shuffles = _nonnegative_integer_value("n_shuffles", control_config.n_shuffles)
    random_seed = _nonnegative_integer_value("random_seed", control_config.random_seed)
    event_indices = tuple(int(event_index) for event_index in event_indices)
    rows: list[dict[str, object]] = []
    for shuffle_index in range(n_shuffles):
        control_encoding = shuffled_encoding(
            encoding,
            mode=mode,
            random_seed=random_seed + shuffle_index,
        )
        for event_index in event_indices:
            emissions = build_emissions(session, control_encoding, event_index, emission_config)
            for requested_model, model in models.items():
                if isinstance(model, StateSpaceReplayModel):
                    score = model.score(
                        emissions,
                        control_encoding.bin_centers,
                        occupancy_s=control_encoding.occupancy_s,
                    )
                else:
                    score = model.score(emissions, control_encoding.bin_centers)
                rows.append(
                    {
                        "session": session.session_id,
                        "event_index": event_index,
                        "requested_model": requested_model,
                        "model": score.model_name,
                        "control_type": mode,
                        "control_index": int(shuffle_index),
                        "log_evidence": float(score.log_likelihood),
                        "n_time": int(score.n_time),
                        "n_spikes": int(score.n_spikes),
                    }
                )
    return pd.DataFrame(rows, columns=SHUFFLE_CONTROL_SCORE_COLUMNS)


def _with_finite_log_evidence(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    if "log_evidence" not in out.columns:
        return out
    values = pd.to_numeric(out["log_evidence"], errors="coerce")
    out["log_evidence"] = values.where(np.isfinite(values), np.nan)
    return out


def add_shuffle_p_values(real_scores: pd.DataFrame, control_scores: pd.DataFrame) -> pd.DataFrame:
    """Add empirical upper-tail p-values against shuffled control evidence."""

    real_scores = _with_finite_log_evidence(real_scores)
    control_scores = _with_finite_log_evidence(control_scores)
    if real_scores.empty or control_scores.empty:
        out = real_scores.copy()
        out["shuffle_p_value"] = np.nan
        out["shuffle_log_evidence_median"] = np.nan
        out["shuffle_log_evidence_mean"] = np.nan
        out["shuffle_log_evidence_std"] = np.nan
        out["shuffle_count"] = np.nan
        return out
    grouped = control_scores.groupby(["session", "event_index", "model"])
    summaries = grouped["log_evidence"].agg(
        shuffle_log_evidence_median="median",
        shuffle_log_evidence_mean="mean",
        shuffle_log_evidence_std="std",
        shuffle_count="count",
    )
    rows = []
    for _, row in real_scores.iterrows():
        key = (row.get("session"), row.get("event_index"), row.get("model"))
        control = grouped.get_group(key)["log_evidence"].to_numpy(float) if key in grouped.groups else np.array([])
        control = control[np.isfinite(control)]
        real_log_evidence = row.get("log_evidence")
        p_value = np.nan
        if control.size and np.isfinite(real_log_evidence):
            p_value = float((1.0 + np.sum(control >= float(real_log_evidence))) / (control.size + 1.0))
        rows.append(p_value)
    out = real_scores.copy()
    out["shuffle_p_value"] = rows
    return out.merge(summaries.reset_index(), on=["session", "event_index", "model"], how="left")

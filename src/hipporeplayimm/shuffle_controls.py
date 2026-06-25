"""Spatial and cell-identity shuffle controls for replay evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .data import ReplaySession
from .encoding import EmissionConfig, EncodingModel, build_emissions
from .state_space_model import StateSpaceReplayModel


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
    else:
        raise ValueError(
            "mode must be one of: cell-permutation, spatial-roll, "
            "spatial-permutation, independent-spatial-permutation"
        )
    return EncodingModel(
        x_edges=encoding.x_edges.copy(),
        y_edges=encoding.y_edges.copy(),
        bin_centers=encoding.bin_centers.copy(),
        rates_hz=rates,
        occupancy_s=encoding.occupancy_s.copy(),
        cell_ids=encoding.cell_ids.copy(),
        config=encoding.config,
    )


def _spatial_roll_rates(rates: np.ndarray, grid_shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    if rates.size == 0:
        return rates.copy()
    out = np.empty_like(rates)
    for cell_index, row in enumerate(rates):
        grid = row.reshape(grid_shape)
        dx = int(rng.integers(0, max(grid_shape[0], 1)))
        dy = int(rng.integers(0, max(grid_shape[1], 1)))
        out[cell_index] = np.roll(np.roll(grid, dx, axis=0), dy, axis=1).reshape(-1)
    return out


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
    event_indices = tuple(int(event_index) for event_index in event_indices)
    rows: list[dict[str, object]] = []
    for shuffle_index in range(int(control_config.n_shuffles)):
        control_encoding = shuffled_encoding(
            encoding,
            mode=control_config.mode,
            random_seed=int(control_config.random_seed) + shuffle_index,
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
                        "control_type": control_config.mode,
                        "control_index": int(shuffle_index),
                        "log_evidence": float(score.log_likelihood),
                        "n_time": int(score.n_time),
                        "n_spikes": int(score.n_spikes),
                    }
                )
    return pd.DataFrame(rows)


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

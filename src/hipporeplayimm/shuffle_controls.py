"""Spatial and cell-identity shuffle controls for replay evidence."""

from __future__ import annotations

from dataclasses import dataclass
import operator
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
_SHUFFLE_P_VALUE_BASE_COLUMNS = ("session", "event_index", "model")
_SHUFFLE_P_VALUE_SCOPE_COLUMNS = (
    "requested_model",
    "benchmark_random_seed",
    "benchmark_cell_split_index",
    "benchmark_cell_split_seed",
    "benchmark_event_subset_seed",
    "benchmark_test_cell_fraction",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
    "window_role",
    "window_index",
    "null_index",
    "matched_null_rank",
    "template_event_index",
    "event_window_variant",
    "window_variant",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "simulation_random_seed",
)
_SHUFFLE_SCOPE_KEY_COLUMN = "__shuffle_scope_key"


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
    try:
        array = np.asarray(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer scalar") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be an integer scalar")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not boolean")

    try:
        integer = operator.index(scalar)
    except TypeError:
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be a finite integer")
        integer = int(round(numeric))
        if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
            raise ValueError(f"{name} must be an integer")

    if integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(integer)


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
    try:
        values = tuple(grid_shape)
    except TypeError as exc:
        raise ValueError("grid_shape must contain exactly two dimensions") from exc
    if len(values) != 2:
        raise ValueError("grid_shape must contain exactly two dimensions")
    n_x, n_y = (_positive_grid_dimension(value) for value in values)
    return n_x, n_y


def _positive_grid_dimension(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("grid_shape dimensions must be positive integers")
    try:
        integer = operator.index(value)
    except TypeError:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("grid_shape dimensions must be positive integers") from exc
        if not np.isfinite(numeric):
            raise ValueError("grid_shape dimensions must be finite positive integers")
        integer = int(round(numeric))
        if not np.isclose(numeric, integer, rtol=0.0, atol=0.0):
            raise ValueError("grid_shape dimensions must be integer-valued")
    if integer <= 0:
        raise ValueError("grid_shape dimensions must be positive")
    return int(integer)


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
    event_indices = tuple(_nonnegative_integer_value("event_indices", event_index) for event_index in event_indices)
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
    """Add empirical upper-tail p-values against shuffled control evidence.

    Independent decode scopes that reuse the same ``session``/``event_index`` and
    ``model`` keys, such as window variants or cell-split repeats, keep separate
    control distributions when matching scope columns are present in both tables.
    """

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

    group_columns = _shuffle_p_value_group_columns(real_scores, control_scores)
    control_scores = control_scores.copy()
    control_scores[_SHUFFLE_SCOPE_KEY_COLUMN] = _scope_keys(control_scores, group_columns)
    grouped = control_scores.groupby(_SHUFFLE_SCOPE_KEY_COLUMN, sort=False, dropna=False)
    control_by_key = {
        key: group["log_evidence"].to_numpy(dtype=float)
        for key, group in grouped
    }
    summaries = grouped["log_evidence"].agg(
        shuffle_log_evidence_median="median",
        shuffle_log_evidence_mean="mean",
        shuffle_log_evidence_std="std",
        shuffle_count="count",
    )

    p_values = []
    real_keys = _scope_keys(real_scores, group_columns)
    for key, (_, row) in zip(real_keys.to_numpy(dtype=object), real_scores.iterrows()):
        control = control_by_key.get(key, np.array([], dtype=float))
        control = control[np.isfinite(control)]
        real_log_evidence = row.get("log_evidence")
        p_value = np.nan
        if control.size and np.isfinite(real_log_evidence):
            p_value = float((1.0 + np.sum(control >= float(real_log_evidence))) / (control.size + 1.0))
        p_values.append(p_value)

    out = real_scores.copy()
    out[_SHUFFLE_SCOPE_KEY_COLUMN] = real_keys
    out["shuffle_p_value"] = p_values
    out = out.merge(summaries.reset_index(), on=_SHUFFLE_SCOPE_KEY_COLUMN, how="left")
    return out.drop(columns=[_SHUFFLE_SCOPE_KEY_COLUMN])


def _shuffle_p_value_group_columns(real_scores: pd.DataFrame, control_scores: pd.DataFrame) -> list[str]:
    missing = [
        column
        for column in _SHUFFLE_P_VALUE_BASE_COLUMNS
        if column not in real_scores.columns or column not in control_scores.columns
    ]
    if missing:
        raise KeyError(f"shuffle score tables missing required columns: {missing}")
    columns = list(_SHUFFLE_P_VALUE_BASE_COLUMNS)
    for column in _SHUFFLE_P_VALUE_SCOPE_COLUMNS:
        if column in real_scores.columns and column in control_scores.columns:
            columns.append(column)
    return columns


def _scope_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.Series(
        [
            tuple(_scope_label(row[column]) for column in columns)
            for _, row in frame.loc[:, columns].iterrows()
        ],
        index=frame.index,
        dtype=object,
    )


def _scope_label(value: object) -> str:
    if _is_missing_scalar(value):
        return "<missing>"
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            return repr(("array", np.asarray(value, dtype=object).reshape(-1).tolist()))
    if isinstance(value, (list, tuple)):
        return repr(("sequence", list(value)))
    if isinstance(value, set):
        return repr(("set", sorted(value, key=repr)))
    numeric = _numeric_scope_label(value)
    if numeric is not None:
        return repr(("numeric", numeric))
    return repr(("scalar", str(value).strip()))


def _numeric_scope_label(value: object) -> str | None:
    """Return a canonical label for numeric scalar scope keys.

    CSV round-trips can turn integer identifiers such as ``event_index`` or
    ``window_index`` into floats in one table but not the other.  Treat numeric
    scalars with the same exact value as the same shuffle-control scope, while
    keeping booleans and textual labels on the normal string path.
    """

    if isinstance(value, (bool, np.bool_)):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    rounded = int(round(numeric))
    if np.isclose(numeric, rounded, rtol=0.0, atol=0.0):
        return str(rounded)
    return repr(numeric)


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)

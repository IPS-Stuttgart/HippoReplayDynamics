"""Score-table metadata helpers shared by model-evidence and ground-truth code."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .encoding import EmissionConfig, EncodingConfig


def encoding_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: EncodingConfig,
) -> EncodingConfig:
    """Build an encoding config from canonical or legacy score metadata.

    Held-out benchmark outputs use canonical ``encoding_*`` columns, whereas
    older model-evidence outputs used shorter names such as ``bin_size_cm``.
    Accept both, but reject conflicting values when both are present.
    """

    return EncodingConfig(
        bin_size_cm=_unique_float_from_columns(
            scores_frame,
            ("encoding_bin_size_cm", "bin_size_cm"),
            fallback.bin_size_cm,
        ),
        smoothing_sigma_bins=_unique_float_from_columns(
            scores_frame,
            ("encoding_smoothing_sigma_bins", "smoothing_sigma_bins"),
            fallback.smoothing_sigma_bins,
        ),
        min_speed_cm_s=_unique_float_from_columns(
            scores_frame,
            ("encoding_min_speed_cm_s", "min_speed_cm_s"),
            fallback.min_speed_cm_s,
        ),
        min_occupancy_s=_unique_float_from_columns(
            scores_frame,
            ("encoding_min_occupancy_s",),
            fallback.min_occupancy_s,
        ),
        rate_floor_hz=_unique_float_from_columns(
            scores_frame,
            ("encoding_rate_floor_hz",),
            fallback.rate_floor_hz,
        ),
        arena_padding_cm=_unique_float_from_columns(
            scores_frame,
            ("encoding_arena_padding_cm",),
            fallback.arena_padding_cm,
        ),
        use_excitatory=_unique_bool_from_column(
            scores_frame,
            "encoding_use_excitatory",
            fallback.use_excitatory,
        ),
    )


def emission_config_for_scores(
    scores_frame: pd.DataFrame,
    fallback: EmissionConfig,
) -> EmissionConfig:
    """Build an emission config from canonical or legacy score metadata."""

    return EmissionConfig(
        time_bin_s=_unique_float_from_columns(
            scores_frame,
            ("emission_time_bin_s", "time_bin_s"),
            fallback.time_bin_s,
        )
    )


def _unique_float_from_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    default: float,
) -> float:
    values: list[float] = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(float(value))
    if not values:
        return float(default)
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)


def _unique_bool_from_column(frame: pd.DataFrame, column: str, default: bool) -> bool:
    values: list[bool] = []
    if column in frame.columns:
        for value in frame[column].dropna():
            text = str(value).strip()
            if text:
                values.append(_parse_bool(value))
    if not values:
        return bool(default)
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"{column} contains multiple values")
    return bool(first)


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")

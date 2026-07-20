"""Spike-rate-scale metadata compatibility patches.

The model-evidence scripts can run with a non-default ``EmissionConfig.spike_rate_scale``.
Post-hoc ground-truth decoding must reconstruct the same emission model from the
saved score table; otherwise decoded endpoints and well-posterior metrics are
computed under the wrong Poisson rate scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .encoding import EmissionConfig

_MISSING_METADATA_STRINGS = {"", "nan", "na", "n/a", "none", "null", "<na>"}
_EMISSION_CONFIG_MARKER = "_spike_rate_metadata_emission_config_wrapper"
_BENCHMARK_METADATA_MARKER = "_spike_rate_metadata_wrapped"
_SPIKE_RATE_SCALE_COLUMNS = (
    "emission_spike_rate_scale",
    "spike_rate_scale",
)


def apply_spike_rate_metadata_patch() -> None:
    """Preserve ``spike_rate_scale`` in score-table readers and writers."""

    from . import benchmarks as bench
    from . import ground_truth as gt
    from . import pyrecest_numeric_metadata_guard as pyrecest_numeric_guard
    from . import score_metadata as score_meta

    pyrecest_numeric_guard.apply_pyrecest_numeric_metadata_guard_patch()

    current_reader = getattr(score_meta, "emission_config_for_scores", None)
    if getattr(current_reader, _EMISSION_CONFIG_MARKER, False):
        emission_config_for_scores = current_reader
    else:

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
                ),
                spike_rate_scale=_unique_float_from_columns(
                    scores_frame,
                    _SPIKE_RATE_SCALE_COLUMNS,
                    fallback.spike_rate_scale,
                ),
                likelihood_temperature=_unique_float_from_columns(
                    scores_frame,
                    (
                        "emission_likelihood_temperature",
                        "likelihood_temperature",
                    ),
                    fallback.likelihood_temperature,
                ),
                negative_binomial_overdispersion=_unique_float_from_columns(
                    scores_frame,
                    (
                        "emission_negative_binomial_overdispersion",
                        "negative_binomial_overdispersion",
                    ),
                    fallback.negative_binomial_overdispersion,
                ),
            )

        setattr(emission_config_for_scores, _EMISSION_CONFIG_MARKER, True)
        score_meta.emission_config_for_scores = emission_config_for_scores

    gt._emission_config_for_scores = emission_config_for_scores

    base_metadata = bench._benchmark_config_metadata
    if not getattr(base_metadata, _BENCHMARK_METADATA_MARKER, False):

        def benchmark_config_metadata(config) -> dict[str, object]:
            metadata = dict(base_metadata(config))
            metadata["emission_spike_rate_scale"] = float(
                config.emissions.spike_rate_scale
            )
            metadata["emission_likelihood_temperature"] = float(
                config.emissions.likelihood_temperature
            )
            metadata["emission_negative_binomial_overdispersion"] = float(
                config.emissions.negative_binomial_overdispersion
            )
            return metadata

        setattr(benchmark_config_metadata, _BENCHMARK_METADATA_MARKER, True)
        bench._benchmark_config_metadata = benchmark_config_metadata

    score_meta._spike_rate_metadata_patch_applied = True


def _finite_numeric_metadata_value(value: object, name: str) -> float:
    """Return one finite numeric metadata scalar without accepting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must contain finite numeric metadata values")
    try:
        scalar = np.asarray(value)
    except ValueError as exc:
        raise ValueError(f"{name} must contain scalar numeric metadata values") from exc
    if scalar.ndim != 0:
        raise ValueError(f"{name} must contain scalar numeric metadata values")
    item = scalar.item()
    if isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must contain finite numeric metadata values")
    try:
        numeric = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric metadata values") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


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
            scalar = np.asarray(value)
            if scalar.ndim != 0:
                raise ValueError(
                    f"{column} must contain scalar numeric metadata values"
                )
            item = scalar.item()
            text = str(item).strip()
            if text.lower() in _MISSING_METADATA_STRINGS:
                continue
            if text:
                values.append(_finite_numeric_metadata_value(item, column))
    if not values:
        return _finite_numeric_metadata_value(default, "metadata fallback")
    first = values[0]
    if columns == _SPIKE_RATE_SCALE_COLUMNS:
        conflicting = any(
            not np.isclose(value, first, rtol=1e-5, atol=1e-8)
            for value in values[1:]
        )
    else:
        conflicting = any(value != first for value in values[1:])
    if conflicting:
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)

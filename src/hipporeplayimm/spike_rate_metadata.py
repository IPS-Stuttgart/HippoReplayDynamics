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


def apply_spike_rate_metadata_patch() -> None:
    """Preserve ``spike_rate_scale`` in score-table readers and writers."""

    from . import benchmarks as bench
    from . import ground_truth as gt
    from . import score_metadata as score_meta

    if getattr(score_meta, "_spike_rate_metadata_patch_applied", False):
        return

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
                ("emission_spike_rate_scale", "spike_rate_scale"),
                fallback.spike_rate_scale,
            ),
            likelihood_temperature=_unique_float_from_columns(
                scores_frame,
                ("emission_likelihood_temperature", "likelihood_temperature"),
                fallback.likelihood_temperature,
            ),
            negative_binomial_overdispersion=_unique_float_from_columns(
                scores_frame,
                ("emission_negative_binomial_overdispersion", "negative_binomial_overdispersion"),
                fallback.negative_binomial_overdispersion,
            ),
        )

    score_meta.emission_config_for_scores = emission_config_for_scores
    gt._emission_config_for_scores = emission_config_for_scores

    base_metadata = bench._benchmark_config_metadata
    if not getattr(base_metadata, "_spike_rate_metadata_wrapped", False):

        def benchmark_config_metadata(config) -> dict[str, object]:
            metadata = dict(base_metadata(config))
            metadata["emission_spike_rate_scale"] = float(config.emissions.spike_rate_scale)
            metadata["emission_likelihood_temperature"] = float(config.emissions.likelihood_temperature)
            metadata["emission_negative_binomial_overdispersion"] = float(config.emissions.negative_binomial_overdispersion)
            return metadata

        benchmark_config_metadata._spike_rate_metadata_wrapped = True
        bench._benchmark_config_metadata = benchmark_config_metadata

    score_meta._spike_rate_metadata_patch_applied = True


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
            if text.lower() in _MISSING_METADATA_STRINGS:
                continue
            if text:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError(f"{column} must be finite")
                values.append(numeric)
    if not values:
        return float(default)
    first = values[0]
    if any(not np.isclose(value, first) for value in values[1:]):
        raise ValueError(f"{' / '.join(columns)} contains multiple values")
    return float(first)

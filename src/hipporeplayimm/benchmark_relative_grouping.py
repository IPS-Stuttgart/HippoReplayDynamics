"""Scope relative benchmark metrics to matching run metadata.

Held-out benchmark score tables are often concatenated across parameter sweeps,
cell-split strategies, or held-out fractions. The relative held-out metrics must
compare each model against the static baselines from the same scoring condition;
otherwise a strong static baseline from one configuration can be subtracted from
a model row from another configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Metadata columns emitted by ``benchmarks._benchmark_config_metadata`` and
# ``benchmarks._benchmark_split_metadata`` that define a compatible held-out
# scoring condition.  Optional columns are used whenever present.  Missing
# values are explicitly scoped by the add-relative-metrics wrapper below so
# legacy rows with absent metadata cannot be mixed with newer sweep rows that
# carry concrete values for the same column.
_BENCHMARK_RELATIVE_SCOPE_COLUMNS = (
    "benchmark_test_cell_fraction",
    "benchmark_n_cell_splits",
    "benchmark_cell_split_count",
    "benchmark_cell_split_seed",
    "benchmark_cell_split_strategy",
    "benchmark_cell_split_strata",
    "benchmark_randomize_event_subset",
    "benchmark_event_subset_seed",
    "benchmark_event_subset_base_seed",
    "encoding_bin_size_cm",
    "encoding_smoothing_sigma_bins",
    "encoding_min_speed_cm_s",
    "encoding_min_occupancy_s",
    "encoding_rate_floor_hz",
    "encoding_arena_padding_cm",
    "encoding_use_excitatory",
    "emission_time_bin_s",
    "emission_spike_rate_scale",
    "emission_likelihood_temperature",
    "emission_negative_binomial_overdispersion",
    "clusterless_mark_smoothing_sigma_bins",
    "clusterless_mark_prior_count",
    "clusterless_mark_variance_floor",
    "clusterless_rate_floor_hz",
    "clusterless_mark_likelihood",
    "clusterless_mark_kde_bandwidth",
    "clusterless_mark_kde_spatial_sigma_bins",
    "clusterless_mark_kde_max_neighbors",
    "clusterless_mark_group_by",
    "state_space_valid_occupancy_threshold_s",
    "state_space_stationary_sigma_cm",
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_max_step_sigma",
    "state_space_imm_mode_stickiness",
    "state_space_imm_switch_tau_s",
    "state_space_trajectory_imm_mode_stickiness",
    "state_space_trajectory_imm_momentum_initial_probability",
    "state_space_trajectory_imm_momentum_switch_probability",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_velocity_decay_tau_s",
    "state_space_momentum_candidate_source",
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_candidate_mass_threshold",
    "state_space_momentum_candidate_min_k",
    "state_space_momentum_candidate_max_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_displacement_radius_bins",
    "state_space_displacement_position_sigma_cm",
    "state_space_displacement_transition_sigma_cm_sqrt_s",
    "state_space_displacement_prior_sigma_cm",
    "goal_state_space_transition_sigma_cm_sqrt_s",
    "goal_state_space_drift_speed_cm_s",
    "goal_state_space_max_step_sigma",
)


def apply_benchmark_relative_grouping_patch() -> None:
    """Patch relative metric grouping to avoid cross-run baseline mixing."""

    from . import benchmarks as bench

    group_columns = bench._benchmark_event_group_columns
    if not getattr(group_columns, "_benchmark_relative_grouping_scoped", False):

        def benchmark_event_group_columns_with_metadata(frame: pd.DataFrame) -> list[str]:
            columns = ["session", "event_index"]
            for column in (
                "benchmark_random_seed",
                "benchmark_cell_split_index",
                *_BENCHMARK_RELATIVE_SCOPE_COLUMNS,
            ):
                if column in columns or not _usable_group_column(frame, column):
                    continue
                columns.append(column)
            return columns

        benchmark_event_group_columns_with_metadata._benchmark_relative_grouping_scoped = True  # type: ignore[attr-defined]
        bench._benchmark_event_group_columns = benchmark_event_group_columns_with_metadata

    add_relative_metrics = bench._add_relative_metrics
    if getattr(add_relative_metrics, "_benchmark_relative_missing_scope_wrapped", False):
        return

    def add_relative_metrics_with_missing_scope(frame: pd.DataFrame) -> pd.DataFrame:
        """Keep missing optional metadata as its own relative-metric scope.

        ``benchmarks._add_relative_metrics`` uses pandas groupby/merge over the
        benchmark event scope.  pandas drops NaN group keys by default, so scope
        columns with mixed missing and concrete values previously had to be
        disabled completely.  That prevented row loss but allowed legacy rows
        with missing metadata to supply static baselines for newer rows carrying
        a concrete value.  Temporarily replacing missing scope values with a
        column-specific sentinel keeps each missing-value group intact without
        leaking that sentinel into the returned score table.
        """

        working = frame.copy()
        group_columns = bench._benchmark_event_group_columns(working)
        missing_sentinels: dict[str, str] = {}
        for column in group_columns:
            if column not in working.columns:
                continue
            missing_mask = working[column].isna()
            if not bool(missing_mask.any()):
                continue
            sentinel = _missing_scope_sentinel(column, working[column])
            working[column] = working[column].astype(object)
            working.loc[missing_mask, column] = sentinel
            missing_sentinels[column] = sentinel

        out = add_relative_metrics(working)
        for column, sentinel in missing_sentinels.items():
            if column not in out.columns:
                continue
            out.loc[out[column].eq(sentinel), column] = np.nan
        return out

    add_relative_metrics_with_missing_scope._benchmark_relative_missing_scope_wrapped = True  # type: ignore[attr-defined]
    bench._add_relative_metrics = add_relative_metrics_with_missing_scope


def _usable_group_column(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns


def _missing_scope_sentinel(column: str, values: pd.Series) -> str:
    base = f"__hipporeplayimm_missing_scope_{column}__"
    sentinel = base
    text_values = values.astype(str)
    suffix = 0
    while bool(text_values.eq(sentinel).any()):
        suffix += 1
        sentinel = f"{base}_{suffix}"
    return sentinel

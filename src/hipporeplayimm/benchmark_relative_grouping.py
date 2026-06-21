"""Scope relative benchmark metrics to matching run metadata.

Held-out benchmark score tables are often concatenated across parameter sweeps,
cell-split strategies, or held-out fractions. The relative held-out metrics must
compare each model against the static baselines from the same scoring condition;
otherwise a strong static baseline from one configuration can be subtracted from
a model row from another configuration.
"""

from __future__ import annotations

import pandas as pd


# Metadata columns emitted by ``benchmarks._benchmark_config_metadata`` and
# ``benchmarks._benchmark_split_metadata`` that define a compatible held-out
# scoring condition. Optional columns are used only when every row has a value;
# this keeps older score tables with missing optional metadata from losing their
# static-baseline groups under pandas' default ``groupby(dropna=True)`` behavior.
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
    if getattr(group_columns, "_benchmark_relative_grouping_scoped", False):
        return

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


def _usable_group_column(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column]
    if values.isna().any():
        return False
    return True

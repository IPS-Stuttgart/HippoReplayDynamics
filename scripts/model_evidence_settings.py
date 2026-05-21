"""Shared validation helpers for model-evidence benchmark outputs."""

from __future__ import annotations

import pandas as pd

_CONSTANT_SETTING_COLUMNS = (
    "bin_size_cm",
    "smoothing_sigma_bins",
    "min_speed_cm_s",
    "time_bin_s",
    "spike_rate_scale",
    "emission_likelihood_temperature",
    "emission_negative_binomial_overdispersion",
    "sorted_spike_emission_model",
    "replay_gain_mode",
    "replay_gain_prior_count",
    "replay_gain_max_gain",
    "negative_binomial_dispersion",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_momentum_candidate_mass_threshold",
    "state_space_momentum_candidate_min_k",
    "state_space_momentum_candidate_max_k",
    "state_space_valid_occupancy_threshold_s",
    "state_space_imm_switch_tau_s",
    "state_space_effective_imm_mode_stickiness",
    "state_space_imm_mode_stickiness_effective",
    "goal_state_space_transition_sigma_cm_sqrt_s",
    "goal_state_space_drift_speed_cm_s",
    "goal_state_space_max_step_sigma",
    "include_clusterless_defaults",
    "valid_state_min_occupancy_s",
    "valid_state_top_occupancy_fraction",
    "valid_state_sigma_cm",
    "valid_state_max_step_sigma",
    "valid_state_grid_diagonal_neighbors",
    "valid_state_grid_stay_probability",
    "window_pre_pads_s",
    "window_post_pads_s",
    "window_min_duration_s",
    "reliability_min_spikes",
    "reliability_min_time_bins",
    "reliability_max_terminal_entropy",
    "reliability_min_candidate_log_mass",
    "clusterless_mark_smoothing_sigma_bins",
    "clusterless_mark_prior_count",
    "clusterless_mark_variance_floor",
    "clusterless_rate_floor_hz",
    "clusterless_mark_likelihood",
    "clusterless_mark_kde_bandwidth",
    "clusterless_mark_kde_spatial_sigma_bins",
    "clusterless_mark_kde_max_neighbors",
    "goal_state_space_transition_sigma_cm_sqrt_s",
    "goal_state_space_drift_speed_cm_s",
    "goal_state_space_max_step_sigma",
    "diagnostic_clusterless_mark_likelihood",
    "diagnostic_clusterless_mark_kde_bandwidth",
    "diagnostic_clusterless_mark_kde_max_neighbors",
)


def _validate_constant_settings(combined: pd.DataFrame) -> None:
    """Reject aggregates that silently mix incompatible benchmark settings."""

    inconsistent: dict[str, list[str]] = {}
    for column in _CONSTANT_SETTING_COLUMNS:
        if column not in combined.columns:
            continue
        values = combined[column].dropna().unique()
        if len(values) > 1:
            inconsistent[column] = sorted(str(value) for value in values)

    if not inconsistent:
        return

    lines = ["Model-evidence shards mix incompatible run settings:"]
    for column, values in sorted(inconsistent.items()):
        lines.append(f"- {column}: {', '.join(values)}")
    raise ValueError("\n".join(lines))

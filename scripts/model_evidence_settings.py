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
    "candidate_top_k",
    "candidate_stationary_sigma_cm",
    "candidate_diffusion_sigma_cm",
    "candidate_momentum_sigma_cm",
    "candidate_velocity_decay",
    "candidate_mode_stickiness",
    "state_space_stationary_sigma_cm",
    "state_space_diffusion_sigma_cm_sqrt_s",
    "state_space_max_step_sigma",
    "state_space_imm_mode_stickiness",
    "state_space_imm_switch_tau_s",
    "state_space_effective_imm_mode_stickiness",
    "state_space_trajectory_imm_mode_stickiness",
    "state_space_trajectory_imm_momentum_initial_probability",
    "state_space_trajectory_imm_momentum_switch_probability",
    "state_space_momentum_sigma_cm_sqrt_s",
    "state_space_momentum_initial_sigma_cm_sqrt_s",
    "state_space_momentum_velocity_decay",
    "state_space_momentum_velocity_decay_tau_s",
    "state_space_momentum_candidate_top_k",
    "state_space_momentum_predicted_candidate_top_k",
    "state_space_momentum_candidate_mass_threshold",
    "state_space_momentum_candidate_min_k",
    "state_space_momentum_candidate_max_k",
    "state_space_momentum_candidate_source",
    "state_space_common_support_top_k",
    "state_space_valid_occupancy_threshold_s",
    "state_space_displacement_radius_bins",
    "state_space_displacement_position_sigma_cm",
    "state_space_displacement_transition_sigma_cm_sqrt_s",
    "state_space_displacement_prior_sigma_cm",
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
    "window_variant_specs",
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

# The session benchmark writes some model settings as top-level columns, while
# the event-scoring script stores the same values in ``diagnostic_*`` columns.
# Treat those schema variants as one setting so shards from different producers
# cannot evade the consistency check.
_CONSTANT_SETTING_ALIASES = {
    "candidate_top_k": ("diagnostic_candidate_top_k",),
    "candidate_stationary_sigma_cm": ("diagnostic_candidate_stationary_sigma_cm",),
    "candidate_diffusion_sigma_cm": ("diagnostic_candidate_diffusion_sigma_cm",),
    "candidate_momentum_sigma_cm": ("diagnostic_candidate_momentum_sigma_cm",),
    "candidate_velocity_decay": ("diagnostic_candidate_velocity_decay",),
    "candidate_mode_stickiness": ("diagnostic_candidate_mode_stickiness",),
    "state_space_stationary_sigma_cm": ("diagnostic_state_space_stationary_sigma_cm",),
    "state_space_diffusion_sigma_cm_sqrt_s": ("diagnostic_state_space_diffusion_sigma_cm_sqrt_s",),
    "state_space_max_step_sigma": ("diagnostic_state_space_max_step_sigma",),
    "state_space_effective_imm_mode_stickiness": (
        "state_space_imm_mode_stickiness_effective",
        "diagnostic_state_space_imm_mode_stickiness",
    ),
    "state_space_imm_switch_tau_s": ("diagnostic_state_space_imm_switch_tau_s",),
    "state_space_momentum_sigma_cm_sqrt_s": ("diagnostic_state_space_momentum_sigma_cm_sqrt_s",),
    "state_space_momentum_initial_sigma_cm_sqrt_s": (
        "diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s",
    ),
    "state_space_momentum_velocity_decay": ("diagnostic_state_space_momentum_velocity_decay",),
    "state_space_momentum_velocity_decay_tau_s": (
        "diagnostic_state_space_momentum_velocity_decay_tau_s",
    ),
    "state_space_momentum_candidate_top_k": (
        "diagnostic_state_space_momentum_candidate_top_k",
        "diagnostic_state_space_imm_candidate_top_k",
    ),
    "state_space_momentum_predicted_candidate_top_k": (
        "diagnostic_state_space_momentum_predicted_candidate_top_k",
        "diagnostic_state_space_imm_predicted_candidate_top_k",
    ),
    "state_space_momentum_candidate_mass_threshold": (
        "diagnostic_state_space_momentum_candidate_mass_threshold",
        "diagnostic_state_space_imm_candidate_mass_threshold",
    ),
    "state_space_momentum_candidate_min_k": (
        "diagnostic_state_space_momentum_candidate_min_k",
        "diagnostic_state_space_imm_candidate_min_k",
    ),
    "state_space_momentum_candidate_max_k": (
        "diagnostic_state_space_momentum_candidate_max_k",
        "diagnostic_state_space_imm_candidate_max_k",
    ),
    "state_space_momentum_candidate_source": (
        "diagnostic_state_space_momentum_candidate_source",
        "diagnostic_state_space_imm_candidate_source",
    ),
    "state_space_valid_occupancy_threshold_s": (
        "diagnostic_state_space_valid_occupancy_threshold_s",
    ),
    "state_space_displacement_radius_bins": (
        "diagnostic_state_space_displacement_radius_bins",
    ),
    "state_space_displacement_position_sigma_cm": (
        "diagnostic_state_space_displacement_position_sigma_cm",
    ),
    "state_space_displacement_transition_sigma_cm_sqrt_s": (
        "diagnostic_state_space_displacement_transition_sigma_cm_sqrt_s",
    ),
    "state_space_displacement_prior_sigma_cm": (
        "diagnostic_state_space_displacement_prior_sigma_cm",
    ),
    "goal_state_space_transition_sigma_cm_sqrt_s": (
        "diagnostic_goal_state_space_transition_sigma_cm_sqrt_s",
    ),
    "goal_state_space_drift_speed_cm_s": ("diagnostic_goal_state_space_drift_speed_cm_s",),
    "goal_state_space_max_step_sigma": ("diagnostic_goal_state_space_max_step_sigma",),
}


def _validate_constant_settings(combined: pd.DataFrame) -> None:
    """Reject aggregates that silently mix incompatible benchmark settings."""

    inconsistent: dict[str, list[str]] = {}
    for setting in _CONSTANT_SETTING_COLUMNS:
        columns = [
            column
            for column in (setting, *_CONSTANT_SETTING_ALIASES.get(setting, ()))
            if column in combined.columns
        ]
        if not columns:
            continue
        values = pd.concat(
            [combined[column] for column in columns],
            ignore_index=True,
        ).dropna().unique()
        if len(values) > 1:
            inconsistent[setting] = sorted(str(value) for value in values)

    if not inconsistent:
        return

    lines = ["Model-evidence shards mix incompatible run settings:"]
    for column, values in sorted(inconsistent.items()):
        lines.append(f"- {column}: {', '.join(values)}")
    raise ValueError("\n".join(lines))

"""Shared validation helpers for model-evidence benchmark outputs."""

from __future__ import annotations

import pandas as pd

_CONSTANT_SETTING_COLUMNS = (
    "bin_size_cm",
    "smoothing_sigma_bins",
    "min_speed_cm_s",
    "time_bin_s",
    "spike_rate_scale",
    "goal_state_space_lateral_sigma_scale",
    "goal_state_space_diffusion_mixture_weight",
    "goal_state_space_reset_probability",
    "goal_state_space_reset_initial_position_prior_weight",
    "goal_state_space_component_switch_probability",
    "goal_state_space_initial_position_prior_direction_mode",
    "goal_state_space_terminal_prior_sigma_cm",
    "goal_state_space_terminal_goal_prior_weight",
    "goal_state_space_initial_goal_prior_sigma_cm",
    "goal_state_space_initial_goal_prior_weight",
    "goal_state_space_toward_direction_prior_weight",
    "goal_state_space_active_goal_prior_weight",
    "goal_state_space_ripple_position_prior_sigma_cm",
    "goal_state_space_ripple_position_prior_weight",
    "goal_state_space_reverse_terminal_position_prior_sigma_cm",
    "goal_state_space_reverse_terminal_position_prior_weight",
    "clusterless_mark_smoothing_sigma_bins",
    "clusterless_mark_prior_count",
    "clusterless_mark_variance_floor",
    "clusterless_rate_floor_hz",
    "clusterless_mark_likelihood",
    "clusterless_mark_kde_bandwidth",
    "clusterless_mark_kde_spatial_sigma_bins",
    "clusterless_mark_kde_max_neighbors",
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

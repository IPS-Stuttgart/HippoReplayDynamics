"""Runtime patch for ground-truth sensitivity metric isolation.

Ground-truth sensitivity analysis reuses one expensive decode and recomputes only
behavioral labels and label-dependent metrics for each label-grid setting.  The
base frame therefore must drop every metric that depends on the reference label
configuration before the next setting is merged in.
"""

from __future__ import annotations

from typing import Any


_SENSITIVITY_METRIC_COLUMNS = frozenset(
    {
        "active_goal_correct",
        "initial_goal_correct",
        "max_over_time_goal_correct",
        "goal_correct_max_posterior",
        "trajectory_mean_goal_correct",
        "goal_correct_integrated",
        "integrated_endpoint_error_cm",
        "true_initial_well_posterior",
        "true_initial_well_rank",
        "true_max_well_posterior",
        "true_max_well_rank",
        "true_trajectory_well_posterior",
        "true_trajectory_well_rank",
        "true_well_max_posterior",
        "true_well_max_rank",
        "true_well_integrated_posterior",
        "true_well_integrated_rank",
        "active_well_posterior",
        "active_initial_well_posterior",
        "active_max_well_posterior",
        "active_trajectory_well_posterior",
        "true_vs_active_trajectory_posterior_margin",
    }
)


def apply_ground_truth_sensitivity_metric_patch(ground_truth_module: Any) -> None:
    """Ensure sensitivity re-labeling drops all label-dependent metrics."""

    existing = set(getattr(ground_truth_module, "_GROUND_TRUTH_COLUMNS_FOR_SENSITIVITY", ()))
    if _SENSITIVITY_METRIC_COLUMNS.issubset(existing):
        return
    ground_truth_module._GROUND_TRUTH_COLUMNS_FOR_SENSITIVITY = existing | set(
        _SENSITIVITY_METRIC_COLUMNS
    )

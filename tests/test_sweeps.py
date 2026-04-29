import numpy as np
import pandas as pd

from hipporeplayimm.sweeps import (
    PyRecEstSweepConfig,
    _ground_truth_summary_metrics,
    pyrecest_parameter_grid,
    sorted_sweep_summary,
)


def test_pyrecest_parameter_grid_cartesian_product():
    config = PyRecEstSweepConfig(
        particles=(64, 128),
        alphas=(0.6, 0.8),
        betas=(1.0,),
        process_noise_sigmas_cm_s=(30.0,),
        position_jump_sigmas_cm=(10.0,),
        jump_probabilities=(0.0, 0.03),
        goal_reset_probabilities=(0.0,),
        initial_velocity_sigmas_cm_s=(120.0,),
    )

    rows = pyrecest_parameter_grid(config)

    assert len(rows) == 8
    assert rows[0]["pyrecest_particles"] == 64
    assert rows[-1]["pyrecest_jump_probability"] == 0.03


def test_ground_truth_summary_metrics_uses_valid_rows():
    comparison = pd.DataFrame(
        {
            "model": [
                "pyrecest-goal-particle",
                "pyrecest-goal-particle",
                "stationary",
                "pyrecest-goal-particle",
            ],
            "valid_label": [True, True, True, False],
            "goal_correct": [True, False, True, np.nan],
            "endpoint_error_cm": [10.0, 20.0, 1.0, np.nan],
            "true_well_posterior": [0.25, 0.75, 0.95, np.nan],
            "true_well_rank": [1.0, 2.0, 1.0, np.nan],
        }
    )

    metrics = _ground_truth_summary_metrics(comparison)

    assert metrics["valid_goal_rows"] == 2
    assert metrics["goal_accuracy"] == 0.5
    assert metrics["median_endpoint_error_cm"] == 15.0
    assert metrics["mean_true_well_posterior"] == 0.5


def test_sorted_sweep_summary_prefers_goal_accuracy_then_likelihood():
    summary = pd.DataFrame(
        {
            "sweep_id": [0, 1, 2],
            "goal_accuracy": [0.25, 0.5, 0.5],
            "mean_heldout_log_likelihood": [-1.0, -3.0, -2.0],
        }
    )

    sorted_summary = sorted_sweep_summary(summary)

    assert list(sorted_summary["sweep_id"]) == [2, 1, 0]

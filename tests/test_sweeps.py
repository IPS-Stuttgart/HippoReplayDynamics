import numpy as np
import pandas as pd

from hipporeplayimm.sweeps import (
    PyRecEstSweepConfig,
    PyRecEstSweepResult,
    _ground_truth_summary_metrics,
    aggregate_sweep_summary,
    pareto_aggregate_sweep_summary,
    pareto_sweep_summary,
    pyrecest_parameter_grid,
    sorted_sweep_summary,
    write_pyrecest_sweep_outputs,
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
    assert rows[0]["random_seed"] == 1
    assert rows[0]["pyrecest_model"] == "pyrecest-goal-particle"
    assert rows[0]["pyrecest_particles"] == 64
    assert rows[-1]["pyrecest_jump_probability"] == 0.03


def test_pyrecest_parameter_grid_can_include_particle_imm_model():
    config = PyRecEstSweepConfig(
        pyrecest_models=("pyrecest-goal-particle", "pyrecest-goal-particle-imm"),
        particles=(64,),
        alphas=(0.8,),
        betas=(1.0,),
        process_noise_sigmas_cm_s=(30.0,),
        position_jump_sigmas_cm=(10.0,),
        jump_probabilities=(0.0,),
        goal_reset_probabilities=(0.0,),
        initial_velocity_sigmas_cm_s=(120.0,),
        imm_mode_stickinesses=(0.9, 0.98),
    )

    rows = pyrecest_parameter_grid(config)

    assert len(rows) == 4
    assert {row["pyrecest_model"] for row in rows} == {
        "pyrecest-goal-particle",
        "pyrecest-goal-particle-imm",
    }
    assert {row["pyrecest_imm_mode_stickiness"] for row in rows} == {0.9, 0.98}


def test_pyrecest_parameter_grid_expands_random_seeds():
    config = PyRecEstSweepConfig(
        random_seed=99,
        random_seeds=(1, 2, 3),
        particles=(64,),
        alphas=(0.8,),
        betas=(1.0,),
        process_noise_sigmas_cm_s=(30.0,),
        position_jump_sigmas_cm=(10.0,),
        jump_probabilities=(0.0,),
        goal_reset_probabilities=(0.0,),
        initial_velocity_sigmas_cm_s=(120.0,),
    )

    rows = pyrecest_parameter_grid(config)

    assert len(rows) == 3
    assert [row["random_seed"] for row in rows] == [1, 2, 3]


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


def test_ground_truth_summary_metrics_can_select_particle_imm_model():
    comparison = pd.DataFrame(
        {
            "model": ["pyrecest-goal-particle", "pyrecest-goal-particle-imm"],
            "valid_label": [True, True],
            "goal_correct": [False, True],
            "endpoint_error_cm": [20.0, 5.0],
            "true_well_posterior": [0.1, 0.8],
            "true_well_rank": [2.0, 1.0],
        }
    )

    metrics = _ground_truth_summary_metrics(
        comparison,
        model="pyrecest-goal-particle-imm",
    )

    assert metrics["valid_goal_rows"] == 1
    assert metrics["goal_accuracy"] == 1.0
    assert metrics["median_endpoint_error_cm"] == 5.0


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


def test_pareto_sweep_summary_keeps_nondominated_tradeoffs():
    summary = pd.DataFrame(
        {
            "sweep_id": [0, 1, 2, 3],
            "goal_accuracy": [0.4, 0.1, 0.2, 0.3],
            "mean_delta_vs_best_static": [-4.0, 1.7, -10.0, -5.0],
            "median_endpoint_error_cm": [53.0, 56.0, 90.0, 54.0],
            "mean_true_well_posterior": [0.15, 0.05, 0.02, 0.10],
        }
    )

    pareto = pareto_sweep_summary(summary)

    assert set(pareto["sweep_id"]) == {0, 1}
    assert list(pareto["sweep_id"]) == [0, 1]


def test_aggregate_sweep_summary_groups_by_hyperparameters_across_seeds():
    summary = pd.DataFrame(
        {
            "sweep_id": [0, 1, 2],
            "random_seed": [1, 2, 1],
            "pyrecest_model": [
                "pyrecest-goal-particle-imm",
                "pyrecest-goal-particle-imm",
                "pyrecest-goal-particle",
            ],
            "pyrecest_particles": [128, 128, 128],
            "goal_accuracy": [0.2, 0.6, 0.1],
            "mean_delta_vs_best_static": [-2.0, 2.0, -1.0],
            "median_endpoint_error_cm": [40.0, 20.0, 30.0],
        }
    )

    aggregate = aggregate_sweep_summary(summary)
    particle_imm = aggregate[
        aggregate["pyrecest_model"] == "pyrecest-goal-particle-imm"
    ].iloc[0]

    assert particle_imm["random_seed_count"] == 2
    assert particle_imm["random_seeds"] == "1,2"
    assert particle_imm["goal_accuracy_mean"] == 0.4
    assert np.isclose(particle_imm["goal_accuracy_std"], np.sqrt(0.08))
    assert particle_imm["mean_delta_vs_best_static_ci95_low"] < 0.0
    assert particle_imm["mean_delta_vs_best_static_ci95_high"] > 0.0


def test_pareto_aggregate_sweep_summary_uses_mean_objectives():
    aggregate = pd.DataFrame(
        {
            "sweep_id": [0, 1, 2],
            "goal_accuracy_mean": [0.2, 0.6, 0.3],
            "mean_delta_vs_best_static_mean": [1.0, -2.0, -3.0],
            "median_endpoint_error_cm_mean": [20.0, 40.0, 50.0],
        }
    )

    pareto = pareto_aggregate_sweep_summary(aggregate)

    assert set(pareto["sweep_id"]) == {0, 1}


def test_write_pyrecest_sweep_outputs_writes_pareto_summary(tmp_path):
    result = PyRecEstSweepResult(
        summary=pd.DataFrame(
            {
                "sweep_id": [0, 1],
                "random_seed": [1, 2],
                "pyrecest_model": [
                    "pyrecest-goal-particle",
                    "pyrecest-goal-particle",
                ],
                "pyrecest_particles": [64, 64],
                "goal_accuracy": [0.3, 0.2],
                "mean_delta_vs_best_static": [0.0, -1.0],
                "median_endpoint_error_cm": [20.0, 25.0],
            }
        ),
        event_scores=pd.DataFrame({"sweep_id": [0], "model": ["stationary"]}),
        ground_truth_comparison=pd.DataFrame(),
        behavioral_ground_truth=None,
    )

    write_pyrecest_sweep_outputs(result, tmp_path)

    pareto = pd.read_csv(tmp_path / "pareto_summary.csv")
    aggregate = pd.read_csv(tmp_path / "aggregate_summary.csv")
    pareto_aggregate = pd.read_csv(tmp_path / "pareto_aggregate_summary.csv")
    assert list(pareto["sweep_id"]) == [0]
    assert list(aggregate["random_seeds"]) == ["1,2"]
    assert list(pareto_aggregate["goal_accuracy_mean"]) == [0.25]

from __future__ import annotations

import pandas as pd

import hipporeplayimm
import hipporeplayimm.sweeps as sweeps
from hipporeplayimm.sweeps import PyRecEstSweepResult


_OPTIONAL_OUTPUT_FILENAMES = (
    "behavioral_ground_truth.csv",
    "ground_truth_comparison.csv",
    "pareto_summary.csv",
    "aggregate_summary.csv",
    "pareto_aggregate_summary.csv",
)


def _result_with_optional_outputs() -> PyRecEstSweepResult:
    return PyRecEstSweepResult(
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
        ground_truth_comparison=pd.DataFrame(
            {"sweep_id": [0], "valid_label": [True]}
        ),
        behavioral_ground_truth=pd.DataFrame({"session": ["Rat1/Open1"]}),
    )


def test_sweep_writer_removes_stale_optional_outputs_on_rerun(tmp_path) -> None:
    assert (
        hipporeplayimm.write_pyrecest_sweep_outputs
        is sweeps.write_pyrecest_sweep_outputs
    )

    hipporeplayimm.write_pyrecest_sweep_outputs(
        _result_with_optional_outputs(),
        tmp_path,
    )
    assert all((tmp_path / filename).exists() for filename in _OPTIONAL_OUTPUT_FILENAMES)

    empty_result = PyRecEstSweepResult(
        summary=pd.DataFrame(),
        event_scores=pd.DataFrame(),
        ground_truth_comparison=pd.DataFrame(),
        behavioral_ground_truth=None,
    )
    hipporeplayimm.write_pyrecest_sweep_outputs(empty_result, tmp_path)

    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "event_scores.csv").exists()
    assert all(
        not (tmp_path / filename).exists()
        for filename in _OPTIONAL_OUTPUT_FILENAMES
    )

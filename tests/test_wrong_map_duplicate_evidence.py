from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    wrong_map_absolute_evidence_deltas,
    wrong_map_delta_summary,
    wrong_map_family_margin_difference_in_differences,
)


def test_wrong_map_delta_summary_collapses_duplicate_model_rows_to_best_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["model-a", "model-a"],
            "log_evidence": [5.0, 3.0],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["model-a", "model-a"],
            "log_evidence": [1.0, 4.0],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_delta_summary(current, wrong)

    assert len(deltas) == 1
    assert float(deltas.loc[0, "log_evidence_current_map"]) == 5.0
    assert float(deltas.loc[0, "log_evidence_wrong_map"]) == 4.0
    assert float(deltas.loc[0, "delta_vs_wrong_environment_map"]) == 1.0


def test_wrong_map_absolute_deltas_use_best_duplicate_model_rows() -> None:
    model = "sorted-spike-state-space-stationary"
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": [model, model],
            "log_evidence": [10.0, 2.0],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "map_session": ["Rat1/Open2", "Rat1/Open2"],
            "model": [model, model],
            "log_evidence": [1.0, 8.0],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_absolute_evidence_deltas(current, wrong)
    fixed = deltas[
        (deltas["statistic"] == model)
        & (deltas["statistic_type"] == "fixed_model")
    ].iloc[0]

    assert float(fixed["log_evidence_real_map"]) == 10.0
    assert float(fixed["log_evidence_wrong_map"]) == 8.0
    assert float(fixed["delta_map_log_evidence"]) == 2.0



def test_wrong_map_difference_in_differences_uses_best_duplicate_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 3,
            "event_index": [0, 0, 0],
            "model": ["stationary", "trajectory", "trajectory"],
            "log_evidence": [1.0, 10.0, 2.0],
            "status": ["success"] * 3,
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "trajectory"],
            "log_evidence": [0.0, 4.0],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_family_margin_difference_in_differences(
        current,
        wrong,
        exact_trajectory_models=("trajectory",),
        nontrajectory_model="stationary",
    )

    assert len(deltas) == 1
    assert deltas.loc[0, "real_best_trajectory_model"] == "trajectory"
    assert deltas.loc[0, "wrong_best_trajectory_model"] == "trajectory"
    assert np.isclose(
        deltas.loc[0, "real_trajectory_minus_nontrajectory_log_evidence"],
        9.0,
    )
    assert np.isclose(
        deltas.loc[0, "wrong_trajectory_minus_nontrajectory_log_evidence"],
        4.0,
    )
    assert np.isclose(
        deltas.loc[0, "margin_difference_in_differences"],
        5.0,
    )


def test_wrong_map_difference_in_differences_skips_impossible_trajectory_family() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "trajectory"],
            "log_evidence": [0.0, -np.inf],
            "status": ["success", "success"],
        }
    )
    wrong = current.copy()

    deltas = wrong_map_family_margin_difference_in_differences(
        current,
        wrong,
        exact_trajectory_models=("trajectory",),
        nontrajectory_model="stationary",
    )

    assert deltas.empty

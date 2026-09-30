from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    wrong_map_absolute_evidence_deltas,
    wrong_map_delta_summary,
    wrong_map_family_margin_difference_in_differences,
)


def test_wrong_map_delta_summary_coerces_csv_string_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": ["9.5", "10.0"],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": ["8.0", "9.0"],
            "status": ["success", "success"],
        }
    )

    out = wrong_map_delta_summary(current, wrong).set_index("model")

    assert np.isclose(out.loc["stationary", "delta_vs_wrong_environment_map"], 1.5)
    assert np.isclose(out.loc["diffusion", "delta_vs_wrong_environment_map"], 1.0)
    assert out["wrong_map_best_model"].unique().tolist() == ["diffusion"]


def test_wrong_map_absolute_deltas_ignore_nonfinite_csv_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": [
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
            ],
            "log_evidence": ["1.0", "12.0"],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "map_session": ["Rat1/Open2", "Rat1/Open2"],
            "model": [
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
            ],
            "log_evidence": ["-1.0", "not-a-number"],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        fixed_models=("sorted-spike-state-space-stationary",),
        exact_core_models=(
            "sorted-spike-state-space-stationary",
            "sorted-spike-state-space-first-order-imm",
        ),
        exact_trajectory_models=("sorted-spike-state-space-first-order-imm",),
    ).set_index("statistic")

    assert np.isclose(
        deltas.loc[
            "sorted-spike-state-space-stationary",
            "delta_map_log_evidence",
        ],
        2.0,
    )
    assert deltas.loc[
        "best_exact_core_model_real_map",
        "selected_model",
    ] == "sorted-spike-state-space-stationary"
    assert "best_exact_trajectory_model_real_map" not in deltas.index


def test_wrong_map_delta_summary_preserves_negative_infinite_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": [1.0, 2.0],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["stationary", "diffusion"],
            "log_evidence": [-np.inf, -np.inf],
            "status": ["success", "success"],
        }
    )

    out = wrong_map_delta_summary(current, wrong).set_index("model")

    assert np.isposinf(out.loc["stationary", "delta_vs_wrong_environment_map"])
    assert np.isposinf(out.loc["diffusion", "delta_vs_wrong_environment_map"])
    assert out["wrong_map_best_model"].unique().tolist() == [""]


def test_wrong_map_absolute_deltas_keep_negative_infinite_wrong_map_rows() -> None:
    stationary = "sorted-spike-state-space-stationary"
    trajectory = "sorted-spike-state-space-first-order-imm"
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": [stationary, trajectory],
            "log_evidence": [1.0, 12.0],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "map_session": ["Rat1/Open2", "Rat1/Open2"],
            "model": [stationary, trajectory],
            "log_evidence": [-np.inf, -np.inf],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        fixed_models=(stationary,),
        exact_core_models=(stationary, trajectory),
        exact_trajectory_models=(trajectory,),
    ).set_index("statistic")

    assert np.isposinf(deltas.loc[stationary, "delta_map_log_evidence"])
    assert deltas.loc["best_exact_core_model_real_map", "selected_model"] == trajectory
    assert np.isposinf(
        deltas.loc["best_exact_core_model_real_map", "delta_map_log_evidence"]
    )
    assert deltas.loc["best_exact_trajectory_model_real_map", "selected_model"] == trajectory


def test_wrong_map_selected_winners_remain_unresolved_when_real_map_is_all_impossible() -> None:
    stationary = "sorted-spike-state-space-stationary"
    trajectory = "sorted-spike-state-space-first-order-imm"
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": [stationary, trajectory],
            "log_evidence": [-np.inf, -np.inf],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "map_session": ["Rat1/Open2", "Rat1/Open2"],
            "model": [stationary, trajectory],
            "log_evidence": [-np.inf, -np.inf],
            "status": ["success", "success"],
        }
    )

    deltas = wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        fixed_models=(stationary,),
        exact_core_models=(stationary, trajectory),
        exact_trajectory_models=(trajectory,),
    ).set_index("statistic")

    assert deltas.index.tolist() == [stationary]
    assert deltas.loc[stationary, "delta_map_log_evidence"] == 0.0

    did = wrong_map_family_margin_difference_in_differences(
        current,
        wrong,
        exact_trajectory_models=(trajectory,),
        nontrajectory_model=stationary,
    )
    assert did.empty

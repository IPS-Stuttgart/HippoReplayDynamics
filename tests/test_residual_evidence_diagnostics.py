from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    model_disagreement_events,
    rat_bootstrap_wrong_map_absolute_evidence_summary,
    wrong_map_absolute_evidence_deltas,
    wrong_map_delta_summary,
)


def test_wrong_map_pairwise_preserves_negative_infinite_evidence() -> None:
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["model-a", "model-b"],
            "log_evidence": [5.0, float("-inf")],
            "status": ["success", "success"],
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["model-a", "model-b"],
            "log_evidence": [float("-inf"), float("-inf")],
            "status": ["success", "success"],
        }
    )

    out = wrong_map_delta_summary(current, wrong).set_index("model")

    assert np.isposinf(out.loc["model-a", "delta_vs_wrong_environment_map"])
    assert out.loc["model-b", "delta_vs_wrong_environment_map"] == 0.0
    assert set(out["wrong_map_best_model"]) == {""}


def test_wrong_map_absolute_keeps_infinite_attenuation_without_pseudo_winner() -> None:
    stationary = "stationary"
    imm = "first-order-imm"
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0, 0, 1, 1],
            "model": [stationary, imm, stationary, imm],
            "log_evidence": [5.0, 6.0, float("-inf"), float("-inf")],
            "status": ["success"] * 4,
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0, 0, 1, 1],
            "map_session": ["Rat1/Open2"] * 4,
            "model": [stationary, imm, stationary, imm],
            "log_evidence": [float("-inf")] * 4,
            "status": ["success"] * 4,
        }
    )

    out = wrong_map_absolute_evidence_deltas(
        current,
        wrong,
        fixed_models=(stationary,),
        exact_core_models=(stationary, imm),
        exact_trajectory_models=(imm,),
    )

    fixed0 = out[
        (out["event_index"] == 0)
        & (out["statistic"] == stationary)
        & (out["statistic_type"] == "fixed_model")
    ].iloc[0]
    fixed1 = out[
        (out["event_index"] == 1)
        & (out["statistic"] == stationary)
        & (out["statistic_type"] == "fixed_model")
    ].iloc[0]

    assert np.isposinf(fixed0["delta_map_log_evidence"])
    assert fixed1["delta_map_log_evidence"] == 0.0
    assert not (
        (out["event_index"] == 1)
        & out["statistic_type"].eq("real_map_selected_model")
    ).any()


def test_model_disagreement_uses_comparable_evidence_and_keeps_impossible_unresolved() -> None:
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0, 0, 1, 1],
            "model": ["exact", "lower-bound", "a", "b"],
            "log_evidence": [1.0, 100.0, float("-inf"), float("-inf")],
            "status": ["success"] * 4,
            "evidence_comparable": [True, False, True, True],
            "is_best_model": [False, True, True, False],
        }
    )

    out = model_disagreement_events(scores).set_index("event_index")

    assert out.loc[0, "best_model"] == "exact"
    assert out.loc[0, "best_model_by_evidence"] == "exact"
    assert out.loc[1, "best_model"] == ""
    assert out.loc[1, "best_model_by_evidence"] == ""



def test_wrong_map_bootstrap_keeps_all_infinite_attenuation_defined() -> None:
    deltas = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat2/Open1"],
            "statistic": ["fixed_model", "fixed_model"],
            "selected_model": ["m", "m"],
            "delta_map_log_evidence": [float("inf"), float("inf")],
        }
    )

    out = rat_bootstrap_wrong_map_absolute_evidence_summary(
        deltas,
        n_bootstrap=8,
        random_seed=3,
    ).iloc[0]

    assert np.isposinf(out["observed_mean_delta_map_log_evidence"])
    assert np.isposinf(out["mean_delta_ci95_low"])
    assert np.isposinf(out["mean_delta_ci95_high"])
    assert np.isposinf(out["median_delta_ci95_low"])
    assert np.isposinf(out["median_delta_ci95_high"])
    assert out["probability_mean_delta_gt_0"] == 1.0

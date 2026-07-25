import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_swr_off_swr_dynamics import (
    DEFAULT_FIRST_ORDER_IMM_MODEL,
    DEFAULT_MARGIN_POSITIVE_MODEL,
    DEFAULT_MARGIN_REFERENCE_MODEL,
    FRAGMENTED_MODEL,
    STATIONARY_MODEL,
    family_margin_summary,
    model_winner_summary,
    write_swr_off_swr_dynamics_outputs,
)


def test_swr_off_swr_comparison_keeps_promoted_windows_distinct(tmp_path):
    swr_path = tmp_path / "all_sessions_event_model_evidence.csv"
    off_path = tmp_path / "promoted_off_swr_candidate_exact_core_event_model_evidence.csv"
    high_path = tmp_path / "off_swr_high_specificity_candidate_table.csv"
    output = tmp_path / "comparison"

    pd.DataFrame(
        [
            *_swr_event("Rat1/Open1", 1, first_order=30.0, momentum=12.0, stationary=0.0, speed=1.0),
            *_swr_event("Rat1/Open1", 2, first_order=25.0, momentum=10.0, stationary=0.0, speed=2.0),
            *_swr_event("Rat2/Open1", 3, first_order=14.0, momentum=22.0, stationary=0.0, speed=1.5),
        ]
    ).to_csv(swr_path, index=False)
    pd.DataFrame(
        [
            *_off_swr_candidate("Rat1/Open1", 10, 0, first_order=40.0, momentum=12.0, stationary=0.0),
            *_off_swr_candidate("Rat1/Open1", 10, 1, first_order=35.0, momentum=14.0, stationary=0.0),
        ]
    ).to_csv(off_path, index=False)
    pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 10,
                "null_index": 0,
                "passes_high_specificity_promotion_filter": True,
                "run_or_immobility_state": "immobile",
                "animal_speed_mean": 1.0,
                "n_spikes": 60,
                "active_cell_count": 22,
                "duration_s": 0.12,
                "distance_to_nearest_swr_s": 50.0,
            },
            {
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 11,
                "null_index": 3,
                "passes_high_specificity_promotion_filter": False,
                "run_or_immobility_state": "run",
                "animal_speed_mean": 24.0,
                "n_spikes": 110,
                "active_cell_count": 45,
                "duration_s": 0.18,
                "distance_to_nearest_swr_s": 20.0,
            },
        ]
    ).to_csv(high_path, index=False)

    outputs = write_swr_off_swr_dynamics_outputs(
        swr_event_model_evidence=swr_path,
        off_swr_event_model_evidence=off_path,
        off_swr_high_specificity_candidates=high_path,
        output=output,
        margin_threshold=5.5,
    )

    comparison = outputs["swr_off_swr_dynamics_comparison.csv"]
    assert comparison.shape[0] == 5
    off = comparison[comparison["event_class"].eq("promoted_off_swr")]
    assert off["candidate_id"].nunique() == 2
    assert sorted(off["null_index"].astype(int).tolist()) == [0, 1]
    assert off["trajectory_confident_claim"].astype(bool).all()

    model = outputs["swr_off_swr_model_winner_summary.csv"]
    swr_first_order = model[model["event_class"].eq("detected_replay_or_swr") & model["best_exact_trajectory_model"].eq(DEFAULT_FIRST_ORDER_IMM_MODEL)].iloc[0]
    assert int(swr_first_order["events"]) == 2

    behavior = outputs["swr_off_swr_behavior_summary.csv"].set_index("event_class")
    assert int(behavior.loc["promoted_off_swr", "immobile_events"]) == 2
    assert int(behavior.loc["rejected_high_specificity_off_swr_candidates", "running_events"]) == 1

    gates = outputs["swr_off_swr_gate_summary.csv"].set_index("gate")
    assert bool(gates.loc["first_order_imm_dominates_both_classes", "passed"])
    assert bool(gates.loc["off_swr_candidates_all_immobile", "passed"])
    assert bool(gates.loc["overall", "passed"])
    for filename in outputs:
        assert (output / filename).exists()


def _swr_event(
    session: str,
    event_index: int,
    *,
    first_order: float,
    momentum: float,
    stationary: float,
    speed: float,
) -> list[dict[str, object]]:
    base = {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "evidence_comparable": True,
        "event_start_s": 10.0 + event_index,
        "event_end_s": 10.12 + event_index,
        "event_duration_s": 0.12,
        "n_spikes": 50 + event_index,
        "active_cell_count": 20 + event_index,
        "animal_speed_mean": speed,
        "run_or_immobility_state": "immobile",
    }
    return _model_rows(
        base,
        {
            STATIONARY_MODEL: stationary,
            DEFAULT_MARGIN_REFERENCE_MODEL: 8.0,
            FRAGMENTED_MODEL: 3.0,
            DEFAULT_FIRST_ORDER_IMM_MODEL: first_order,
            DEFAULT_MARGIN_POSITIVE_MODEL: momentum,
        },
    )


def _off_swr_candidate(
    session: str,
    event_index: int,
    null_index: int,
    *,
    first_order: float,
    momentum: float,
    stationary: float,
) -> list[dict[str, object]]:
    base = {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "window_role": "promoted_off_swr_candidate",
        "null_index": null_index,
        "evidence_comparable": True,
        "window_start_s": 100.0 + null_index,
        "window_end_s": 100.12 + null_index,
        "window_duration_s": 0.12,
        "n_spikes": 60 + null_index,
        "active_cell_count": 25 + null_index,
        "animal_speed_mean": 1.0 + null_index,
        "animal_speed_median": 0.8 + null_index,
        "animal_speed_max": 2.0 + null_index,
        "run_or_immobility_state": "immobile",
        "distance_to_nearest_swr_s": 60.0 + null_index,
        "overlaps_known_swr": False,
        "candidate_rank": null_index + 1,
    }
    return _model_rows(
        base,
        {
            STATIONARY_MODEL: stationary,
            DEFAULT_MARGIN_REFERENCE_MODEL: 9.0,
            FRAGMENTED_MODEL: 4.0,
            DEFAULT_FIRST_ORDER_IMM_MODEL: first_order,
            DEFAULT_MARGIN_POSITIVE_MODEL: momentum,
        },
    )


def _model_rows(base: dict[str, object], values: dict[str, float]) -> list[dict[str, object]]:
    return [{**base, "model": model, "log_evidence": value} for model, value in values.items()]


def test_swr_off_swr_summaries_ignore_missing_winner_labels():
    comparison = pd.DataFrame(
        [
            {
                "event_class": "detected_replay_or_swr",
                "best_exact_trajectory_model": "",
                "required_models_complete": False,
                "margin_decision": "incomplete_core",
                "trajectory_minus_nontrajectory_margin": float("nan"),
                "trajectory_confident_claim": False,
                "nontrajectory_confident_claim": False,
            },
            {
                "event_class": "detected_replay_or_swr",
                "best_exact_trajectory_model": DEFAULT_FIRST_ORDER_IMM_MODEL,
                "required_models_complete": True,
                "margin_decision": "trajectory",
                "trajectory_minus_nontrajectory_margin": 12.0,
                "trajectory_confident_claim": True,
                "nontrajectory_confident_claim": False,
            },
        ]
    )

    winners = model_winner_summary(comparison)
    assert winners["best_exact_trajectory_model"].tolist() == [DEFAULT_FIRST_ORDER_IMM_MODEL]
    assert winners["events"].tolist() == [1]
    assert winners["fraction_of_event_class"].tolist() == [0.5]

    family = family_margin_summary(comparison)
    assert family.loc[0, "most_common_best_exact_trajectory_model"] == DEFAULT_FIRST_ORDER_IMM_MODEL

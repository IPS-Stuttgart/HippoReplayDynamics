import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_behavior_alignment import (  # noqa: E402
    ALIGNMENT_OUTPUT,
    LOO_OUTPUT,
    PREDICTION_SUMMARY_OUTPUT,
    RAT_SUMMARY_OUTPUT,
    build_event_evidence_features,
    write_replay_behavior_alignment_outputs,
)


def test_replay_behavior_alignment_outputs_prediction_summaries(tmp_path):
    evidence = pd.DataFrame(
        [
            *_event_rows("Rat1/Open1", 0, trajectory=8.0, stationary=0.0, endpoint=(10.0, 0.0), momentum_bonus=0.0),
            *_event_rows("Rat1/Open1", 1, trajectory=2.0, stationary=0.0, endpoint=(-10.0, 0.0), momentum_bonus=-4.0),
            *_event_rows("Rat2/Open1", 0, trajectory=7.0, stationary=0.0, endpoint=(0.0, 10.0), momentum_bonus=0.0),
            *_event_rows("Rat2/Open1", 1, trajectory=1.0, stationary=0.0, endpoint=(0.0, -10.0), momentum_bonus=-4.0),
        ]
    )
    behavior = pd.DataFrame(
        [
            _context("Rat1/Open1", 0, current=(0.0, 0.0), previous=(-2.0, 0.0), future=(10.0, 0.0), goal=(12.0, 0.0)),
            _context("Rat1/Open1", 1, current=(0.0, 0.0), previous=(-2.0, 0.0), future=(10.0, 0.0), goal=(12.0, 0.0)),
            _context("Rat2/Open1", 0, current=(0.0, 0.0), previous=(0.0, -2.0), future=(0.0, 10.0), goal=(0.0, 12.0)),
            _context("Rat2/Open1", 1, current=(0.0, 0.0), previous=(0.0, -2.0), future=(0.0, 10.0), goal=(0.0, 12.0)),
        ]
    )

    outputs = write_replay_behavior_alignment_outputs(evidence, tmp_path, behavior_context=behavior)

    alignment = outputs[ALIGNMENT_OUTPUT]
    assert len(alignment) == 4
    assert alignment.loc[alignment["event_index"].eq(0), "alignment_with_next_movement"].tolist() == [1.0, 1.0]
    assert alignment.loc[alignment["event_index"].eq(1), "alignment_with_next_movement"].tolist() == [-1.0, -1.0]
    assert alignment["distance_to_current_position_cm"].tolist() == [10.0, 10.0, 10.0, 10.0]

    summary = outputs[PREDICTION_SUMMARY_OUTPUT].set_index("analysis")
    assert summary.loc["endpoint_predicts_next_movement", "mean_alignment_with_next_movement"] == pytest.approx(0.0)
    assert summary.loc["trajectory_confidence_predicts_alignment", "predictor_correlation"] > 0.9
    assert summary.loc["trajectory_confidence_predicts_alignment", "high_minus_low_alignment"] == pytest.approx(2.0)
    assert summary.loc["momentum_index_predicts_alignment", "predictor_correlation"] > 0.9

    rat = outputs[RAT_SUMMARY_OUTPUT]
    assert set(rat["rat"]) == {"Rat1", "Rat2"}
    assert rat["high_minus_low_trajectory_confidence_alignment"].tolist() == [2.0, 2.0]

    loo = outputs[LOO_OUTPUT]
    assert set(loo["held_out_rat"]) == {"Rat1", "Rat2"}
    assert loo["test_high_minus_low_alignment"].tolist() == [2.0, 2.0]

    for filename in outputs:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_behavior_alignment_keeps_legacy_real_event_rows():
    legacy = pd.DataFrame(
        _event_rows("Rat1/Open1", 0, trajectory=8.0, stationary=0.0, endpoint=(10.0, 0.0), momentum_bonus=0.0)
    )
    legacy["status"] = ""
    legacy["window_role"] = pd.NA
    failed = pd.DataFrame(
        _event_rows("Rat1/Open1", 1, trajectory=100.0, stationary=0.0, endpoint=(20.0, 0.0), momentum_bonus=0.0)
    )
    failed["status"] = "failed"
    failed["window_role"] = "real"
    off_swr = pd.DataFrame(
        _event_rows("Rat1/Open1", 2, trajectory=100.0, stationary=0.0, endpoint=(30.0, 0.0), momentum_bonus=0.0)
    )
    off_swr["status"] = "success"
    off_swr["window_role"] = "promoted_off_swr_candidate"
    evidence = pd.concat([legacy, failed, off_swr], ignore_index=True)

    features = build_event_evidence_features(evidence)

    assert len(features) == 1
    assert features.iloc[0]["session"] == "Rat1/Open1"
    assert int(features.iloc[0]["event_index"]) == 0
    assert bool(features.iloc[0]["exact_core_complete"])


def _event_rows(
    session: str,
    event_index: int,
    *,
    trajectory: float,
    stationary: float,
    endpoint: tuple[float, float],
    momentum_bonus: float,
) -> list[dict[str, object]]:
    models = [
        ("sorted-spike-state-space-stationary", stationary, "nontrajectory", (0.0, 0.0)),
        ("sorted-spike-state-space-diffusion", trajectory - 3.0, "trajectory", endpoint),
        ("sorted-spike-state-space-fragmented", trajectory - 2.0, "trajectory", endpoint),
        ("sorted-spike-state-space-first-order-imm", trajectory - 1.0, "trajectory", endpoint),
        ("sorted-spike-state-space-momentum-exact-sparse", trajectory + momentum_bonus, "trajectory", endpoint),
    ]
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "model": model,
            "requested_model": model,
            "model_family": family,
            "log_evidence": log_evidence,
            "diagnostic_decoded_endpoint_x": xy[0],
            "diagnostic_decoded_endpoint_y": xy[1],
            "diagnostic_decoded_map_x": xy[0],
            "diagnostic_decoded_map_y": xy[1],
            "diagnostic_terminal_posterior_entropy": 0.5,
            "diagnostic_mean_trajectory_posterior_entropy": 0.6,
            "evidence_comparable": True,
        }
        for model, log_evidence, family, xy in models
    ]


def _context(
    session: str,
    event_index: int,
    *,
    current: tuple[float, float],
    previous: tuple[float, float],
    future: tuple[float, float],
    goal: tuple[float, float],
) -> dict[str, object]:
    return {
        "session": session,
        "event_index": event_index,
        "event_start_s": float(event_index),
        "event_end_s": float(event_index) + 0.1,
        "event_peak_s": float(event_index) + 0.05,
        "current_x": current[0],
        "current_y": current[1],
        "previous_x": previous[0],
        "previous_y": previous[1],
        "future_x": future[0],
        "future_y": future[1],
        "active_goal_id": 1,
        "active_goal_x": goal[0],
        "active_goal_y": goal[1],
        "nearest_current_well_id": 1,
        "nearest_current_well_x": goal[0],
        "nearest_current_well_y": goal[1],
        "well_locations_available": True,
    }

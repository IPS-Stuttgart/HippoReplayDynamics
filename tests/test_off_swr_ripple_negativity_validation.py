import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from off_swr_ripple_negativity_validation import write_outputs  # noqa: E402


def test_ripple_negativity_gate_passes_with_complete_negative_lfp(tmp_path):
    promoted = tmp_path / "promoted.csv"
    windows = tmp_path / "windows.csv"
    dynamics = tmp_path / "dynamics.csv"
    output = tmp_path / "out"

    pd.DataFrame(
        [
            _promoted("Rat1/Open1", 1, 0, exact_margin=50.0),
            _promoted("Rat1/Open1", 2, 1, exact_margin=60.0),
        ]
    ).to_csv(promoted, index=False)
    pd.DataFrame(
        [
            _lfp_window("Rat1/Open1", 1, 0, peak=1.2, mean=0.4, promoted=True),
            _lfp_window("Rat1/Open1", 2, 1, peak=1.7, mean=0.5, promoted=True),
            _lfp_window("Rat1/Open1", 3, 2, peak=4.2, mean=1.4, promoted=False),
        ]
    ).to_csv(windows, index=False)
    pd.DataFrame(
        [
            _dynamics("detected_replay_or_swr", "Rat1/Open1", 10, pd.NA, peak=5.0, margin=80.0),
            _dynamics("promoted_off_swr", "Rat1/Open1", 1, 0, peak=1.2, margin=50.0),
            _dynamics("promoted_off_swr", "Rat1/Open1", 2, 1, peak=1.7, margin=60.0),
        ]
    ).to_csv(dynamics, index=False)

    outputs = write_outputs(
        promoted_decisions=promoted,
        off_swr_window_table=windows,
        swr_off_swr_dynamics=dynamics,
        output=output,
        ripple_z_threshold=3.0,
    )

    gate = outputs["off_swr_candidate_lfp_gate_summary.csv"].set_index("gate")
    assert bool(gate.loc["overall", "passed"])
    assert gate.loc["candidate_buffers_ripple_negative", "observed"] == "2/2"

    promoted_lfp = outputs["off_swr_candidate_lfp_ripple_power.csv"]
    assert promoted_lfp["ripple_negative_with_buffers"].astype(bool).all()
    assert promoted_lfp["peak_ripple_band_power_z"].tolist() == [1.2, 1.7]

    null_summary = outputs["off_swr_candidate_ripple_power_matched_null.csv"].set_index("comparison_group")
    assert int(null_summary.loc["non_promoted_off_swr_windows", "window_threshold_crossing_windows"]) == 1

    joint = outputs["swr_off_swr_lfp_dynamics_comparison.csv"].set_index("event_class")
    assert int(joint.loc["promoted_off_swr", "ripple_negative_events"]) == 2
    assert int(joint.loc["detected_replay_or_swr", "window_threshold_crossing_events"]) == 1

    for filename in outputs:
        assert (output / filename).exists()


def test_ripple_negativity_gate_blocks_when_lfp_columns_are_absent(tmp_path):
    promoted = tmp_path / "promoted.csv"
    windows = tmp_path / "windows.csv"
    output = tmp_path / "out"

    pd.DataFrame([_promoted("Rat1/Open1", 1, 0, exact_margin=50.0)]).to_csv(promoted, index=False)
    pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "null_index": 0,
                "distance_to_nearest_swr_s": 100.0,
                "run_or_immobility_state": "immobile",
            }
        ]
    ).to_csv(windows, index=False)

    outputs = write_outputs(
        promoted_decisions=promoted,
        off_swr_window_table=windows,
        output=output,
    )

    promoted_lfp = outputs["off_swr_candidate_lfp_ripple_power.csv"]
    assert promoted_lfp.iloc[0]["lfp_validation_status"] == "unsupported_no_lfp_power_columns"

    gate = outputs["off_swr_candidate_lfp_gate_summary.csv"].set_index("gate")
    assert not bool(gate.loc["lfp_power_or_crossing_fields_available", "passed"])
    assert not bool(gate.loc["overall", "passed"])


def _promoted(session: str, event_index: int, null_index: int, *, exact_margin: float) -> dict[str, object]:
    return {
        "session": session,
        "rat": session.split("/")[0],
        "event_index": event_index,
        "null_index": null_index,
        "window_start_s": 10.0 + event_index,
        "window_end_s": 10.1 + event_index,
        "window_duration_s": 0.1,
        "run_or_immobility_state": "immobile",
        "animal_speed_mean": 1.0,
        "distance_to_nearest_swr_s": 100.0,
        "trajectory_family_margin": exact_margin,
        "trajectory_minus_nontrajectory_log_evidence": exact_margin,
        "trajectory_confident_claim": True,
        "nontrajectory_confident_claim": False,
    }


def _lfp_window(
    session: str,
    event_index: int,
    null_index: int,
    *,
    peak: float,
    mean: float,
    promoted: bool,
) -> dict[str, object]:
    return {
        "session": session,
        "event_index": event_index,
        "null_index": null_index,
        "peak_ripple_band_power_z": peak,
        "mean_ripple_band_power_z": mean,
        "sharp_wave_power_z": mean + 0.2,
        "ripple_threshold_crossing_window": peak >= 3.0,
        "ripple_threshold_crossing_pm50ms": peak >= 3.0,
        "ripple_threshold_crossing_pm100ms": peak >= 3.0,
        "ripple_threshold_crossing_pm250ms": peak >= 3.0,
        "run_or_immobility_state": "immobile" if promoted else "run",
        "animal_speed_mean": 1.0 if promoted else 20.0,
        "distance_to_nearest_swr_s": 100.0,
    }


def _dynamics(
    event_class: str,
    session: str,
    event_index: int,
    null_index: object,
    *,
    peak: float,
    margin: float,
) -> dict[str, object]:
    return {
        "event_class": event_class,
        "session": session,
        "event_index": event_index,
        "null_index": null_index,
        "trajectory_confident_claim": True,
        "best_exact_trajectory_model": "sorted-spike-state-space-first-order-imm",
        "trajectory_minus_nontrajectory_margin": margin,
        "peak_ripple_band_power_z": peak,
        "ripple_threshold_crossing_window": peak >= 3.0,
    }

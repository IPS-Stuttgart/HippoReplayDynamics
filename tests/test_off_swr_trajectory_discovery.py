import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from off_swr_trajectory_discovery import (  # noqa: E402
    AMBIGUOUS_CLASS,
    EXCLUDED_SWR_OVERLAP_CLASS,
    STATIC_NONTRAJECTORY_CLASS,
    TRAJECTORY_CANDIDATE_CLASS,
    write_off_swr_trajectory_discovery_outputs,
)


def test_off_swr_discovery_classifies_clusters_and_reports_covariates(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows("Rat1/Open1", 0, "matched_null", 0, stationary=0.0, trajectory=8.0, start=10.0, ripple_power=1.2),
            *_event_rows("Rat1/Open1", 1, "matched_null", 0, stationary=0.0, trajectory=9.0, start=10.25, ripple_power=1.4),
            *_event_rows("Rat1/Open1", 2, "matched_null", 0, stationary=12.0, trajectory=0.0, start=11.0, ripple_power=0.2),
            *_event_rows("Rat1/Open1", 3, "matched_null", 0, stationary=0.0, trajectory=2.0, start=12.0, ripple_power=0.6),
            *_event_rows(
                "Rat1/Open1",
                4,
                "matched_null",
                0,
                stationary=0.0,
                trajectory=9.0,
                start=13.0,
                ripple_power=1.5,
                off_swr=False,
            ),
            *_event_rows("Rat1/Open1", 5, "real", -1, stationary=0.0, trajectory=20.0, start=1.0, ripple_power=2.0, off_swr=False),
        ]
    )

    outputs = write_off_swr_trajectory_discovery_outputs(scores, tmp_path, cluster_gap_s=0.5)

    decisions = outputs["off_swr_trajectory_discovery_decisions.csv"]
    classes_by_event = dict(zip(decisions["event_index"], decisions["candidate_class"], strict=True))
    assert classes_by_event[0] == TRAJECTORY_CANDIDATE_CLASS
    assert classes_by_event[1] == TRAJECTORY_CANDIDATE_CLASS
    assert classes_by_event[2] == STATIC_NONTRAJECTORY_CLASS
    assert classes_by_event[3] == AMBIGUOUS_CLASS
    assert classes_by_event[4] == EXCLUDED_SWR_OVERLAP_CLASS
    assert 5 not in classes_by_event

    candidates = outputs["off_swr_trajectory_candidate_events.csv"]
    assert len(candidates) == 2
    assert set(candidates["event_index"]) == {0, 1}
    assert candidates["passes_known_swr_exclusion"].all()

    summary = outputs["off_swr_trajectory_candidate_summary.csv"].iloc[0]
    assert int(summary["windows"]) == 5
    assert int(summary["off_swr_windows"]) == 4
    assert int(summary["trajectory_family_candidates"]) == 2
    assert int(summary["static_nontrajectory_windows"]) == 1
    assert int(summary["ambiguous_windows"]) == 1
    assert int(summary["excluded_known_swr_overlap_windows"]) == 1

    clusters = outputs["off_swr_candidate_clusters.csv"]
    assert len(clusters) == 1
    assert int(clusters.iloc[0]["window_count"]) == 2
    assert clusters.iloc[0]["template_event_indices"] == "0 1"

    behavior = outputs["off_swr_candidate_behavior_lfp_summary.csv"]
    ripple_rows = behavior[behavior["feature"].eq("ripple_power")]
    assert not ripple_rows.empty
    assert ripple_rows["feature_available"].all()
    assert set(ripple_rows["candidate_class"]) >= {TRAJECTORY_CANDIDATE_CLASS, STATIC_NONTRAJECTORY_CLASS, AMBIGUOUS_CLASS}

    gates = outputs["off_swr_candidate_gate_summary.csv"]
    overall = gates[gates["gate"].eq("overall")].iloc[0]
    assert bool(overall["passed"])
    candidate_gate = gates[gates["gate"].eq("trajectory_candidates_detected")].iloc[0]
    assert bool(candidate_gate["passed"])
    assert not bool(candidate_gate["required_for_overall"])

    for filename in outputs:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def _event_rows(
    session: str,
    event_index: int,
    window_role: str,
    null_index: int,
    *,
    stationary: float,
    trajectory: float,
    start: float,
    ripple_power: float,
    off_swr: bool = True,
) -> list[dict[str, object]]:
    models = [
        ("sorted-spike-state-space-stationary", stationary, "nontrajectory"),
        ("sorted-spike-state-space-diffusion", trajectory - 3.0, "trajectory"),
        ("sorted-spike-state-space-fragmented", trajectory - 2.0, "trajectory"),
        ("sorted-spike-state-space-first-order-imm", trajectory, "trajectory"),
        ("sorted-spike-state-space-momentum-exact-sparse", trajectory - 1.0, "trajectory"),
    ]
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "window_role": window_role,
            "event_window_variant": "core" if window_role == "real" else "matched_null",
            "null_index": null_index,
            "matched_null_rank": max(null_index, 0),
            "template_event_index": event_index,
            "window_start_s": start,
            "window_end_s": start + 0.1,
            "window_duration_s": 0.1,
            "real_event_start_s": 1.0 + event_index,
            "real_event_end_s": 1.1 + event_index,
            "real_event_duration_s": 0.1,
            "model": model,
            "requested_model": model,
            "model_family": family,
            "log_evidence": log_evidence,
            "n_time": 25,
            "n_spikes": 10 + event_index,
            "null_active_cell_count": 5,
            "real_n_spikes": 10,
            "n_spikes_delta": event_index,
            "n_spikes_relative_delta": event_index / 10.0,
            "ripple_power": ripple_power,
            "off_swr": off_swr,
            "evidence_comparable": True,
        }
        for model, log_evidence, family in models
    ]

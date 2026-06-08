import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from causal_trajectory_family_detector import (  # noqa: E402
    EVENT_TABLE_OUTPUT,
    FALSE_POSITIVE_OUTPUT,
    GATE_OUTPUT,
    LATENCY_OUTPUT,
    OFFLINE_AGREEMENT_OUTPUT,
    TIME_BIN_TABLE_OUTPUT,
    TRAJECTORY_LABEL,
    write_causal_replay_detection_outputs,
)


def test_causal_detector_reports_latency_agreement_and_false_positives(tmp_path):
    prefix_scores = pd.DataFrame(
        [
            *_prefix_rows("Rat1/Open1", 0, "real", -1, 1, stationary=0.0, trajectory=3.0, start=1.0, prefix_end=1.01),
            *_prefix_rows("Rat1/Open1", 0, "real", -1, 2, stationary=0.0, trajectory=7.0, start=1.0, prefix_end=1.02),
            *_prefix_rows("Rat1/Open1", 0, "real", -1, 3, stationary=0.0, trajectory=8.0, start=1.0, prefix_end=1.03),
            *_prefix_rows("Rat1/Open1", 0, "matched_null", 0, 1, stationary=0.0, trajectory=2.0, start=3.0, prefix_end=3.01, off_swr=True),
            *_prefix_rows("Rat1/Open1", 0, "matched_null", 0, 2, stationary=0.0, trajectory=3.0, start=3.0, prefix_end=3.02, off_swr=True),
            *_prefix_rows("Rat1/Open1", 0, "matched_null", 1, 1, stationary=0.0, trajectory=6.0, start=4.0, prefix_end=4.01, off_swr=True),
        ]
    )
    offline_scores = pd.DataFrame(
        [
            *_offline_rows("Rat1/Open1", 0, stationary=0.0, trajectory=9.0),
        ]
    )

    outputs = write_causal_replay_detection_outputs(
        prefix_scores,
        tmp_path,
        offline_event_model_evidence=offline_scores,
    )

    time_bins = outputs[TIME_BIN_TABLE_OUTPUT]
    assert set(time_bins["causal_label"]) >= {TRAJECTORY_LABEL, "ambiguous"}
    assert "p_static" in time_bins
    assert "p_nonlocal_replay" in time_bins

    event_table = outputs[EVENT_TABLE_OUTPUT]
    real = event_table[event_table["window_role"].eq("real")].iloc[0]
    assert bool(real["has_causal_trajectory_claim"])
    assert real["latency_to_trajectory_claim_s"] == pytest.approx(0.02)
    assert real["offline_label"] == TRAJECTORY_LABEL
    assert bool(real["final_agrees_with_offline_label"])

    agreement = outputs[OFFLINE_AGREEMENT_OUTPUT].iloc[0]
    assert int(agreement["offline_labeled_windows"]) == 1
    assert agreement["agreement_fraction"] == pytest.approx(1.0)

    latency = outputs[LATENCY_OUTPUT]
    real_latency = latency[latency["window_set"].eq("real_windows")].iloc[0]
    assert int(real_latency["windows_with_causal_trajectory_claim"]) == 1
    assert real_latency["median_latency_s"] == pytest.approx(0.02)

    false_positive = outputs[FALSE_POSITIVE_OUTPUT]
    off_swr = false_positive[false_positive["window_set"].eq("matched_off_swr_windows")].iloc[0]
    assert int(off_swr["windows"]) == 2
    assert int(off_swr["any_prefix_trajectory_claims"]) == 1
    assert off_swr["any_prefix_false_positive_fraction"] == pytest.approx(0.5)

    gates = outputs[GATE_OUTPUT]
    assert bool(gates[gates["gate"].eq("overall")].iloc[0]["passed"])
    assert bool(gates[gates["gate"].eq("offline_agreement_evaluated")].iloc[0]["passed"])
    assert bool(gates[gates["gate"].eq("off_swr_false_positive_evaluated")].iloc[0]["passed"])

    for filename in outputs:
        path = tmp_path / filename
        assert path.exists()
        assert path.stat().st_size > 0


def _prefix_rows(
    session: str,
    event_index: int,
    window_role: str,
    null_index: int,
    prefix_time_bin_index: int,
    *,
    stationary: float,
    trajectory: float,
    start: float,
    prefix_end: float,
    off_swr: bool = False,
) -> list[dict[str, object]]:
    models = [
        ("sorted-spike-state-space-stationary", stationary, "nontrajectory"),
        ("sorted-spike-state-space-diffusion", trajectory - 3.0, "trajectory"),
        ("sorted-spike-state-space-fragmented", trajectory - 2.0, "trajectory"),
        ("sorted-spike-state-space-first-order-imm", trajectory, "trajectory"),
        ("sorted-spike-state-space-momentum-exact-sparse", trajectory - 1.0, "trajectory"),
    ]
    window_end = start + 0.03
    return [
        {
            "status": "success",
            "session": session,
            "event_index": event_index,
            "window_role": window_role,
            "event_window_variant": "core" if window_role == "real" else "matched_null",
            "null_index": null_index,
            "prefix_time_bin_index": prefix_time_bin_index,
            "prefix_end_s": prefix_end,
            "prefix_duration_s": prefix_end - start,
            "prefix_fraction": (prefix_end - start) / (window_end - start),
            "window_start_s": start,
            "window_end_s": window_end,
            "window_duration_s": window_end - start,
            "model": model,
            "requested_model": model,
            "model_family": family,
            "log_evidence": log_evidence,
            "n_time": prefix_time_bin_index,
            "n_spikes": 5 + prefix_time_bin_index,
            "off_swr": off_swr,
            "evidence_comparable": True,
        }
        for model, log_evidence, family in models
    ]


def _offline_rows(session: str, event_index: int, *, stationary: float, trajectory: float) -> list[dict[str, object]]:
    return _prefix_rows(
        session,
        event_index,
        "real",
        -1,
        99,
        stationary=stationary,
        trajectory=trajectory,
        start=1.0,
        prefix_end=1.03,
        off_swr=False,
    )

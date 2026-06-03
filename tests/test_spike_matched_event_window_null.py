from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.data import ReplaySession

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spike_matched_event_window_null import (  # noqa: E402
    aggregate_matched_null_scores,
    spike_matched_null_windows,
)


def test_spike_matched_null_windows_select_off_swr_spike_matched_window(tmp_path):
    session = _toy_session(tmp_path)

    nulls = spike_matched_null_windows(
        session,
        0,
        nulls_per_event=2,
        random_seed=1,
        candidate_step_s=0.1,
        spike_count_tolerance_fraction=0.1,
    )

    assert len(nulls) == 2
    assert nulls["window_duration_s"].tolist() == [pytest.approx(0.1), pytest.approx(0.1)]
    assert nulls["off_swr"].all()
    assert not ((nulls["window_start_s"] < 1.1) & (nulls["window_end_s"] > 1.0)).any()
    assert int(nulls.iloc[0]["null_n_spikes"]) == 2
    assert int(nulls.iloc[0]["real_n_spikes"]) == 2


def test_spike_matched_null_aggregate_writes_empirical_p_values_and_gates(tmp_path):
    scores = pd.DataFrame(
        [
            *_event_rows("Rat1/Open1", 0, "real", -1, stationary=0.0, trajectory=10.0),
            *_event_rows("Rat1/Open1", 0, "matched_null", 0, stationary=0.0, trajectory=2.0),
            *_event_rows("Rat1/Open1", 0, "matched_null", 1, stationary=0.0, trajectory=5.0),
            *_event_rows("Rat2/Open1", 1, "real", -1, stationary=0.0, trajectory=12.0),
            *_event_rows("Rat2/Open1", 1, "matched_null", 0, stationary=0.0, trajectory=1.0),
            *_event_rows("Rat2/Open1", 1, "matched_null", 1, stationary=0.0, trajectory=3.0),
        ]
    )
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path, index=False)
    out = tmp_path / "out"

    aggregate_matched_null_scores(str(score_path), out, bootstrap_samples=100)

    p_values = pd.read_csv(out / "matched_null_empirical_p_values.csv")
    gates = pd.read_csv(out / "matched_null_control_gate_summary.csv")
    rat_summary = pd.read_csv(out / "rat_matched_null_summary.csv")

    assert p_values["empirical_p_value"].tolist() == [1 / 3, 1 / 3]
    assert p_values["real_minus_median_null_family_margin"].tolist() == [6.5, 10.0]
    assert bool(gates.set_index("gate").loc["overall", "passed"])
    assert rat_summary["median_real_minus_median_null_family_margin"].tolist() == [6.5, 10.0]
    for expected in (
        "matched_null_event_model_evidence.csv",
        "matched_null_family_margin_decisions.csv",
        "matched_null_family_margin_summary.csv",
        "session_matched_null_summary.csv",
        "leave_one_rat_out_matched_null_summary.csv",
        "rat_bootstrap_matched_null_summary.csv",
    ):
        assert (out / expected).exists()


def test_spike_matched_null_workflow_exposes_control_outputs():
    workflow = Path(".github/workflows/spike-matched-event-window-null.yml").read_text(encoding="utf-8")

    assert "name: Spike-matched event-window null controls" in workflow
    assert "nulls_per_event:" in workflow
    assert 'default: "10"' in workflow
    assert "scripts/spike_matched_event_window_null.py score" in workflow
    assert "scripts/spike_matched_event_window_null.py aggregate" in workflow
    for expected in (
        "matched_null_event_model_evidence.csv",
        "matched_null_family_margin_decisions.csv",
        "matched_null_family_margin_summary.csv",
        "matched_null_empirical_p_values.csv",
        "session_matched_null_summary.csv",
        "rat_matched_null_summary.csv",
        "leave_one_rat_out_matched_null_summary.csv",
        "rat_bootstrap_matched_null_summary.csv",
        "matched_null_control_gate_summary.csv",
    ):
        assert expected in workflow


def _toy_session(tmp_path: Path) -> ReplaySession:
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=tmp_path,
        position=np.array([[0.0, 0.0, 0.0], [10.0, 1.0, 1.0]], dtype=float),
        spikes=np.array(
            [
                [1.01, 1],
                [1.02, 2],
                [2.01, 1],
                [2.02, 2],
                [3.01, 1],
                [5.01, 1],
                [5.02, 2],
            ],
            dtype=float,
        ),
        tetrode_cell_ids=np.empty((0, 2)),
        excitatory_neurons=np.array([1, 2], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.array([[1.0, 1.1, 1.05, 0.0, 0.0, 0.0]], dtype=float),
        run_times=np.array([[0.0, 6.0]], dtype=float),
        sleep_box_immobile_times=np.empty((0, 2)),
        sleep_times=np.empty((0, 2)),
        rem_times=np.empty((0, 2)),
        well_sequence=None,
        metadata={},
    )


def _event_rows(
    session: str,
    event_index: int,
    window_role: str,
    null_index: int,
    *,
    stationary: float,
    trajectory: float,
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
            "window_start_s": 1.0 + max(null_index, 0),
            "window_end_s": 1.1 + max(null_index, 0),
            "window_duration_s": 0.1,
            "model": model,
            "requested_model": model,
            "model_family": family,
            "log_evidence": log_evidence,
            "n_time": 25,
            "n_spikes": 10,
            "null_active_cell_count": 5,
            "real_n_spikes": 10,
            "n_spikes_delta": 0,
            "n_spikes_relative_delta": 0.0,
            "evidence_comparable": True,
        }
        for model, log_evidence, family in models
    ]

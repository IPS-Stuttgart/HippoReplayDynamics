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
    empirical_p_value,
    matched_null_empirical_p_values,
    matched_null_family_margin_decisions,
    rat_bootstrap_matched_null_summary,
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
    assert {"animal_speed_mean", "animal_speed_median", "animal_speed_max", "animal_x", "animal_y", "position_sample_count"}.issubset(
        nulls.columns
    )
    assert nulls["animal_speed_mean"].notna().all()
    assert (nulls["position_sample_count"] > 0).all()


def test_non_run_spike_matched_null_windows_cap_large_candidate_pool(tmp_path):
    session = _toy_session(tmp_path)

    nulls = spike_matched_null_windows(
        session,
        0,
        nulls_per_event=3,
        random_seed=1,
        spike_count_tolerance_fraction=0.1,
        restrict_to_run_times=False,
        max_candidate_windows=20,
    )

    assert len(nulls) == 3
    assert set(nulls["candidate_sampling_mode"]) == {"sampled"}
    assert set(nulls["candidate_pool_size"]) == {20}
    assert nulls["candidate_pool_exhaustive_size"].iloc[0] > 20
    assert not nulls["restrict_to_run_times"].any()
    assert not ((nulls["window_start_s"] < 1.1) & (nulls["window_end_s"] > 1.0)).any()


def test_explicit_candidate_step_keeps_non_run_null_pool_exhaustive(tmp_path):
    session = _toy_session(tmp_path)

    nulls = spike_matched_null_windows(
        session,
        0,
        nulls_per_event=2,
        random_seed=1,
        candidate_step_s=0.1,
        spike_count_tolerance_fraction=0.1,
        restrict_to_run_times=False,
        max_candidate_windows=5,
    )

    assert len(nulls) == 2
    assert set(nulls["candidate_sampling_mode"]) == {"exhaustive"}
    assert nulls["candidate_pool_size"].iloc[0] > 5


def test_non_run_spike_matched_nulls_prefer_position_covered_support(tmp_path):
    session = _toy_session(tmp_path)
    session.position = session.position[session.position[:, 0] <= 2.0]
    session.spikes = np.vstack(
        [
            session.spikes,
            np.array(
                [
                    [50.0, 1],
                    [50.1, 2],
                    [80.0, 1],
                ],
                dtype=float,
            ),
        ]
    )

    nulls = spike_matched_null_windows(
        session,
        0,
        nulls_per_event=3,
        random_seed=1,
        spike_count_tolerance_fraction=0.1,
        restrict_to_run_times=False,
        max_candidate_windows=50,
    )

    assert len(nulls) == 3
    assert float(nulls["window_start_s"].min()) >= 0.0
    assert float(nulls["window_end_s"].max()) <= 2.0
    assert nulls["animal_speed_mean"].notna().all()
    assert (nulls["position_sample_count"] > 0).all()


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


def test_spike_matched_null_aggregate_accepts_lightweight_required_models(tmp_path):
    stationary = "sorted-spike-state-space-stationary"
    trajectory = "sorted-spike-state-space-first-order-imm"
    scores = pd.DataFrame(
        [
            *_lightweight_event_rows(
                "Rat1/Open1",
                0,
                "real",
                -1,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=10.0,
            ),
            *_lightweight_event_rows(
                "Rat1/Open1",
                0,
                "matched_null",
                0,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=1.0,
            ),
        ]
    )
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path, index=False)
    out = tmp_path / "out"

    aggregate_matched_null_scores(
        str(score_path),
        out,
        comparison_scope="lightweight-first-order-imm-vs-stationary",
        required_models=(stationary, trajectory),
        bootstrap_samples=10,
    )

    decisions = pd.read_csv(out / "matched_null_family_margin_decisions.csv")
    p_values = pd.read_csv(out / "matched_null_empirical_p_values.csv")

    assert decisions["required_models_total"].tolist() == [2, 2]
    assert decisions["required_models_complete"].tolist() == [True, True]
    assert decisions["required_models_present"].tolist() == [2, 2]
    assert decisions["missing_required_models"].fillna("").tolist() == ["", ""]
    assert decisions["comparison_scope"].tolist() == [
        "lightweight-first-order-imm-vs-stationary",
        "lightweight-first-order-imm-vs-stationary",
    ]
    decisions_by_role = decisions.set_index("window_role")["margin_decision"].to_dict()
    assert decisions_by_role == {"real": "trajectory", "matched_null": "ambiguous"}
    assert p_values["real_trajectory_confident_claim"].tolist() == [True]
    assert (out / "lightweight_matched_null_control_gate_summary.csv").exists()


def test_spike_matched_null_full_core_scope_remains_strict_for_two_model_input(tmp_path):
    stationary = "sorted-spike-state-space-stationary"
    trajectory = "sorted-spike-state-space-first-order-imm"
    scores = pd.DataFrame(
        [
            *_lightweight_event_rows(
                "Rat1/Open1",
                0,
                "real",
                -1,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=10.0,
            ),
            *_lightweight_event_rows(
                "Rat1/Open1",
                0,
                "matched_null",
                0,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=1.0,
            ),
        ]
    )
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path, index=False)
    out = tmp_path / "out"

    aggregate_matched_null_scores(
        str(score_path),
        out,
        comparison_scope="full-core",
        bootstrap_samples=10,
    )

    decisions = pd.read_csv(out / "matched_null_family_margin_decisions.csv")

    assert decisions["required_models_total"].tolist() == [5, 5]
    assert decisions["required_models_present"].tolist() == [2, 2]
    assert decisions["required_models_complete"].tolist() == [False, False]
    assert decisions["margin_decision"].tolist() == ["incomplete_core", "incomplete_core"]


def test_matched_null_decisions_do_not_treat_string_false_as_comparable():
    stationary = "sorted-spike-state-space-stationary"
    trajectory = "sorted-spike-state-space-first-order-imm"
    scores = pd.DataFrame(
        [
            *_lightweight_event_rows(
                "Rat1/Open1",
                0,
                "real",
                -1,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=10.0,
            ),
        ]
    )
    scores["evidence_comparable"] = scores["evidence_comparable"].astype(object)
    scores.loc[scores["model"].eq(trajectory), "evidence_comparable"] = "False"

    decisions = matched_null_family_margin_decisions(
        scores,
        comparison_scope="lightweight-first-order-imm-vs-stationary",
        required_models=(stationary, trajectory),
        margin_threshold=5.5,
    )

    assert decisions["required_models_present"].tolist() == [1]
    assert decisions["required_models_complete"].tolist() == [False]
    assert decisions["missing_required_models"].tolist() == [trajectory]
    assert decisions["margin_decision"].tolist() == ["incomplete_core"]


def test_matched_null_empirical_p_values_parse_string_false_claim_flags():
    decisions = pd.DataFrame(
        [
            {
                "comparison_scope": "full_core",
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_role": "real",
                "null_index": -1,
                "trajectory_minus_nontrajectory_log_evidence": 10.0,
                "best_trajectory_log_evidence_per_spike": 1.0,
                "best_trajectory_log_evidence_per_time_bin": 1.0,
                "trajectory_confident_claim": "True",
                "nontrajectory_confident_claim": "False",
            },
            {
                "comparison_scope": "full_core",
                "session": "Rat1/Open1",
                "event_index": 0,
                "window_role": "matched_null",
                "null_index": 0,
                "trajectory_minus_nontrajectory_log_evidence": 2.0,
                "best_trajectory_log_evidence_per_spike": 0.5,
                "best_trajectory_log_evidence_per_time_bin": 0.5,
                "trajectory_confident_claim": "False",
                "nontrajectory_confident_claim": "False",
            },
        ]
    )

    p_values = matched_null_empirical_p_values(decisions)

    assert bool(p_values.loc[0, "real_trajectory_confident_claim"]) is True
    assert bool(p_values.loc[0, "real_nontrajectory_confident_claim"]) is False


def test_rat_bootstrap_matched_null_summary_keeps_comparison_scopes_separate():
    p_values = pd.DataFrame(
        {
            "comparison_scope": ["scope-a", "scope-a", "scope-b", "scope-b"],
            "rat": ["Rat1", "Rat2", "Rat1", "Rat2"],
            "session": ["Rat1/Open1", "Rat2/Open1", "Rat1/Open1", "Rat2/Open1"],
            "event_index": [0, 1, 0, 1],
            "real_minus_median_null_family_margin": [2.0, 4.0, -10.0, -8.0],
            "empirical_p_value": [0.1, 0.1, 0.9, 0.9],
            "matched_null_windows": [10, 10, 10, 10],
            "real_trajectory_confident_claim": [True, True, False, False],
            "real_nontrajectory_confident_claim": [False, False, True, True],
        }
    )

    summary = rat_bootstrap_matched_null_summary(p_values, random_seed=1, n_bootstrap=50)
    by_scope = summary.set_index("comparison_scope")

    assert summary["comparison_scope"].tolist() == ["scope-a", "scope-b"]
    assert float(by_scope.loc["scope-a", "median_delta_median"]) > 0.0
    assert float(by_scope.loc["scope-b", "median_delta_median"]) < 0.0


def test_empirical_p_value_uses_plus_one_resolution_for_k50():
    nulls = np.arange(50, dtype=float)

    assert empirical_p_value(100.0, nulls) == pytest.approx(1 / 51)
    assert empirical_p_value(10.0, np.array([11.0, *np.zeros(49)])) == pytest.approx(2 / 51)


def test_targeted_session_diagnostics_identify_rat2_open2_caveat(tmp_path):
    stationary = "sorted-spike-state-space-stationary"
    trajectory = "sorted-spike-state-space-first-order-imm"
    scores = pd.DataFrame(
        [
            *_lightweight_event_rows(
                "Rat2/Open1",
                0,
                "real",
                -1,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=12.0,
            ),
            *_lightweight_event_rows(
                "Rat2/Open1",
                0,
                "matched_null",
                0,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=2.0,
            ),
            *_lightweight_event_rows(
                "Rat2/Open2",
                1,
                "real",
                -1,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=1.0,
            ),
            *_lightweight_event_rows(
                "Rat2/Open2",
                1,
                "matched_null",
                0,
                stationary_model=stationary,
                trajectory_model=trajectory,
                stationary_score=0.0,
                trajectory_score=8.0,
            ),
        ]
    )
    score_path = tmp_path / "scores.csv"
    scores.to_csv(score_path, index=False)
    out = tmp_path / "out"

    aggregate_matched_null_scores(
        str(score_path),
        out,
        comparison_scope="lightweight-first-order-imm-vs-stationary",
        bootstrap_samples=10,
    )

    session_diagnostics = pd.read_csv(out / "targeted_matched_null_session_diagnostics.csv")
    by_session = session_diagnostics.set_index("session")

    assert by_session.loc["Rat2/Open1", "median_real_minus_median_null_family_margin"] > 0
    assert by_session.loc["Rat2/Open2", "median_real_minus_median_null_family_margin"] < 0


def test_spike_matched_null_workflow_exposes_control_outputs():
    workflow = Path(".github/workflows/spike-matched-event-window-null.yml").read_text(encoding="utf-8")

    assert "name: Spike-matched event-window null controls" in workflow
    assert "nulls_per_event:" in workflow
    assert 'default: "10"' in workflow
    assert "null_count:" in workflow
    assert "events_per_session_for_k50:" in workflow
    assert "comparison_scope:" in workflow
    assert "session_filter:" in workflow
    assert "event_index_filter:" in workflow
    assert "null_models:" in workflow
    assert 'models="${NULL_MODELS:-${MODELS}}"' in workflow
    assert 'nulls_per_event="${NULL_COUNT:-${NULLS_PER_EVENT}}"' in workflow
    assert "--comparison-scope" in workflow
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
        "targeted_matched_null_session_diagnostics.csv",
        "targeted_matched_null_event_diagnostics.csv",
        "lightweight_matched_null_control_gate_summary.csv",
    ):
        assert expected in workflow


def _toy_session(tmp_path: Path) -> ReplaySession:
    position_times = np.arange(0.0, 6.05, 0.05)
    return ReplaySession(
        rat="Rat1",
        name="Open1",
        path=tmp_path,
        position=np.column_stack([position_times, position_times * 10.0, np.zeros_like(position_times)]),
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


def _lightweight_event_rows(
    session: str,
    event_index: int,
    window_role: str,
    null_index: int,
    *,
    stationary_model: str,
    trajectory_model: str,
    stationary_score: float,
    trajectory_score: float,
) -> list[dict[str, object]]:
    rows = [
        (stationary_model, stationary_score, "nontrajectory"),
        (trajectory_model, trajectory_score, "trajectory"),
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
        for model, log_evidence, family in rows
    ]

from __future__ import annotations

from argparse import Namespace

import pandas as pd

from scripts.run_exact_sparse_momentum_gate import (
    DIFFUSION_MODEL,
    EXACT_SPARSE_MOMENTUM_MODEL,
    FIRST_ORDER_IMM_MODEL,
    aggregate_gate_results,
    build_event_summary,
    build_simulation_command,
    parse_sessions,
    safe_session_id,
)


def test_parse_sessions_and_safe_session_id():
    assert parse_sessions("Rat1/Open1, Rat2/Open2") == ["Rat1/Open1", "Rat2/Open2"]
    assert safe_session_id("Rat1/Open1") == "Rat1_Open1"


def test_build_simulation_command_uses_exact_sparse_recovery_knobs(tmp_path):
    args = Namespace(
        python_executable="python",
        dataset_root="data/DataSetFromPfeifferFoster",
        events="run",
        max_template_events=5,
        events_per_model=1,
        true_models="diffusion momentum",
        models=f"{DIFFUSION_MODEL} {EXACT_SPARSE_MOMENTUM_MODEL}",
        time_bin_ms=3.0,
        spike_rate_scale=1.0,
        bin_size_cm=6.0,
        smoothing_sigma_bins=2.0,
        min_speed_cm_s=5.0,
        state_space_diffusion_sigma_cm_sqrt_s=85.0,
        state_space_momentum_sigma_cm_sqrt_s=85.0,
        state_space_momentum_initial_sigma_cm_sqrt_s=85.0,
        state_space_momentum_velocity_decay_tau_s=0.060,
        state_space_max_step_sigma=4.0,
        state_space_imm_mode_stickiness=0.95,
        state_space_momentum_candidate_top_k=128,
        state_space_momentum_predicted_candidate_top_k=8,
        continue_on_error=True,
    )

    command = build_simulation_command(
        args,
        session="Rat1/Open1",
        session_output=tmp_path / "Rat1_Open1",
        random_seed=7,
    )

    assert "simulate-recovery" in command
    assert "--true-state-space-momentum-velocity-decay-tau-s" in command
    assert "0.06" in command
    assert EXACT_SPARSE_MOMENTUM_MODEL in " ".join(command)
    assert "--continue-on-error" in command


def test_aggregate_gate_results_passes_on_exact_sparse_momentum(tmp_path):
    _write_fake_scores(
        tmp_path,
        "Rat1_Open1",
        [
            _fake_event("Rat1/Open1", 0, "diffusion", DIFFUSION_MODEL),
            _fake_event("Rat1/Open1", 1, "momentum", EXACT_SPARSE_MOMENTUM_MODEL),
        ],
    )
    _write_fake_scores(
        tmp_path,
        "Rat2_Open1",
        [
            _fake_event("Rat2/Open1", 0, "diffusion", DIFFUSION_MODEL),
            _fake_event("Rat2/Open1", 1, "momentum", EXACT_SPARSE_MOMENTUM_MODEL),
        ],
    )

    status = aggregate_gate_results(
        tmp_path,
        min_momentum_recovery=0.70,
        min_diffusion_recovery=0.70,
    )

    assert status["gate_passed"] is True
    assert status["momentum_exact_surrogate_recovered_events"] == 2
    assert status["diffusion_recovered_events"] == 2
    assert (tmp_path / "exact_sparse_momentum_gate.md").exists()


def test_aggregate_gate_results_fails_when_momentum_not_recovered(tmp_path):
    _write_fake_scores(
        tmp_path,
        "Rat1_Open1",
        [
            _fake_event("Rat1/Open1", 0, "diffusion", DIFFUSION_MODEL),
            _fake_event("Rat1/Open1", 1, "momentum", DIFFUSION_MODEL),
        ],
    )

    status = aggregate_gate_results(
        tmp_path,
        min_momentum_recovery=0.70,
        min_diffusion_recovery=0.70,
    )

    assert status["gate_passed"] is False
    assert status["momentum_exact_surrogate_recovery_accuracy"] == 0.0


def test_build_event_summary_reports_exact_sparse_margins():
    frame = pd.DataFrame(
        _fake_event("Rat1/Open1", 0, "momentum", EXACT_SPARSE_MOMENTUM_MODEL)
    )

    summary = build_event_summary(frame)

    assert summary.loc[0, "best_model"] == EXACT_SPARSE_MOMENTUM_MODEL
    assert bool(summary.loc[0, "exact_surrogate_recovered"]) is True
    assert summary.loc[0, "exact_sparse_minus_diffusion"] > 0.0


def test_build_event_summary_excludes_string_false_comparable_rows():
    frame = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "expected_exact_surrogate_model": EXACT_SPARSE_MOMENTUM_MODEL,
                "model": DIFFUSION_MODEL,
                "requested_model": DIFFUSION_MODEL,
                "log_evidence": 1.0,
                "evidence_comparable": "True",
                "n_time": 5,
                "n_spikes": 3,
            },
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 0,
                "true_model": "momentum",
                "expected_model": "sorted-spike-state-space-momentum",
                "expected_exact_surrogate_model": EXACT_SPARSE_MOMENTUM_MODEL,
                "model": EXACT_SPARSE_MOMENTUM_MODEL,
                "requested_model": EXACT_SPARSE_MOMENTUM_MODEL,
                "log_evidence": 100.0,
                "evidence_comparable": "False",
                "n_time": 5,
                "n_spikes": 3,
            },
        ]
    )

    summary = build_event_summary(frame)

    assert summary.loc[0, "best_model"] == DIFFUSION_MODEL
    assert bool(summary.loc[0, "exact_surrogate_recovered"]) is False


def _write_fake_scores(tmp_path, session_dir: str, events: list[list[dict[str, object]]]) -> None:
    output = tmp_path / session_dir
    output.mkdir(parents=True)
    rows = [row for event in events for row in event]
    pd.DataFrame(rows).to_csv(output / "simulation_recovery_event_scores.csv", index=False)


def _fake_event(
    session: str,
    event_index: int,
    true_model: str,
    best_model: str,
) -> list[dict[str, object]]:
    expected_model = (
        DIFFUSION_MODEL
        if true_model == "diffusion"
        else "sorted-spike-state-space-momentum"
    )
    surrogate_model = (
        EXACT_SPARSE_MOMENTUM_MODEL if true_model == "momentum" else expected_model
    )
    evidences = {
        DIFFUSION_MODEL: 12.0,
        EXACT_SPARSE_MOMENTUM_MODEL: 11.0,
        "sorted-spike-state-space-fragmented": 8.0,
        FIRST_ORDER_IMM_MODEL: 10.0,
    }
    evidences[best_model] = 20.0
    rows = []
    for model, evidence in evidences.items():
        rows.append(
            {
                "status": "success",
                "session": session,
                "event_index": event_index,
                "true_model": true_model,
                "expected_model": expected_model,
                "expected_exact_surrogate_model": surrogate_model,
                "model": model,
                "requested_model": model,
                "log_evidence": evidence,
                "evidence_comparable": True,
                "best_model": best_model,
                "recovered_expected_model": best_model == expected_model,
                "exact_surrogate_recovered_expected_model": best_model == surrogate_model,
                "n_time": 5,
                "n_spikes": 3,
                "runtime_s": 0.1,
            }
        )
    return rows

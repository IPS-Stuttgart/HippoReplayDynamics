from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.build_trajectory_family_paper_pack import build_trajectory_family_paper_pack


STATIONARY = "sorted-spike-state-space-stationary"
DIFFUSION = "sorted-spike-state-space-diffusion"
FRAGMENTED = "sorted-spike-state-space-fragmented"
FIRST_ORDER_IMM = "sorted-spike-state-space-first-order-imm"
MOMENTUM = "sorted-spike-state-space-momentum-exact-sparse"


def _write_full_core_artifact(path, rows):
    path.mkdir()
    pd.DataFrame(rows).to_csv(path / "all_sessions_event_model_evidence.csv", index=False)


def _event_rows(session: str, event_index: int, values: dict[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, log_evidence in values.items():
        rows.append(
            {
                "status": "success",
                "session": session,
                "event_index": event_index,
                "model": model,
                "model_family": "nontrajectory" if model == STATIONARY else "trajectory",
                "log_evidence": log_evidence,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
                "time_bin_s": 0.004,
                "spike_rate_scale": 2.0,
                "emission_likelihood_temperature": 0.3,
                "diagnostic_state_space_diffusion_sigma_cm_sqrt_s": 60.0,
                "diagnostic_state_space_momentum_sigma_cm_sqrt_s": 50.0,
                "diagnostic_state_space_momentum_initial_sigma_cm_sqrt_s": 45.0,
                "diagnostic_state_space_momentum_velocity_decay": 0.93,
            }
        )
    return rows


def _synthetic_full_core_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        _event_rows(
            "Rat1/Open1",
            0,
            {
                STATIONARY: 0.0,
                DIFFUSION: 15.0,
                FRAGMENTED: 12.0,
                FIRST_ORDER_IMM: 30.0,
                MOMENTUM: 14.0,
            },
        )
    )
    rows.extend(
        _event_rows(
            "Rat1/Open1",
            1,
            {
                STATIONARY: 5.0,
                DIFFUSION: 18.0,
                FRAGMENTED: 12.0,
                FIRST_ORDER_IMM: 20.0,
                MOMENTUM: 26.0,
            },
        )
    )
    rows.extend(
        _event_rows(
            "Rat2/Open2",
            0,
            {
                STATIONARY: 1.0,
                DIFFUSION: 20.0,
                FRAGMENTED: 18.0,
                FIRST_ORDER_IMM: 28.0,
                MOMENTUM: 18.0,
            },
        )
    )
    rows.extend(
        _event_rows(
            "Rat2/Open2",
            1,
            {
                STATIONARY: 4.0,
                DIFFUSION: 19.0,
                FRAGMENTED: 14.0,
                FIRST_ORDER_IMM: 31.0,
                MOMENTUM: 17.0,
            },
        )
    )
    return rows


def _write_control_artifact(path, gate_name: str, summary_name: str):
    path.mkdir()
    pd.DataFrame(
        [
            {"gate": "overall_control_gate", "passed": True, "observed": "1", "criterion": "> 0", "details": ""},
            {"gate": gate_name, "passed": True, "observed": "1", "criterion": "> 0", "details": ""},
        ]
    ).to_csv(path / gate_name, index=False)
    pd.DataFrame([{"events": 4, "mean_margin": 10.0, "median_margin": 9.0}]).to_csv(path / summary_name, index=False)


def test_build_trajectory_family_paper_pack_writes_all_outputs(tmp_path):
    full_core = tmp_path / "model-evidence-all-sessions-27011374643"
    _write_full_core_artifact(full_core, _synthetic_full_core_rows())
    wrong_map = tmp_path / "wrong-map-control-27000000001"
    event_window = tmp_path / "event-window-26884355750"
    cell_split = tmp_path / "cell-split-26965909403"
    matched_null_k10 = tmp_path / "matched-null-26886723196"
    matched_null_k50 = tmp_path / "matched-null-27060148887"
    _write_control_artifact(wrong_map, "wrong_map_control_gate_summary.csv", "wrong_map_family_evidence_attenuation_summary.csv")
    _write_control_artifact(event_window, "event_window_control_gate_summary.csv", "event_window_family_margin_summary.csv")
    _write_control_artifact(cell_split, "cell_split_control_gate_summary.csv", "cell_split_heldout_family_margin_summary.csv")
    _write_control_artifact(matched_null_k10, "matched_null_control_gate_summary.csv", "matched_null_family_margin_summary.csv")
    _write_control_artifact(
        matched_null_k50,
        "lightweight_matched_null_control_gate_summary.csv",
        "matched_null_family_margin_summary.csv",
    )

    output = tmp_path / "paper-pack"
    manifest = build_trajectory_family_paper_pack(
        full_core_artifact=full_core,
        wrong_map_artifact=wrong_map,
        event_window_artifact=event_window,
        cell_split_artifact=cell_split,
        matched_null_k10_artifact=matched_null_k10,
        matched_null_k50_artifact=matched_null_k50,
        output=output,
        confidence_threshold=5.5,
        n_bootstrap=25,
        random_seed=1,
        code_commit="test-commit",
        require_controls=True,
    )

    expected = {
        "paper_claim_manifest.json",
        "main_trajectory_family_summary.csv",
        "rat_trajectory_family_summary.csv",
        "leave_one_rat_out_trajectory_family_summary.csv",
        "bootstrap_trajectory_family_summary.csv",
        "exact_core_model_winner_summary.csv",
        "paired_momentum_diffusion_summary.csv",
        "control_stack_summary.csv",
        "matched_null_summary.csv",
        "cell_split_summary.csv",
        "event_window_summary.csv",
        "wrong_map_summary.csv",
        "figure_source_manifest.csv",
        "trajectory_family_paper_claim_summary.md",
    }
    assert expected.issubset({path.name for path in output.iterdir()})

    manifest_from_disk = json.loads((output / "paper_claim_manifest.json").read_text(encoding="utf-8"))
    assert manifest_from_disk["code_commit"] == "test-commit"
    assert manifest_from_disk["artifact_run_ids"]["full_core_model_evidence"] == "27011374643"
    assert manifest_from_disk["confidence_threshold"] == 5.5
    assert manifest_from_disk["event_count"] == 4
    assert "calibrated_row_parameters" in manifest_from_disk
    assert manifest["primary_claim"].startswith("Exact trajectory-family")

    main = pd.read_csv(output / "main_trajectory_family_summary.csv")
    assert int(main.loc[0, "events"]) == 4
    assert int(main.loc[0, "trajectory_confident_claims"]) == 4
    assert int(main.loc[0, "nontrajectory_confident_claims"]) == 0

    exact_core = pd.read_csv(output / "exact_core_model_winner_summary.csv")
    first_order = exact_core[exact_core["model"] == FIRST_ORDER_IMM].iloc[0]
    assert int(first_order["raw_best_events"]) == 3

    paired = pd.read_csv(output / "paired_momentum_diffusion_summary.csv")
    assert int(paired.loc[0, "positive_raw_wins"]) == 1

    control_stack = pd.read_csv(output / "control_stack_summary.csv")
    assert set(control_stack["status"]) == {"ok"}
    assert int(control_stack["known_tables_found"].sum()) >= 10

    matched_null = pd.read_csv(output / "matched_null_summary.csv")
    assert set(matched_null["artifact_label"]) == {"matched_null_k10_control", "matched_null_k50_control"}

    claim_text = (output / "trajectory_family_paper_claim_summary.md").read_text(encoding="utf-8")
    assert "Exact-sparse momentum is a recovered paired momentum-vs-diffusion signal" in claim_text


def test_missing_optional_controls_are_recorded(tmp_path):
    full_core = tmp_path / "model-evidence-all-sessions-27011374643"
    _write_full_core_artifact(full_core, _synthetic_full_core_rows())

    output = tmp_path / "paper-pack"
    build_trajectory_family_paper_pack(
        full_core_artifact=full_core,
        output=output,
        n_bootstrap=5,
        code_commit="test-commit",
    )

    control_stack = pd.read_csv(output / "control_stack_summary.csv")
    assert set(control_stack["status"]) == {"missing_path"}
    assert (output / "matched_null_summary.csv").exists()


def test_require_controls_rejects_missing_control_paths(tmp_path):
    full_core = tmp_path / "model-evidence-all-sessions-27011374643"
    _write_full_core_artifact(full_core, _synthetic_full_core_rows())

    with pytest.raises(FileNotFoundError, match="Required control artifact"):
        build_trajectory_family_paper_pack(
            full_core_artifact=full_core,
            output=tmp_path / "paper-pack",
            n_bootstrap=5,
            code_commit="test-commit",
            require_controls=True,
        )

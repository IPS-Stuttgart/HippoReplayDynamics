from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = repo_root / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    module_path = scripts_path / "report_olafsdottir_1d_sleeppost_evidence.py"
    spec = importlib.util.spec_from_file_location("report_olafsdottir_1d_sleeppost_evidence", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reporter_summarizes_existing_smoke_without_rescoring(tmp_path: Path) -> None:
    module = _load_module()
    evidence_dir = tmp_path / "evidence"
    report_dir = tmp_path / "report"
    evidence_dir.mkdir()
    decoder_csv = tmp_path / "decoder.csv"
    pilot_csv = tmp_path / "pilot.csv"

    _write_smoke_outputs(evidence_dir, decoder_csv=decoder_csv, pilot_csv=pilot_csv)

    tables = module.run_report(evidence_dir=evidence_dir, output_dir=report_dir, write_figures=False)

    assert (report_dir / module.QUALITY_OUTPUT).is_file()
    assert (report_dir / module.MODEL_RANK_OUTPUT).is_file()
    assert (report_dir / module.TRAJECTORY_MARGIN_OUTPUT).is_file()
    assert (report_dir / module.IMM_FRAGMENTED_AUDIT_OUTPUT).is_file()
    assert (report_dir / module.PAIR_DEBUG_OUTPUT).is_file()
    assert (report_dir / module.ANIMAL_DEBUG_OUTPUT).is_file()
    assert (report_dir / module.REPORT_OUTPUT).is_file()

    quality = tables["quality"]
    assert len(quality) == 2
    assert "posterior_mean_error_cm_median" in quality.columns
    assert "candidate_tier" in quality.columns
    assert quality["candidate_tier"].tolist() == ["moderate", "weak"]
    assert tables["classification"]["technical_classification"] == "technical-pass"
    assert tables["classification"]["biological_classification"] == "biological-ambiguous"

    imm_audit = tables["imm_fragmented_audit"]
    assert imm_audit["imm_confident_win_at_5p5"].tolist() == [True, False]
    assert imm_audit["imm_fragmented_ambiguous"].tolist() == [False, True]
    report = (report_dir / module.REPORT_OUTPUT).read_text(encoding="utf-8")
    assert "technical-pass" in report
    assert "biological-ambiguous" in report
    assert "does not rescore events" in report


def _write_smoke_outputs(evidence_dir: Path, *, decoder_csv: Path, pilot_csv: Path) -> None:
    events = [
        {
            "animal": "R1",
            "date": "2020-01-01",
            "track1_session": "R1_track1",
            "sleeppost_session": "R1_sleepPOST",
            "pilot_tier": "pilot_20_decoder_available_debug",
            "decoder_filter": "scoring_available",
            "event_index": 0,
            "event_id": 10,
            "start_time_s": 1.0,
            "end_time_s": 1.08,
            "duration_ms": 80.0,
            "n_spikes": 20,
            "n_active_units": 7,
            "mean_speed_cm_s": 0.2,
            "decoder_qc_passed": True,
            "linearization_qc_passed": True,
            "best_model": "first_order_imm",
            "runner_up_model": "fragmented",
            "best_minus_runner_up_log_evidence": 6.0,
            "logZ_stationary": 0.0,
            "logZ_diffusion": 1.0,
            "logZ_fragmented": 2.0,
            "logZ_first_order_imm": 8.0,
            "delta_best_trajectory_minus_stationary": 8.0,
            "delta_imm_minus_fragmented": 6.0,
            "trajectory_family_claim": "trajectory_confident",
            "imm_clean_vs_fragmented_claim": True,
            "fragmented_claim": False,
            "brownian_diffusion_claim": False,
            "ambiguous_claim": False,
        },
        {
            "animal": "R2",
            "date": "2020-01-02",
            "track1_session": "R2_track1",
            "sleeppost_session": "R2_sleepPOST",
            "pilot_tier": "pilot_20_decoder_available_debug",
            "decoder_filter": "scoring_available",
            "event_index": 1,
            "event_id": 20,
            "start_time_s": 2.0,
            "end_time_s": 2.08,
            "duration_ms": 80.0,
            "n_spikes": 8,
            "n_active_units": 4,
            "mean_speed_cm_s": 0.4,
            "decoder_qc_passed": True,
            "linearization_qc_passed": True,
            "best_model": "first_order_imm",
            "runner_up_model": "fragmented",
            "best_minus_runner_up_log_evidence": 2.0,
            "logZ_stationary": 0.0,
            "logZ_diffusion": -0.2,
            "logZ_fragmented": 1.0,
            "logZ_first_order_imm": 3.0,
            "delta_best_trajectory_minus_stationary": 3.0,
            "delta_imm_minus_fragmented": 2.0,
            "trajectory_family_claim": "ambiguous",
            "imm_clean_vs_fragmented_claim": False,
            "fragmented_claim": False,
            "brownian_diffusion_claim": False,
            "ambiguous_claim": True,
        },
    ]
    pd.DataFrame(events).to_csv(evidence_dir / "olafsdottir_1d_sleep_model_claim_decisions.csv", index=False)

    evidence_rows = []
    for event in events:
        for model, logz in {
            "stationary": event["logZ_stationary"],
            "diffusion": event["logZ_diffusion"],
            "fragmented": event["logZ_fragmented"],
            "first_order_imm": event["logZ_first_order_imm"],
        }.items():
            evidence_rows.append(
                {
                    **{key: event[key] for key in event if key not in {"best_model", "runner_up_model"}},
                    "model": model,
                    "model_family": "trajectory" if model != "stationary" else "stationary",
                    "log_evidence": logz,
                    "status": "success",
                    "failure_reason": "",
                    "runtime_s": 0.01,
                }
            )
    pd.DataFrame(evidence_rows).to_csv(evidence_dir / "olafsdottir_1d_sleep_event_model_evidence.csv", index=False)
    pd.DataFrame(
        [
            {"gate": "overall", "passed": True, "status": "pass", "value": "passed=10/10", "note": "all readiness gates pass"}
        ]
    ).to_csv(evidence_dir / "olafsdottir_1d_sleep_gate_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "animal": "R1",
                "date": "2020-01-01",
                "track1_session": "R1_track1",
                "sleeppost_session": "R1_sleepPOST",
                "decoder_status": "fail",
                "decoder_qc_paper_ready": False,
                "decoder_qc_scoring_available": True,
                "encoding_units_passing_qc": 8,
                "posterior_mean_error_cm_median": 42.0,
                "map_error_cm_median": 58.0,
                "posterior_coverage_fraction": 0.9,
            },
            {
                "animal": "R2",
                "date": "2020-01-02",
                "track1_session": "R2_track1",
                "sleeppost_session": "R2_sleepPOST",
                "decoder_status": "fail",
                "decoder_qc_paper_ready": False,
                "decoder_qc_scoring_available": True,
                "encoding_units_passing_qc": 6,
                "posterior_mean_error_cm_median": 60.0,
                "map_error_cm_median": 80.0,
                "posterior_coverage_fraction": 0.82,
            },
        ]
    ).to_csv(decoder_csv, index=False)
    pd.DataFrame(
        [
            {
                "selection_tier": "pilot_20_decoder_available_debug",
                "animal": "R1",
                "date": "2020-01-01",
                "track1_session": "R1_track1",
                "sleeppost_session": "R1_sleepPOST",
                "event_id": 10,
                "candidate_tier": "moderate",
                "mean_mua_rate_hz": 120.0,
                "peak_mua_rate_hz": 200.0,
            },
            {
                "selection_tier": "all_immobile_qc_valid",
                "animal": "R1",
                "date": "2020-01-01",
                "track1_session": "R1_track1",
                "sleeppost_session": "R1_sleepPOST",
                "event_id": 10,
                "candidate_tier": "wrong_tier_duplicate",
                "mean_mua_rate_hz": 1.0,
                "peak_mua_rate_hz": 2.0,
            },
            {
                "selection_tier": "pilot_20_decoder_available_debug",
                "animal": "R2",
                "date": "2020-01-02",
                "track1_session": "R2_track1",
                "sleeppost_session": "R2_sleepPOST",
                "event_id": 20,
                "candidate_tier": "weak",
                "mean_mua_rate_hz": 60.0,
                "peak_mua_rate_hz": 100.0,
            },
            {
                "selection_tier": "pilot_50_decoder_available_debug",
                "animal": "R2",
                "date": "2020-01-02",
                "track1_session": "R2_track1",
                "sleeppost_session": "R2_sleepPOST",
                "event_id": 20,
                "candidate_tier": "wrong_tier_duplicate",
                "mean_mua_rate_hz": 1.0,
                "peak_mua_rate_hz": 2.0,
            },
        ]
    ).to_csv(pilot_csv, index=False)
    (evidence_dir / "olafsdottir_1d_sleep_manifest.json").write_text(
        json.dumps(
            {
                "pilot_tier": "pilot_20_decoder_available_debug",
                "code_commit": "test-commit",
                "margin_threshold": 5.5,
                "decoder_qc": str(decoder_csv),
                "pilot_selection": str(pilot_csv),
            }
        ),
        encoding="utf-8",
    )

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = repo_root / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    module_path = scripts_path / "decide_olafsdottir_1d_next_pilot.py"
    spec = importlib.util.spec_from_file_location("decide_olafsdottir_1d_next_pilot", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_next_pilot_decision_recommends_high_information_when_event_strength_tracks_margin(tmp_path: Path) -> None:
    module = _load_module()
    report_dir = tmp_path / "report"
    output_dir = tmp_path / "decision"
    report_dir.mkdir()
    _write_quality_table(
        report_dir,
        [
            _event("R1", "2020-01-01", 1, n_spikes=5, n_active=3, duration=20, traj=-1.0, imm=0.5),
            _event("R1", "2020-01-01", 2, n_spikes=10, n_active=4, duration=30, traj=0.0, imm=1.0),
            _event("R2", "2020-01-02", 3, n_spikes=20, n_active=6, duration=40, traj=3.0, imm=2.0),
            _event("R2", "2020-01-02", 4, n_spikes=40, n_active=8, duration=50, traj=7.0, imm=6.0),
        ],
    )

    tables = module.run_next_pilot_decision(report_dir=report_dir, output_dir=output_dir)

    assert (output_dir / module.CORRELATION_OUTPUT).is_file()
    assert (output_dir / module.EVENT_STRENGTH_OUTPUT).is_file()
    assert (output_dir / module.DECODER_QUALITY_OUTPUT).is_file()
    assert (output_dir / module.PAIR_DECISION_OUTPUT).is_file()
    assert (output_dir / module.RECOMMENDATION_OUTPUT).is_file()
    assert (output_dir / module.SUMMARY_OUTPUT).is_file()
    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["recommendation"] == "run_high_information_pilot20_debug"
    assert recommendation["best_event_strength_predictor"] in set(module.EVENT_STRENGTH_PREDICTORS)
    summary = (output_dir / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")
    assert "does not rescore events" in summary


def test_next_pilot_decision_recommends_pair_targeted_when_signal_is_localized(tmp_path: Path) -> None:
    module = _load_module()
    report_dir = tmp_path / "report"
    output_dir = tmp_path / "decision"
    report_dir.mkdir()
    _write_quality_table(
        report_dir,
        [
            _event("R1", "2020-01-01", 1, n_spikes=20, n_active=8, duration=30, traj=8.0, imm=6.0),
            _event("R1", "2020-01-01", 2, n_spikes=18, n_active=7, duration=30, traj=-0.5, imm=6.5),
            _event("R2", "2020-01-02", 3, n_spikes=40, n_active=10, duration=30, traj=-0.8, imm=1.0),
            _event("R2", "2020-01-02", 4, n_spikes=30, n_active=9, duration=30, traj=-0.6, imm=2.0),
            _event("R3", "2020-01-03", 5, n_spikes=50, n_active=11, duration=30, traj=-0.9, imm=1.5),
            _event("R3", "2020-01-03", 6, n_spikes=45, n_active=10, duration=30, traj=-0.7, imm=2.5),
        ],
    )

    tables = module.run_next_pilot_decision(report_dir=report_dir, output_dir=output_dir)

    recommendation = tables["recommendation"].iloc[0]
    assert recommendation["recommendation"] == "run_pair_targeted_debug"
    assert bool(recommendation["localized_signal"])
    assert "R1" in recommendation["primary_reason"]


def _write_quality_table(report_dir: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(report_dir / "olafsdottir_1d_sleep_debug_quality_table.csv", index=False)


def _event(
    animal: str,
    date: str,
    event_id: int,
    *,
    n_spikes: int,
    n_active: int,
    duration: float,
    traj: float,
    imm: float,
) -> dict[str, object]:
    return {
        "animal": animal,
        "date": date,
        "track1_session": f"{animal}_track1",
        "sleeppost_session": f"{animal}_sleepPOST",
        "pilot_tier": "pilot_20_decoder_available_debug",
        "decoder_filter": "scoring_available",
        "event_id": event_id,
        "duration_ms": duration,
        "n_spikes": n_spikes,
        "n_active_units": n_active,
        "mean_speed_cm_s": 0.2,
        "mean_mua_rate_hz": n_spikes / max(duration / 1000.0, 1e-9),
        "peak_mua_rate_hz": n_spikes / max(duration / 1000.0, 1e-9) * 1.5,
        "candidate_tier": "extreme",
        "best_model": "first_order_imm" if traj > 0 else "stationary",
        "runner_up_model": "fragmented",
        "best_minus_runner_up_log_evidence": abs(traj),
        "delta_best_trajectory_minus_stationary": traj,
        "delta_imm_minus_fragmented": imm,
        "trajectory_family_claim": "trajectory_confident" if traj >= 5.5 else "ambiguous",
        "imm_clean_vs_fragmented_claim": bool(imm >= 5.5),
        "fragmented_claim": False,
        "brownian_diffusion_claim": False,
        "ambiguous_claim": bool(traj < 5.5 and imm < 5.5),
        "decoder_qc_passed": True,
        "linearization_qc_passed": True,
        "decoder_status": "fail",
        "decoder_qc_paper_ready": False,
        "decoder_qc_scoring_available": True,
        "encoding_units_passing_qc": 10,
        "posterior_mean_error_cm_median": 100.0,
        "map_error_cm_median": 120.0,
        "posterior_coverage_fraction": 1.0,
        "logZ_stationary": 0.0,
        "logZ_diffusion": traj / 2.0,
        "logZ_fragmented": imm,
        "logZ_first_order_imm": imm + 1.0,
    }

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
    module_path = scripts_path / "compare_olafsdottir_1d_pilot_tiers.py"
    spec = importlib.util.spec_from_file_location("compare_olafsdottir_1d_pilot_tiers", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pilot_tier_comparison_writes_raw_and_normalized_tables(tmp_path: Path) -> None:
    module = _load_module()
    balanced = _write_report_dir(
        tmp_path / "balanced",
        label="pilot_20_decoder_available_debug",
        trajectory_margins=[-1.0, 1.0],
        imm_margins=[1.0, 2.0],
        durations_ms=[50.0, 100.0],
        spikes=[10, 20],
    )
    high_info = _write_report_dir(
        tmp_path / "high-info",
        label="pilot_20_high_information_debug",
        trajectory_margins=[2.0, 4.0],
        imm_margins=[4.0, 7.0],
        durations_ms=[100.0, 100.0],
        spikes=[20, 20],
    )
    holdout = _write_report_dir(
        tmp_path / "holdout",
        label="pilot_20_high_information_holdout_debug",
        trajectory_margins=[8.0, 10.0],
        imm_margins=[6.0, 8.0],
        durations_ms=[100.0, 100.0],
        spikes=[20, 20],
    )
    out = tmp_path / "comparison"

    tables = module.run_pilot_tier_comparison(
        report_dirs=[balanced, high_info, holdout],
        labels=["balanced_debug", "high_information_debug", "high_information_holdout19_debug"],
        output_dir=out,
    )

    assert (out / module.COMPARISON_OUTPUT).is_file()
    assert (out / module.NORMALIZED_OUTPUT).is_file()
    assert (out / module.BY_ANIMAL_OUTPUT).is_file()
    assert (out / module.BY_PAIR_OUTPUT).is_file()
    assert (out / module.DECISION_OUTPUT).is_file()
    assert (out / module.REPORT_OUTPUT).is_file()
    comparison = tables["comparison"].set_index("tier_label")
    assert comparison.loc["balanced_debug", "events"] == 2
    assert comparison.loc["high_information_holdout19_debug", "trajectory_confident_events"] == 2
    assert comparison.loc["high_information_holdout19_debug", "median_trajectory_minus_stationary_per_second"] == 90.0
    assert comparison.loc["high_information_holdout19_debug", "median_trajectory_minus_stationary_per_spike"] == 0.45
    decision = tables["decision"].iloc[0]
    assert decision["recommendation"] == "define_frozen_high_information_confirmation_tier"
    report = (out / module.REPORT_OUTPUT).read_text(encoding="utf-8")
    assert "does not select events and does not rescore evidence" in report
    assert "biological_claim_assessed: false" in report


def test_pilot_tier_comparison_detects_imm_only_improvement(tmp_path: Path) -> None:
    module = _load_module()
    balanced = _write_report_dir(
        tmp_path / "balanced",
        label="pilot_20_decoder_available_debug",
        trajectory_margins=[-0.2, -1.0],
        imm_margins=[1.0, 2.0],
    )
    high_info = _write_report_dir(
        tmp_path / "high-info",
        label="pilot_20_high_information_debug",
        trajectory_margins=[-0.5, -1.2],
        imm_margins=[6.0, 7.0],
    )
    holdout = _write_report_dir(
        tmp_path / "holdout",
        label="pilot_20_high_information_holdout_debug",
        trajectory_margins=[-0.8, -1.2],
        imm_margins=[6.0, 8.0],
    )

    tables = module.run_pilot_tier_comparison(
        report_dirs=[balanced, high_info, holdout],
        labels=["balanced_debug", "high_information_debug", "high_information_holdout19_debug"],
        output_dir=tmp_path / "comparison",
    )

    decision = tables["decision"].iloc[0]
    assert decision["recommendation"] == "continue_imm_fragmented_taxonomy_audit_only"
    assert "IMM-vs-fragmented" in decision["reason"]


def test_labelled_report_dir_cli_parsing(tmp_path: Path) -> None:
    module = _load_module()
    labels, paths = module.parse_labelled_report_dirs(
        [
            f"balanced_debug={tmp_path / 'a'}",
            f"high_information_debug={tmp_path / 'b'}",
        ]
    )
    assert labels == ["balanced_debug", "high_information_debug"]
    assert paths == [tmp_path / "a", tmp_path / "b"]


def _write_report_dir(
    root: Path,
    *,
    label: str,
    trajectory_margins: list[float],
    imm_margins: list[float],
    durations_ms: list[float] | None = None,
    spikes: list[int] | None = None,
) -> Path:
    root.mkdir()
    durations_ms = durations_ms or [100.0] * len(trajectory_margins)
    spikes = spikes or [20] * len(trajectory_margins)
    rows = []
    for index, (trajectory, imm) in enumerate(zip(trajectory_margins, imm_margins, strict=True)):
        rows.append(
            {
                "animal": f"R{index + 1}",
                "date": f"2020-01-0{index + 1}",
                "track1_session": f"R{index + 1}_track1",
                "sleeppost_session": f"R{index + 1}_sleepPOST",
                "pilot_tier": label,
                "decoder_filter": "scoring_available",
                "event_id": index,
                "duration_ms": durations_ms[index],
                "n_spikes": spikes[index],
                "n_active_units": 5 + index,
                "mean_speed_cm_s": 0.5,
                "mean_mua_rate_hz": 100.0,
                "peak_mua_rate_hz": 200.0,
                "candidate_tier": "strong",
                "best_model": "first_order_imm",
                "runner_up_model": "fragmented",
                "best_minus_runner_up_log_evidence": abs(imm),
                "delta_best_trajectory_minus_stationary": trajectory,
                "delta_imm_minus_fragmented": imm,
                "trajectory_family_claim": "trajectory_confident" if trajectory >= 5.5 else "ambiguous",
                "imm_clean_vs_fragmented_claim": imm >= 5.5,
                "fragmented_claim": False,
                "brownian_diffusion_claim": False,
                "ambiguous_claim": trajectory < 5.5,
                "decoder_qc_passed": True,
                "linearization_qc_passed": True,
                "decoder_status": "fail",
                "decoder_qc_paper_ready": False,
                "decoder_qc_scoring_available": True,
                "encoding_units_passing_qc": 8,
                "posterior_mean_error_cm_median": 50.0,
                "map_error_cm_median": 75.0,
                "posterior_coverage_fraction": 0.8,
                "logZ_stationary": 0.0,
                "logZ_diffusion": 0.5,
                "logZ_fragmented": 1.0,
                "logZ_first_order_imm": 1.0 + imm,
            }
        )
    pd.DataFrame(rows).to_csv(root / "olafsdottir_1d_sleep_debug_quality_table.csv", index=False)
    (root / "olafsdottir_1d_sleep_debug_report_manifest.json").write_text(
        '{"source_pilot_tier": "' + label + '"}\n',
        encoding="utf-8",
    )
    return root

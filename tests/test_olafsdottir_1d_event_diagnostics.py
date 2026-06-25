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
    module_path = scripts_path / "plot_olafsdottir_1d_event_diagnostics.py"
    spec = importlib.util.spec_from_file_location("plot_olafsdottir_1d_event_diagnostics", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_diagnostics_selects_positive_negative_and_writes_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    pairs_csv = tmp_path / "pairs.csv"
    linearization_qc = tmp_path / "linearization.csv"
    evidence_a = tmp_path / "balanced"
    evidence_b = tmp_path / "high_information"
    output_dir = tmp_path / "event-panels"
    evidence_a.mkdir()
    evidence_b.mkdir()

    _write_pairs(pairs_csv)
    linearization_qc.write_text("animal,date,track_session,sleeppost_session,linearization_status\n", encoding="utf-8")
    _write_evidence_dir(evidence_a, tier_label="balanced_debug", rows=_decision_rows(offset=0))
    _write_evidence_dir(evidence_b, tier_label="high_information_debug", rows=_decision_rows(offset=100))

    manifest = module.run_event_diagnostics(
        dataset_root=tmp_path / "dataset",
        pairs_csv=pairs_csv,
        linearization_qc=linearization_qc,
        evidence_dirs=[f"balanced_debug={evidence_a}", f"high_information_debug={evidence_b}"],
        output_dir=output_dir,
        max_events=8,
        skip_data_panels=True,
    )

    assert (output_dir / module.OUTPUT_MANIFEST).is_file()
    assert (output_dir / module.OUTPUT_RUN_MANIFEST).is_file()
    assert not manifest.empty
    assert "top_trajectory_minus_stationary" in ";".join(manifest["selection_reasons"].astype(str))
    assert "top_imm_minus_fragmented" in ";".join(manifest["selection_reasons"].astype(str))
    assert "most_stationary_favored" in ";".join(manifest["selection_reasons"].astype(str))
    assert any(manifest["selection_reasons"].astype(str).str.contains("R2192_positive"))
    assert any(manifest["source_tier_labels"].astype(str).str.contains("high_information_debug"))

    first = manifest.iloc[0]
    for column in ["raster_path", "posterior_heatmap_path", "model_evidence_bars_path", "summary_path"]:
        assert Path(first[column]).is_file()
    assert "trajectory-family claim" in Path(first["summary_path"]).read_text(encoding="utf-8")


def test_parse_labeled_path_and_safe_token() -> None:
    module = _load_module()
    assert module.parse_labeled_path("my tier=/tmp/example") == ("my_tier", "/tmp/example")
    assert module.safe_token("R2192/2014 09 17") == "R2192_2014_09_17"


def _write_pairs(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "animal": "R2192",
                "date": "2014-09-17",
                "track_session": "R2192_track1",
                "sleepPOST_session": "R2192_sleepPOST",
                "hippocampal_tetrodes": "9,10",
                "usable_pair": True,
            },
            {
                "animal": "R2335",
                "date": "2015-10-26",
                "track_session": "R2335_track1",
                "sleepPOST_session": "R2335_sleepPOST",
                "hippocampal_tetrodes": "9,10",
                "usable_pair": True,
            },
        ]
    ).to_csv(path, index=False)


def _write_evidence_dir(path: Path, *, tier_label: str, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path / "olafsdottir_1d_sleep_model_claim_decisions.csv", index=False)
    (path / "olafsdottir_1d_sleep_manifest.json").write_text(
        json.dumps(
            {
                "pilot_tier": tier_label,
                "time_bin_s": 0.02,
                "position_bin_size_cm": 5.0,
                "min_unit_spikes": 5,
                "min_encoding_units": 1,
                "smoothing_bins": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _decision_rows(offset: int) -> list[dict[str, object]]:
    return [
        _row("R2192", "2014-09-17", "R2192_track1", "R2192_sleepPOST", offset + 1, traj=9.0, imm=6.5, best="diffusion"),
        _row("R2192", "2014-09-17", "R2192_track1", "R2192_sleepPOST", offset + 2, traj=1.0, imm=9.0, best="stationary"),
        _row("R2335", "2015-10-26", "R2335_track1", "R2335_sleepPOST", offset + 3, traj=7.0, imm=6.0, best="first_order_imm"),
        _row("R2335", "2015-10-26", "R2335_track1", "R2335_sleepPOST", offset + 4, traj=-8.0, imm=1.0, best="stationary"),
    ]


def _row(
    animal: str,
    date: str,
    track: str,
    sleep: str,
    event_id: int,
    *,
    traj: float,
    imm: float,
    best: str,
) -> dict[str, object]:
    stationary = -100.0 - event_id
    fragmented = stationary + max(traj - 2.0, -1.0)
    first_order_imm = fragmented + imm
    diffusion = stationary + traj if best == "diffusion" else stationary + max(traj - 1.0, -2.0)
    logz = {
        "stationary": stationary,
        "diffusion": diffusion,
        "fragmented": fragmented,
        "first_order_imm": first_order_imm,
    }
    winner = max(logz, key=logz.get)
    sorted_values = sorted(logz.values(), reverse=True)
    return {
        "animal": animal,
        "date": date,
        "track1_session": track,
        "sleeppost_session": sleep,
        "pilot_tier": "synthetic_tier",
        "decoder_filter": "scoring_available",
        "event_index": event_id,
        "event_id": event_id,
        "start_time_s": float(event_id),
        "end_time_s": float(event_id) + 0.06,
        "duration_ms": 60.0,
        "n_spikes": 50 + event_id,
        "n_active_units": 8,
        "mean_speed_cm_s": 0.1,
        "decoder_qc_passed": True,
        "linearization_qc_passed": True,
        "best_model": winner,
        "runner_up_model": "stationary",
        "best_minus_runner_up_log_evidence": sorted_values[0] - sorted_values[1],
        "logZ_stationary": logz["stationary"],
        "logZ_diffusion": logz["diffusion"],
        "logZ_fragmented": logz["fragmented"],
        "logZ_first_order_imm": logz["first_order_imm"],
        "delta_best_trajectory_minus_stationary": traj,
        "delta_imm_minus_fragmented": imm,
        "trajectory_family_claim": "trajectory_confident" if traj >= 5.5 else "ambiguous",
        "imm_clean_vs_fragmented_claim": imm >= 5.5,
        "fragmented_claim": False,
        "brownian_diffusion_claim": False,
        "ambiguous_claim": traj < 5.5 and imm < 5.5,
    }

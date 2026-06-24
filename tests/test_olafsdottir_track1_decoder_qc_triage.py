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
    module_path = scripts_path / "triage_olafsdottir_track1_decoder_qc.py"
    spec = importlib.util.spec_from_file_location("triage_olafsdottir_track1_decoder_qc", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_decoder_qc_triage_writes_failure_and_threshold_tables(tmp_path: Path) -> None:
    module = _load_module()
    decoder_csv = tmp_path / "decoder.csv"
    unit_csv = tmp_path / "units.csv"
    pairs_csv = tmp_path / "pairs.csv"
    _decoder_qc().to_csv(decoder_csv, index=False)
    _unit_qc().to_csv(unit_csv, index=False)
    _pairs().to_csv(pairs_csv, index=False)

    tables = module.run_decoder_qc_triage(
        decoder_qc=decoder_csv,
        unit_qc=unit_csv,
        pairs_csv=pairs_csv,
        output_dir=tmp_path / "triage",
    )

    out = tmp_path / "triage"
    assert (out / module.FAILURE_OUTPUT).is_file()
    assert (out / module.SENSITIVITY_OUTPUT).is_file()
    assert (out / module.METRIC_OUTPUT).is_file()
    assert (out / module.PAIR_AUDIT_OUTPUT).is_file()
    assert (out / module.GATE_OUTPUT).is_file()
    assert (out / module.SUMMARY_OUTPUT).is_file()

    pair_audit = tables["pair_audit"]
    failure = tables["failure_summary"].set_index("failure_reason")
    sensitivity = tables["threshold_sensitivity"]
    gates = tables["gates"].set_index("gate")

    assert len(pair_audit) == 4
    assert set(module.PAIR_AUDIT_COLUMNS).issubset(pair_audit.columns)
    assert bool(pair_audit.loc[pair_audit["animal"].eq("R2"), "failed_posterior_mean_error"].iloc[0])
    assert bool(pair_audit.loc[pair_audit["animal"].eq("R3"), "failed_map_error"].iloc[0])
    assert bool(pair_audit.loc[pair_audit["animal"].eq("R3"), "failed_posterior_coverage"].iloc[0])
    assert bool(pair_audit.loc[pair_audit["animal"].eq("R4"), "failed_schema"].iloc[0])
    assert failure.loc["poor_posterior_mean_error", "failed_pairs"] == 1
    assert failure.loc["poor_map_error", "failed_pairs"] == 1
    assert failure.loc["low_posterior_coverage", "failed_pairs"] == 1
    assert failure.loc["schema_status_mismatch", "failed_pairs"] == 1
    assert len(sensitivity) == 4 * 5 * 5 * 3
    assert sensitivity["decoder_pass_pairs"].max() == 3
    assert sensitivity["animals_retained"].max() == 3
    assert bool(gates.loc["decoder_qc_loaded", "passed"])
    assert bool(gates.loc["threshold_sensitivity_grid_complete", "passed"])
    assert not bool(gates.loc["status_consistent_with_default_thresholds", "passed"])
    assert "diagnostic-only" in (out / module.SUMMARY_OUTPUT).read_text(encoding="utf-8")


def test_decoder_qc_triage_rejects_missing_required_columns(tmp_path: Path) -> None:
    module = _load_module()
    decoder_csv = tmp_path / "decoder.csv"
    pd.DataFrame([{"animal": "R1"}]).to_csv(decoder_csv, index=False)

    try:
        module.load_decoder_qc(decoder_csv)
    except ValueError as exc:
        assert "posterior_mean_error_cm_median" in str(exc)
    else:
        raise AssertionError("load_decoder_qc should reject incomplete decoder QC tables")


def _decoder_qc() -> pd.DataFrame:
    rows = []
    specs = [
        ("R1", "pass", 6, 30.0, 40.0, 0.90),
        ("R2", "fail", 7, 80.0, 40.0, 0.90),
        ("R3", "fail", 8, 30.0, 80.0, 0.40),
        ("R4", "fail", 9, 30.0, 40.0, 0.90),
    ]
    for animal, status, units, posterior, map_error, coverage in specs:
        rows.append(
            {
                "animal": animal,
                "date": "2020-01-01",
                "track1_session": f"{animal}_track1",
                "sleeppost_session": f"{animal}_sleepPOST",
                "decoder_status": status,
                "n_units_track1": units + 1,
                "encoding_units_passing_qc": units,
                "posterior_mean_error_cm_median": posterior,
                "map_error_cm_median": map_error,
                "posterior_coverage_fraction": coverage,
            }
        )
    return pd.DataFrame(rows)


def _unit_qc() -> pd.DataFrame:
    rows = []
    for animal in ["R1", "R2", "R3", "R4"]:
        for unit in range(3):
            rows.append(
                {
                    "animal": animal,
                    "date": "2020-01-01",
                    "track1_session": f"{animal}_track1",
                    "unit_id": unit,
                    "n_spikes_track1": 20 + unit,
                    "mean_rate_hz": 0.5,
                    "peak_rate_hz": 2.0,
                    "spatial_information": 0.2,
                    "place_field_width_cm": 30.0,
                    "unit_qc_passed": True,
                }
            )
    return pd.DataFrame(rows)


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "animal": animal,
                "date": "2020-01-01",
                "track_session": f"{animal}_track1",
                "sleepPOST_session": f"{animal}_sleepPOST",
                "usable_pair": True,
            }
            for animal in ["R1", "R2", "R3", "R4"]
        ]
    )

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def load_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "report_hc11_native_ripple_geometry_sensitivity.py"
    spec = importlib.util.spec_from_file_location("report_hc11_native_ripple_geometry_sensitivity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = load_module()


def write_run(directory: Path, *, clean_event: bool) -> None:
    directory.mkdir()
    rows = []
    for event_id, geometry in enumerate(("linear", "circular")):
        clean = clean_event and event_id == 0
        rows.append(
            {
                "animal": "RatA",
                "session": f"RatA_{geometry}",
                "geometry": geometry,
                "event_id": event_id,
                "best_model": "first_order_imm" if clean else "fragmented",
                "trajectory_confident_claim": clean,
                "stationary_confident_claim": False,
                "imm_confident_over_fragmented": clean,
                "fragmented_confident_over_imm": False,
                "delta_trajectory_minus_stationary": 8.0 if clean else 1.0,
                "delta_imm_minus_fragmented": 7.0 if clean else -1.0,
                "n_time_bins": 5,
                "n_spikes": 10,
            }
        )
    pd.DataFrame(rows).to_csv(directory / "hc11_native_ripple_model_claim_decisions.csv", index=False)
    pd.DataFrame(
        {
            "animal": ["RatA", "RatA"],
            "session": ["RatA_linear", "RatA_circular"],
            "geometry": ["linear", "circular"],
            "event_id": [0, 1],
        }
    ).to_csv(directory / "hc11_native_ripple_event_selection.csv", index=False)
    pd.DataFrame([{"gate": "overall_technical", "passed": True, "detail": "ok"}]).to_csv(
        directory / "hc11_native_ripple_gate_summary.csv",
        index=False,
    )


def test_report_keeps_strict_clean_imm_intersection_and_stopgate(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "report"
    write_run(first, clean_event=True)
    write_run(second, clean_event=False)

    result = report.run([("first", first), ("second", second)], output)

    first_summary = result["summary"].set_index("run").loc["first"]
    assert first_summary["strict_clean_imm_count"] == 1
    assert result["strict"]["event_id"].tolist() == [0]
    gates = result["gates"].set_index("gate")
    assert bool(gates.loc["all_input_runs_technical_pass", "passed"])
    assert not bool(gates.loc["external_clean_imm_replication_supported", "passed"])
    assert (output / report.REPORT_OUTPUT).exists()

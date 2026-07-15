from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.audit_pfeiffer_imm_gate_convergence import (
    CORE_CONTROLS,
    build_event_table,
    correlation_tables,
    partial_spearman,
)


def _fixtures(n: int = 12):
    rows4 = []
    rows2 = []
    rows3 = []
    rows_map = []
    for index in range(n):
        rat = f"Rat{index % 4 + 1}"
        session = f"{rat}/Open{index % 2 + 1}"
        quality = float(index + 1)
        switching = float((index * 7) % n + 1)
        outcome = 2.0 * quality + 3.0 * switching
        rows4.append(
            {
                "scope": "all_events",
                "session": session,
                "rat": rat,
                "event_index": index,
                "heldout_delta_event_median": outcome,
                "median_train_cell_count": 20 + quality,
                "median_test_spikes": 5 + quality,
                "median_n_time": 10 + index,
                "median_train_imm_posterior_entropy": 5.0 - quality / 10.0,
            }
        )
        rows2.append(
            {
                "session": session,
                "rat": rat,
                "event_index": index,
                "real_order_advantage": switching,
                "wrong_order_advantage": switching / 2.0,
                "order_by_map_interaction": switching / 2.0,
            }
        )
        rows3.append(
            {
                "session": session,
                "rat": rat,
                "event_index": index,
                "mean_nonstationary_mode_probability": switching / n,
                "fraction_time_map_nonstationary": switching / n,
                "expected_switch_count": switching,
                "map_mode_switch_count": int(switching),
                "posterior_expected_path_length_cm": switching * 20.0,
                "posterior_net_displacement_cm": switching * 5.0,
            }
        )
        rows_map.append(
            {
                "session": session,
                "rat": rat,
                "event_index": index,
                "real_minus_null_median_delta_imm_minus_fragmented": switching / 4.0,
                "real_minus_null_median_mean_nonstationary_mode_probability": switching / 100.0,
                "real_minus_null_median_posterior_expected_path_length_cm": switching * 2.0,
                "real_minus_null_median_posterior_net_displacement_cm": switching,
            }
        )
    return tuple(pd.DataFrame(rows) for rows in (rows4, rows2, rows3, rows_map))


def test_build_event_table_joins_scopes_and_transforms_controls() -> None:
    gate4, gate2, gate3, map_frame = _fixtures()
    extra = gate4.iloc[[0]].copy()
    extra["scope"] = "frozen_clean_imm_sensitivity"
    events = build_event_table(pd.concat([gate4, extra]), gate2, gate3, map_frame)

    assert len(events) == 12
    assert events["gate2_available"].all()
    assert events["gate3_available"].all()
    assert np.isfinite(events["log1p_train_cell_count"]).all()
    assert np.isfinite(events["gate1b_margin_map_excess"]).all()


def test_partial_spearman_retains_switching_signal_after_quality_controls() -> None:
    events = build_event_table(*_fixtures(40))
    estimate, _, n_events, n_rats = partial_spearman(
        events,
        "gate3_expected_switch_count",
        "heldout_delta_imm_minus_fragmented",
        CORE_CONTROLS,
    )
    assert n_events == 40
    assert n_rats == 4
    assert estimate > 0.8


def test_correlation_tables_report_both_control_sets() -> None:
    events = build_event_table(*_fixtures(40))
    raw, partial, by_rat = correlation_tables(events, bootstrap_replicates=30, seed=2)

    assert len(raw) == 8
    assert len(partial) == 16
    assert set(partial["control_set"]) == {"decodability_core", "decodability_plus_duration"}
    assert by_rat["rat"].nunique() == 4
    target = partial[
        partial["analysis_id"].eq("heldout_vs_expected_switch_count")
        & partial["control_set"].eq("decodability_core")
    ].iloc[0]
    assert target["partial_spearman_rho"] > 0.8


def test_cli_writes_complete_non_rescoring_pack(tmp_path: Path) -> None:
    gate4, gate2, gate3, map_frame = _fixtures(20)
    inputs = {}
    for name, frame in {
        "gate4": gate4,
        "gate2": gate2,
        "gate3": gate3,
        "map": map_frame,
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        inputs[name] = path
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_pfeiffer_imm_gate_convergence.py",
            "--gate4-event-medians",
            str(inputs["gate4"]),
            "--gate2-factorial-decisions",
            str(inputs["gate2"]),
            "--gate3-posterior-content",
            str(inputs["gate3"]),
            "--map-permutation-decisions",
            str(inputs["map"]),
            "--output-dir",
            str(output),
            "--bootstrap-replicates",
            "20",
        ],
        check=True,
    )
    expected = {
        "pfeiffer_imm_gate_convergence_event_table.csv",
        "pfeiffer_imm_gate_convergence_correlations.csv",
        "pfeiffer_imm_gate_convergence_partial_correlations.csv",
        "pfeiffer_imm_gate_convergence_by_rat.csv",
        "pfeiffer_imm_gate_convergence_gate_summary.csv",
        "pfeiffer_imm_gate_convergence_manifest.json",
        "pfeiffer_imm_gate_convergence_report.md",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = (output / "pfeiffer_imm_gate_convergence_manifest.json").read_text()
    assert "exploratory_not_confirmatory" in manifest

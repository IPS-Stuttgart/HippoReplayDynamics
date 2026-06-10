from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from aggregate_event_window_sensitivity import _bool_column as event_window_bool_column  # noqa: E402
from audit_sweep_completeness import _bool_series as sweep_bool_series  # noqa: E402
from cell_split_heldout_control import _bool_column as cell_split_bool_column  # noqa: E402
from compare_wrong_map_evidence_controls import _bool_column as wrong_map_bool_column  # noqa: E402
from hipporeplayimm.advanced_result_diagnostics import _bool_column as advanced_bool_column  # noqa: E402
from hipporeplayimm.result_improvement_extensions import _bool_series as extension_bool_series  # noqa: E402
from hipporeplayimm.result_quality_audit import _bool_series as quality_audit_bool_series  # noqa: E402
from run_exact_sparse_momentum_gate import _bool_series as exact_gate_bool_series  # noqa: E402
from select_state_space_parameters import _bool_series as selector_bool_series  # noqa: E402
from topological_replay_comparator import _bool_column as topology_bool_column  # noqa: E402
from triage_momentum_recovery import _bool_series as triage_bool_series  # noqa: E402


def test_numeric_string_bool_helpers_parse_csv_round_trips() -> None:
    values = pd.Series(["1.0", "0.0", "True", "False", 1.0, 0.0])
    expected = [True, False, True, False, True, False]
    frame = pd.DataFrame({"flag": values})

    column_helpers = (
        advanced_bool_column,
        cell_split_bool_column,
        event_window_bool_column,
        topology_bool_column,
        wrong_map_bool_column,
    )
    for helper in column_helpers:
        assert helper(frame, "flag").tolist() == expected

    series_helpers = (
        exact_gate_bool_series,
        extension_bool_series,
        quality_audit_bool_series,
        selector_bool_series,
        sweep_bool_series,
        triage_bool_series,
    )
    for helper in series_helpers:
        assert helper(values).tolist() == expected

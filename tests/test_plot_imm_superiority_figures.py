from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_imm_superiority_figures.py"
_SPEC = importlib.util.spec_from_file_location("plot_imm_superiority_figures", _SCRIPT)
assert _SPEC is not None
plot_imm_superiority_figures = importlib.util.module_from_spec(_SPEC)
sys.modules["plot_imm_superiority_figures"] = plot_imm_superiority_figures
assert _SPEC.loader is not None
_SPEC.loader.exec_module(plot_imm_superiority_figures)


def _write_audit_table(root: Path, event_indices: list[object]) -> None:
    pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * len(event_indices),
            "event_index": event_indices,
        }
    ).to_csv(root / "trajectory_taxonomy_event_table.csv", index=False)


def test_read_audit_table_rejects_fractional_event_indices(tmp_path: Path) -> None:
    _write_audit_table(tmp_path, [7, 7.5])

    with pytest.raises(ValueError, match="event_index.*integer-valued"):
        plot_imm_superiority_figures._read_audit_table(tmp_path)


def test_read_audit_table_accepts_integral_float_event_indices(tmp_path: Path) -> None:
    _write_audit_table(tmp_path, [7.0, 8.0])

    table = plot_imm_superiority_figures._read_audit_table(tmp_path)

    assert table["event_index"].tolist() == [7, 8]
    assert pd.api.types.is_integer_dtype(table["event_index"])

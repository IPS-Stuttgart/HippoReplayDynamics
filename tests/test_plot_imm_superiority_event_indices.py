from __future__ import annotations

import pandas as pd
import pytest

from scripts.plot_imm_superiority_figures import (
    _coerce_event_indices,
    _read_audit_table,
)


def test_coerce_event_indices_preserves_adjacent_large_decimal_ids():
    values = pd.Series(["9007199254740992", "9007199254740993"], dtype="string")

    parsed = _coerce_event_indices(values)

    assert parsed.tolist() == [9007199254740992, 9007199254740993]
    assert parsed.nunique() == 2


def test_coerce_event_indices_rejects_fractional_and_unsafe_float_ids():
    with pytest.raises(ValueError, match="integer-valued"):
        _coerce_event_indices(pd.Series(["12.5"], dtype="string"))

    with pytest.raises(ValueError, match="integer-valued"):
        _coerce_event_indices(pd.Series([float(2**53)]))


def test_read_audit_table_preserves_large_event_ids_from_csv(tmp_path):
    path = tmp_path / "trajectory_taxonomy_event_table.csv"
    path.write_text(
        "session,event_index\n"
        "Rat1/Open1,9007199254740992\n"
        "Rat1/Open1,9007199254740993\n",
        encoding="utf-8",
    )

    table = _read_audit_table(tmp_path)

    assert table["event_index"].tolist() == [9007199254740992, 9007199254740993]
    assert table["event_index"].nunique() == 2

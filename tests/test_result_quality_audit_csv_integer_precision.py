from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_model_evidence_results import _optional_frame, _read_audit_csv  # noqa: E402


def _write_scope_csv(path: Path) -> tuple[int, int]:
    lower = 2**53
    upper = lower + 1
    path.write_text(
        "session,event_index,model,log_evidence,null_index,null_random_seed,benchmark_cell_split_index\n"
        f"Rat1/Open1,1,imm,5.0,{lower},{lower},{lower}\n"
        f"Rat1/Open1,1,momentum,4.0,{upper},{upper},{upper}\n"
        "Rat1/Open1,2,imm,3.0,,,\n",
        encoding="utf-8",
    )
    return lower, upper


def test_read_audit_csv_preserves_nullable_large_scope_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    lower, upper = _write_scope_csv(path)

    frame = _read_audit_csv(path)

    for column in (
        "null_index",
        "null_random_seed",
        "benchmark_cell_split_index",
    ):
        assert frame[column].iloc[0] == lower
        assert frame[column].iloc[1] == upper
        assert frame[column].iloc[0] != frame[column].iloc[1]
        assert pd.isna(frame[column].iloc[2])


def test_optional_audit_frame_uses_exact_integer_reader(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    lower, upper = _write_scope_csv(path)

    frame = _optional_frame(str(path))

    assert frame is not None
    assert frame["null_index"].iloc[:2].tolist() == [lower, upper]


def test_read_audit_csv_rejects_fractional_scope_identifier(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "session,event_index,model,log_evidence,null_index\n"
        "Rat1/Open1,1,imm,5.0,1.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="null_index must contain integer-valued identifiers"):
        _read_audit_csv(path)

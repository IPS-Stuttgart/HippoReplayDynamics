from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from deduplicate_off_swr_candidates import (  # noqa: E402
    _read_required_csv,
    build_one_per_source_group_decisions,
)


def _large_decimal_validation() -> pd.DataFrame:
    first = 2**53
    second = first + 1
    return pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": f"{first}.0",
                "null_index": "0",
                "trajectory_minus_nontrajectory_log_evidence": 10.0,
                "trajectory_family_margin": 9.0,
                "window_start_s": 1.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": f"{second}.0",
                "null_index": "0",
                "trajectory_minus_nontrajectory_log_evidence": 11.0,
                "trajectory_family_margin": 10.0,
                "window_start_s": 2.0,
            },
        ]
    )


def test_off_swr_dedup_preserves_adjacent_decimal_ids_above_binary64_precision() -> None:
    decisions = build_one_per_source_group_decisions(_large_decimal_validation())
    strongest = decisions[
        decisions["selection_rule"].eq("strongest_exact_margin")
    ]

    expected = {2**53, 2**53 + 1}
    assert set(strongest["event_index"].tolist()) == expected
    assert set(strongest["source_event_index"].tolist()) == expected
    assert set(strongest["source_event_group_id"].tolist()) == {
        f"Rat1/Open1|event={value}" for value in expected
    }


def test_off_swr_csv_reader_preserves_decimal_identifier_text(tmp_path: Path) -> None:
    path = tmp_path / "validation.csv"
    _large_decimal_validation().to_csv(path, index=False)

    loaded = _read_required_csv(path)
    decisions = build_one_per_source_group_decisions(loaded)
    strongest = decisions[
        decisions["selection_rule"].eq("strongest_exact_margin")
    ]

    assert loaded["event_index"].tolist() == [
        f"{2**53}.0",
        f"{2**53 + 1}.0",
    ]
    assert set(strongest["event_index"].tolist()) == {2**53, 2**53 + 1}


def test_off_swr_dedup_rejects_ambiguous_large_float_identifier() -> None:
    validation = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": float(2**53),
                "null_index": 0,
            }
        ]
    )

    with pytest.raises(ValueError, match=r"floating-point values with magnitude >= 2\*\*53"):
        build_one_per_source_group_decisions(validation)

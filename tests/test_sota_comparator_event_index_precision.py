from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_sota_comparator_pack import (
    REQUIRED_EXACT_CORE_MODELS,
    build_sota_comparator_event_table,
    read_event_model_evidence,
)


def _event_rows(event_index: object) -> list[dict[str, object]]:
    return [
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": event_index,
            "model": model,
            "log_evidence": float(index),
            "evidence_comparable": True,
        }
        for index, model in enumerate(REQUIRED_EXACT_CORE_MODELS)
    ]


def test_sota_comparator_preserves_adjacent_decimal_event_ids_above_2_to_53(
    tmp_path: Path,
) -> None:
    lower = 2**53
    upper = lower + 1
    path = tmp_path / "event_model_evidence.csv"
    rows = [
        *_event_rows(f"{lower}.0"),
        *_event_rows(f"{upper}.0"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

    evidence = read_event_model_evidence(path)
    event_table = build_sota_comparator_event_table(evidence)

    assert sorted(evidence["event_index"].unique().tolist()) == [lower, upper]
    assert event_table["event_index"].tolist() == [lower, upper]
    assert event_table["exact_core_complete"].tolist() == [True, True]


def test_sota_comparator_rejects_fractional_event_ids() -> None:
    evidence = pd.DataFrame(_event_rows("1.5"))

    with pytest.raises(ValueError, match="integer-valued"):
        build_sota_comparator_event_table(evidence)


def test_sota_comparator_rejects_unsafe_preparsed_large_float_event_ids() -> None:
    evidence = pd.DataFrame(_event_rows(float(2**53)))

    with pytest.raises(ValueError, match=r"at or above 2\*\*53 is unsafe"):
        build_sota_comparator_event_table(evidence)


def test_sota_comparator_rejects_boolean_event_ids() -> None:
    evidence = pd.DataFrame(_event_rows(True))

    with pytest.raises(ValueError, match="not booleans"):
        build_sota_comparator_event_table(evidence)

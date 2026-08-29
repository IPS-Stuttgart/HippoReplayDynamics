import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_swr_off_swr_dynamics import (  # noqa: E402
    DEFAULT_FIRST_ORDER_IMM_MODEL,
    DEFAULT_MARGIN_POSITIVE_MODEL,
    DEFAULT_MARGIN_REFERENCE_MODEL,
    FRAGMENTED_MODEL,
    PROMOTED_OFF_SWR_CLASS,
    STATIONARY_MODEL,
    _candidate_id,
    _nullable_integer_series,
    _read_required_csv,
    build_comparison_table,
)


def _model_rows(base: dict[str, object]) -> list[dict[str, object]]:
    values = {
        STATIONARY_MODEL: 0.0,
        DEFAULT_MARGIN_REFERENCE_MODEL: 2.0,
        FRAGMENTED_MODEL: 1.0,
        DEFAULT_FIRST_ORDER_IMM_MODEL: 10.0,
        DEFAULT_MARGIN_POSITIVE_MODEL: 9.0,
    }
    return [{**base, "model": model, "log_evidence": value} for model, value in values.items()]


def test_candidate_id_preserves_adjacent_large_integer_null_indices():
    lower = 2**53
    upper = lower + 1

    lower_id = _candidate_id(
        pd.Series({"session": "Rat1/Open1", "event_index": 10, "null_index": lower})
    )
    upper_id = _candidate_id(
        pd.Series({"session": "Rat1/Open1", "event_index": 10, "null_index": upper})
    )

    assert lower_id == f"Rat1/Open1|event=10|null={lower}"
    assert upper_id == f"Rat1/Open1|event=10|null={upper}"
    assert lower_id != upper_id


def test_required_csv_preserves_large_event_keys_when_column_has_missing_value(tmp_path):
    lower = 2**53
    upper = lower + 1
    rows = [
        *_model_rows(
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": str(lower),
                "evidence_comparable": True,
            }
        ),
        {
            "status": "success",
            "session": "Rat1/Open1",
            "event_index": "",
            "evidence_comparable": True,
            "model": "ignored-missing-event-key",
            "log_evidence": 0.0,
        },
        *_model_rows(
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": str(upper),
                "evidence_comparable": True,
            }
        ),
    ]
    path = tmp_path / "swr.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    loaded = _read_required_csv(path)
    comparison = build_comparison_table(
        swr_event_model_evidence=loaded,
        off_swr_event_model_evidence=pd.DataFrame(),
    )

    assert str(loaded["event_index"].dtype).startswith("string")
    assert comparison["event_index"].tolist() == [lower, upper]


def test_nullable_integer_series_rejects_nonmissing_invalid_identifier():
    with pytest.raises(TypeError, match="null_index must contain integer values"):
        _nullable_integer_series(pd.Series(["candidate-A"]))


def test_nullable_integer_series_keeps_textual_missing_values_nullable():
    values = _nullable_integer_series(pd.Series(["", "nan", "none", pd.NA]))

    assert str(values.dtype) == "Int64"
    assert values.isna().all()


def test_comparison_preserves_large_null_indices_across_swr_concat():
    lower = 2**53
    upper = lower + 1
    swr = pd.DataFrame(
        _model_rows(
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "evidence_comparable": True,
            }
        )
    )
    off = pd.DataFrame(
        [
            * _model_rows(
                {
                    "status": "success",
                    "session": "Rat1/Open1",
                    "event_index": 10,
                    "window_role": "promoted_off_swr_candidate",
                    "null_index": lower,
                    "evidence_comparable": True,
                }
            ),
            * _model_rows(
                {
                    "status": "success",
                    "session": "Rat1/Open1",
                    "event_index": 10,
                    "window_role": "promoted_off_swr_candidate",
                    "null_index": upper,
                    "evidence_comparable": True,
                }
            ),
        ]
    )

    comparison = build_comparison_table(
        swr_event_model_evidence=swr,
        off_swr_event_model_evidence=off,
    )
    promoted = comparison[comparison["event_class"].eq(PROMOTED_OFF_SWR_CLASS)].reset_index(drop=True)

    assert str(promoted["null_index"].dtype) == "Int64"
    assert promoted["null_index"].tolist() == [lower, upper]
    assert promoted["candidate_id"].tolist() == [
        f"Rat1/Open1|event=10|null={lower}",
        f"Rat1/Open1|event=10|null={upper}",
    ]


def test_comparison_rejects_nonmissing_invalid_off_swr_null_index():
    swr = pd.DataFrame(
        _model_rows(
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 1,
                "evidence_comparable": True,
            }
        )
    )
    off = pd.DataFrame(
        _model_rows(
            {
                "status": "success",
                "session": "Rat1/Open1",
                "event_index": 10,
                "window_role": "promoted_off_swr_candidate",
                "null_index": "candidate-A",
                "evidence_comparable": True,
            }
        )
    )

    with pytest.raises(TypeError, match="null_index must contain integer values"):
        build_comparison_table(
            swr_event_model_evidence=swr,
            off_swr_event_model_evidence=off,
        )

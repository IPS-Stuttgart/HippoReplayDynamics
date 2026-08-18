from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm  # noqa: F401
import hipporeplayimm.ground_truth_window_scope as window_scope


def _window_rows(**updates: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "session": "rat1",
        "event_index": 0,
        "window_start_s": 1.0,
        "window_end_s": 2.0,
        "ripple_peak": 1.25,
    }
    values.update(updates)
    return pd.DataFrame(
        {column: pd.Series([value], dtype=object) for column, value in values.items()}
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("window_start_s", np.nan),
        ("window_end_s", np.nan),
        ("window_start_s", "nan"),
        ("window_end_s", None),
    ],
)
def test_saved_window_rejects_partially_missing_bounds(column: str, value: object) -> None:
    rows = _window_rows(**{column: value})

    with pytest.raises(
        ValueError,
        match="window_start_s and window_end_s must either both be present or both be missing",
    ):
        window_scope._window_from_score_rows(rows)


def test_window_scope_does_not_silently_fallback_for_partial_bounds() -> None:
    rows = _window_rows(window_end_s=np.nan)

    with pytest.raises(ValueError, match="window_start_s and window_end_s"):
        window_scope._score_table_needs_window_scoped_decode(rows)


def test_saved_window_allows_both_bounds_to_be_missing() -> None:
    rows = _window_rows(window_start_s=np.nan, window_end_s="nan")

    assert window_scope._window_from_score_rows(rows) is None

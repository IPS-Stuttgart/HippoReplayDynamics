from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm  # noqa: F401
import hipporeplayimm.ground_truth_window_scope as window_scope


def _window_rows(**updates: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "window_start_s": 1.0,
        "window_end_s": 2.0,
        "ripple_peak": 1.25,
    }
    values.update(updates)
    return pd.DataFrame({column: pd.Series([value], dtype=object) for column, value in values.items()})


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("window_start_s", True),
        ("window_end_s", "not-a-number"),
        ("ripple_peak", np.array([1.25], dtype=float)),
        ("window_start_s", 1.0 + 0.0j),
        ("window_end_s", np.inf),
    ],
)
def test_saved_window_rejects_malformed_present_float_metadata(column: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"{column} must contain finite numeric scalar values"):
        window_scope._window_from_score_rows(_window_rows(**{column: value}))


def test_saved_window_accepts_numeric_text_bytes_and_zero_dimensional_scalars() -> None:
    window = window_scope._window_from_score_rows(
        _window_rows(
            window_start_s="1.0",
            window_end_s=np.array(2.0),
            ripple_peak=b"1.25",
        )
    )

    assert window is not None
    assert window.start == 1.0
    assert window.end == 2.0
    assert window.peak == 1.25


def test_saved_window_treats_explicit_missing_peak_as_midpoint_fallback() -> None:
    window = window_scope._window_from_score_rows(_window_rows(ripple_peak="nan"))

    assert window is not None
    assert window.peak == 1.5

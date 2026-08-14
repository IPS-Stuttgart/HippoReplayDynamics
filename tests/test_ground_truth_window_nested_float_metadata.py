from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hipporeplayimm  # noqa: F401
import hipporeplayimm.ground_truth_window_scope as window_scope


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _window_rows(**updates: object) -> pd.DataFrame:
    values: dict[str, object] = {
        "window_start_s": 1.0,
        "window_end_s": 2.0,
        "ripple_peak": 1.25,
    }
    values.update(updates)
    return pd.DataFrame({column: pd.Series([value], dtype=object) for column, value in values.items()})


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_saved_window_rejects_nested_boolean_float_metadata(value: object) -> None:
    with pytest.raises(ValueError, match=r"window_start_s must contain finite numeric scalar values"):
        window_scope._window_from_score_rows(
            _window_rows(window_start_s=_nested_object_scalar(value))
        )


def test_saved_window_accepts_nested_numeric_float_metadata() -> None:
    window = window_scope._window_from_score_rows(
        _window_rows(
            window_start_s=_nested_object_scalar(1.0),
            window_end_s=_nested_object_scalar("2.0"),
            ripple_peak=_nested_object_scalar(np.float64(1.25)),
        )
    )

    assert window is not None
    assert window.start == 1.0
    assert window.end == 2.0
    assert window.peak == 1.25

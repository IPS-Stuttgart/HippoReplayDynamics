from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.data import RippleEvent, _as_two_dimensional


def _valid_row() -> np.ndarray:
    return np.array([1.0, 1.2, 1.1, 5.0, 2.0, 1.5], dtype=float)


def test_ripple_events_reject_nonfinite_temporal_metadata() -> None:
    for column in range(3):
        row = _valid_row()
        row[column] = np.nan
        with pytest.raises(ValueError, match="finite start, end, and peak"):
            _as_two_dimensional(row, "Ripple_Events")


def test_ripple_events_reject_reversed_interval() -> None:
    row = _valid_row()
    row[0], row[1] = 1.3, 1.2

    with pytest.raises(ValueError, match="end times"):
        _as_two_dimensional(row, "Ripple_Events")


def test_ripple_events_reject_peak_outside_event_interval() -> None:
    for peak in (0.9, 1.3):
        row = _valid_row()
        row[2] = peak
        with pytest.raises(ValueError, match="peak times"):
            _as_two_dimensional(row, "Ripple_Events")


def test_column_major_ripple_events_are_validated_after_transpose() -> None:
    rows = np.vstack([_valid_row(), _valid_row() + np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])])
    column_major = rows.T

    validated = _as_two_dimensional(column_major, "Ripple_Events")

    np.testing.assert_allclose(validated, rows)
    assert validated.dtype == float


def test_ripple_event_from_row_rejects_malformed_temporal_metadata() -> None:
    row = _valid_row()
    row[2] = np.inf

    with pytest.raises(ValueError, match="finite start, end, and peak"):
        RippleEvent.from_row(row)


def test_ripple_event_from_row_rejects_short_rows_cleanly() -> None:
    with pytest.raises(ValueError, match="at least six"):
        RippleEvent.from_row(np.array([1.0, 1.2, 1.1]))

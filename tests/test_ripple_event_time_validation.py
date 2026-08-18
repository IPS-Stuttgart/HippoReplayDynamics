from __future__ import annotations

import warnings

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


@pytest.mark.parametrize(
    "imaginary_part",
    [0.0, 2.0],
)
def test_ripple_events_reject_complex_dtype_without_cast_warning(
    imaginary_part: float,
) -> None:
    row = _valid_row().astype(complex)
    row[3] = complex(row[3].real, imaginary_part)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real numeric"):
            _as_two_dimensional(row, "Ripple_Events")


def test_ripple_event_from_row_rejects_nested_complex_metadata_without_cast_warning() -> None:
    nested = np.empty((), dtype=object)
    nested[()] = np.complex128(2.0 + 1.0j)
    row = _valid_row().astype(object)
    row[4] = nested

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real numeric"):
            RippleEvent.from_row(row)

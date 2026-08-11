import numpy as np

from hipporeplayimm import encoding
from hipporeplayimm.encoding_position_support_patch import (
    _max_contiguous_sample_gap_s,
    _tracking_support_intervals,
)
import hipporeplayimm.place_field_run_local_kinematics as run_local_kinematics
from hipporeplayimm.position_validation import _decode_windows


def test_position_decode_windows_stop_at_tracking_dropout() -> None:
    times = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2], dtype=float)
    x = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0], dtype=float)
    xy = np.column_stack([x, np.zeros_like(x)])
    movement = np.ones(times.shape, dtype=bool)

    windows = _decode_windows(
        times,
        xy,
        movement,
        np.array([[0.0, 3.2]], dtype=float),
        1.0,
    )

    np.testing.assert_allclose(
        np.array(
            [[window["start_time"], window["end_time"]] for window in windows],
            dtype=float,
        ),
        np.array([[0.0, 0.3], [3.0, 3.2]], dtype=float),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.array([[window["true_x"], window["true_y"]] for window in windows]),
        np.array([[1.0, 0.0], [100.5, 0.0]]),
        atol=1e-12,
    )


def test_short_trace_dropout_does_not_inflate_nominal_tracking_cadence() -> None:
    times = np.array([0.0, 0.1, 3.0], dtype=float)
    xy = np.column_stack(
        [
            np.array([0.0, 1.0, 100.0], dtype=float),
            np.zeros(3, dtype=float),
        ]
    )

    max_gap_s = _max_contiguous_sample_gap_s(times)
    assert np.isclose(max_gap_s, 0.5)
    assert np.isclose(
        run_local_kinematics._max_contiguous_sample_gap_s(times),
        0.5,
    )

    segments = run_local_kinematics._split_indices_at_sample_gaps(
        np.arange(times.size),
        times,
        max_gap_s,
    )
    assert [segment.tolist() for segment in segments] == [[0, 1], [2]]

    interpolated = encoding._interp_positions(times, xy, np.array([1.0], dtype=float))
    assert np.isnan(interpolated).all()

    support = _tracking_support_intervals(
        times,
        np.array([[0.0, 3.1]], dtype=float),
    )
    np.testing.assert_allclose(
        support,
        np.array([[0.0, 0.2], [3.0, 3.1]], dtype=float),
        atol=1e-12,
    )

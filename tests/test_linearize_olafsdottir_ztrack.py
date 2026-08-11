import numpy as np

from scripts.linearize_olafsdottir_ztrack import (
    _sample_durations,
    smooth_positions,
    speed_from_linear_position,
)


def test_smooth_positions_does_not_zero_pad_edges():
    xy = np.tile(np.array([[12.5, -4.0]], dtype=float), (7, 1))
    valid = np.ones(xy.shape[0], dtype=bool)

    smoothed = smooth_positions(xy, valid, window_samples=5)

    np.testing.assert_allclose(smoothed, xy)


def test_smooth_positions_keeps_output_length_for_even_window():
    xy = np.column_stack(
        [
            np.arange(6.0),
            np.arange(6.0) + 10.0,
        ]
    )
    valid = np.ones(xy.shape[0], dtype=bool)

    assert smooth_positions(xy, valid, window_samples=4).shape == xy.shape


def test_smooth_positions_does_not_bridge_long_tracking_dropout():
    gap_samples = 10
    xy = np.vstack(
        [
            np.zeros((3, 2), dtype=float),
            np.full((gap_samples, 2), np.nan, dtype=float),
            np.full((3, 2), 100.0, dtype=float),
        ]
    )
    valid = np.r_[
        np.ones(3, dtype=bool),
        np.zeros(gap_samples, dtype=bool),
        np.ones(3, dtype=bool),
    ]
    times = np.arange(xy.shape[0], dtype=float) * 0.02

    smoothed = smooth_positions(
        xy,
        valid,
        window_samples=5,
        times_s=times,
    )

    np.testing.assert_allclose(smoothed[:3], 0.0)
    assert np.isnan(smoothed[3:-3]).all()
    np.testing.assert_allclose(smoothed[-3:], 100.0)


def test_speed_does_not_bridge_long_tracking_dropout():
    gap_samples = 10
    times = np.arange(6 + gap_samples, dtype=float) * 0.02
    linear = np.r_[
        [0.0, 1.0, 2.0],
        np.full(gap_samples, np.nan),
        [100.0, 101.0, 102.0],
    ]
    valid = np.isfinite(linear)

    speed = speed_from_linear_position(times, linear, valid)

    np.testing.assert_allclose(speed[valid], np.full(6, 50.0), atol=1.0e-12)
    assert np.isnan(speed[~valid]).all()


def test_sample_durations_do_not_charge_long_timestamp_gap_as_occupancy():
    times = np.array([0.0, 0.02, 0.04, 2.0, 2.02, 2.04], dtype=float)

    durations = _sample_durations(times)

    np.testing.assert_allclose(durations, np.full(times.shape, 0.02), atol=1.0e-12)

import numpy as np

from scripts.detect_olafsdottir_sleep_replay_events import _moving_average


def test_moving_average_does_not_zero_pad_edges():
    values = np.full(7, 12.5, dtype=float)

    smoothed = _moving_average(values, window=5)

    np.testing.assert_allclose(smoothed, values)


def test_moving_average_keeps_output_length_for_even_window():
    values = np.arange(6.0)

    smoothed = _moving_average(values, window=4)

    assert smoothed.shape == values.shape

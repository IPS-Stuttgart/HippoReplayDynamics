import numpy as np

from scripts.linearize_olafsdottir_ztrack import smooth_positions


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

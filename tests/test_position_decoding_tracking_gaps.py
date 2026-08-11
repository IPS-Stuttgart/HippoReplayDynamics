import numpy as np

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

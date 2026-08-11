import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from replay_behavior_alignment import _position_at_time  # noqa: E402


def test_behavior_position_lookup_rejects_queries_inside_tracking_gaps():
    position = np.column_stack(
        [
            np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2], dtype=float),
            np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0], dtype=float),
            np.zeros(6, dtype=float),
        ]
    )
    query_times = np.array([0.15, 0.2, 0.21, 1.0, 2.99, 3.0, 3.05], dtype=float)

    interpolated = np.vstack([_position_at_time(position, query_time) for query_time in query_times])

    assert np.isnan(interpolated[2:5]).all()
    np.testing.assert_allclose(
        interpolated[[0, 1, 5, 6]],
        np.array([[1.5, 0.0], [2.0, 0.0], [100.0, 0.0], [100.5, 0.0]]),
    )


def test_behavior_position_lookup_rejects_short_trace_dropout():
    position = np.column_stack(
        [
            np.array([0.0, 0.1, 3.0], dtype=float),
            np.array([0.0, 1.0, 100.0], dtype=float),
            np.zeros(3, dtype=float),
        ]
    )

    interpolated = _position_at_time(position, 1.0)

    assert np.isnan(interpolated).all()

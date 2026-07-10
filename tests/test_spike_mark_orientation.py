from __future__ import annotations

import numpy as np

from hipporeplayimm.data import _coerce_mark_matrix


def test_square_feature_major_mark_matrix_uses_embedded_times_to_find_spike_axis() -> None:
    spike_times = np.array([1.0, 2.0])
    feature_major = np.array(
        [
            [1.0, 2.0],
            [40.0, 50.0],
        ]
    )

    marks = _coerce_mark_matrix(
        feature_major,
        spike_count=spike_times.size,
        spike_times=spike_times,
    )

    assert marks is not None
    np.testing.assert_allclose(marks, np.array([[40.0], [50.0]]))


def test_square_spike_major_mark_matrix_keeps_existing_orientation() -> None:
    spike_times = np.array([1.0, 2.0])
    spike_major = np.array(
        [
            [1.0, 40.0],
            [2.0, 50.0],
        ]
    )

    marks = _coerce_mark_matrix(
        spike_major,
        spike_count=spike_times.size,
        spike_times=spike_times,
    )

    assert marks is not None
    np.testing.assert_allclose(marks, np.array([[40.0], [50.0]]))

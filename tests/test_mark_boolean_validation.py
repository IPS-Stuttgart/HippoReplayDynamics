import numpy as np

from hipporeplayimm.data import _coerce_mark_matrix


def test_boolean_spike_mark_candidate_is_rejected() -> None:
    mark_candidate = np.array(
        [
            [True, False],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )

    marks = _coerce_mark_matrix(
        mark_candidate,
        spike_count=3,
        spike_times=np.array([0.0, 1.0, 2.0]),
    )

    assert marks is None


def test_mixed_boolean_numeric_spike_mark_candidate_is_rejected() -> None:
    mark_candidate = [
        [0.1, True],
        [0.2, False],
        [0.3, 1.0],
    ]

    marks = _coerce_mark_matrix(
        mark_candidate,
        spike_count=3,
        spike_times=np.array([0.0, 1.0, 2.0]),
    )

    assert marks is None

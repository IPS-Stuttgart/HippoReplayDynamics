from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hipporeplayimm.result_improvements import shuffle_cell_identities_session


@dataclass(frozen=True)
class _Marks:
    cell_ids: np.ndarray | None


@dataclass(frozen=True)
class _Session:
    spikes: np.ndarray
    spike_marks: _Marks | None = None


def test_cell_identity_shuffle_rejects_identity_draw() -> None:
    session = _Session(
        spikes=np.array(
            [
                [0.1, 10.0],
                [0.2, 11.0],
                [0.3, 10.0],
            ],
            dtype=float,
        ),
        spike_marks=_Marks(cell_ids=np.array([10, 11, 10], dtype=int)),
    )

    shuffled = shuffle_cell_identities_session(session, random_seed=0)

    np.testing.assert_array_equal(
        shuffled.spikes[:, 1],
        np.array([11.0, 10.0, 11.0]),
    )
    assert shuffled.spike_marks is not None
    np.testing.assert_array_equal(
        shuffled.spike_marks.cell_ids,
        np.array([11, 10, 11]),
    )


def test_cell_identity_shuffle_preserves_unavoidable_singleton() -> None:
    session = _Session(
        spikes=np.array([[0.1, 10.0], [0.2, 10.0]], dtype=float),
        spike_marks=_Marks(cell_ids=np.array([10, 10], dtype=int)),
    )

    shuffled = shuffle_cell_identities_session(session, random_seed=0)

    np.testing.assert_array_equal(shuffled.spikes, session.spikes)
    assert shuffled.spike_marks is not None
    np.testing.assert_array_equal(shuffled.spike_marks.cell_ids, np.array([10, 10]))


def test_cell_identity_shuffle_rejects_fractional_spike_cell_ids() -> None:
    session = _Session(
        spikes=np.array([[0.1, 10.5], [0.2, 11.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="spike cell IDs must be integer-valued"):
        shuffle_cell_identities_session(session, random_seed=0)


def test_cell_identity_shuffle_rejects_fractional_mark_cell_ids() -> None:
    session = _Session(
        spikes=np.array([[0.1, 10.0], [0.2, 11.0]], dtype=float),
        spike_marks=_Marks(cell_ids=np.array([10.5, 11.0], dtype=float)),
    )

    with pytest.raises(ValueError, match="spike mark cell IDs must be integer-valued"):
        shuffle_cell_identities_session(session, random_seed=0)

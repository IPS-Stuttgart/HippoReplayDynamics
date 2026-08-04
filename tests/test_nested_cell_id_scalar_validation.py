from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from hipporeplayimm.data import ReplaySession
from hipporeplayimm.spike_cell_id_emission_validation import _coerce_integral_ids


def _nested_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


def _session_with_cell_id(value: object) -> ReplaySession:
    spikes = np.empty((1, 2), dtype=object)
    spikes[0, 0] = 0.1
    spikes[0, 1] = value
    return ReplaySession(
        rat="RatX",
        name="OpenY",
        path=Path("unused"),
        position=np.array([[0.0, 0.0, 0.0]], dtype=float),
        spikes=spikes,
        tetrode_cell_ids=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.array([], dtype=int),
        inhibitory_neurons=np.array([], dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.empty((0, 2), dtype=float),
        sleep_box_immobile_times=np.empty((0, 2), dtype=float),
        sleep_times=np.empty((0, 2), dtype=float),
        rem_times=np.empty((0, 2), dtype=float),
        well_sequence=None,
        metadata={},
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "boolean identifiers"),
        (np.bool_(False), "boolean identifiers"),
        (complex(2.0, 0.0), "real integer identifiers"),
        (np.complex128(2.0 + 3.0j), "real integer identifiers"),
        (np.clongdouble(2.0 + 3.0j), "real integer identifiers"),
    ],
)
def test_replay_session_cell_ids_reject_nested_nonreal_scalars(
    value: object,
    message: str,
) -> None:
    session = _session_with_cell_id(_nested_scalar(value))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=message):
            _ = session.cell_ids


def test_replay_session_cell_ids_preserve_nested_real_scalar() -> None:
    session = _session_with_cell_id(_nested_scalar(np.longdouble(2.0)))

    np.testing.assert_array_equal(session.cell_ids, np.array([2], dtype=int))


def test_cell_id_parser_rejects_self_referential_scalar_array() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic
    values = np.empty(1, dtype=object)
    values[0] = cyclic

    with pytest.raises(ValueError, match="scalar integer identifiers"):
        _coerce_integral_ids(values, "spike cell IDs")

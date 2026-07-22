from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.position_validation import (
    PositionDecodingConfig,
    validate_session_position_decoding,
)


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 2.0, 1.0]),
    ],
)
def test_position_decoding_rejects_nonmonotonic_timestamps(times: np.ndarray) -> None:
    session = SimpleNamespace(
        position=np.column_stack([times, np.zeros_like(times), np.zeros_like(times)]),
        spikes=np.empty((0, 2), dtype=float),
        excitatory_neurons=np.empty(0, dtype=int),
        inhibitory_neurons=np.empty(0, dtype=int),
        ripple_events=np.empty((0, 6), dtype=float),
        run_times=np.array([[0.0, 2.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="position times must be strictly increasing"):
        validate_session_position_decoding(session, PositionDecodingConfig())

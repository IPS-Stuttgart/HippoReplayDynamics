from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.result_improvement_extensions import _apply_replay_gains


@pytest.mark.parametrize("mode", ["none", "cell", "event", "event-cell"])
def test_replay_gain_modes_reject_caps_below_one(mode: str) -> None:
    rates_hz = np.ones((2, 1), dtype=float)
    counts = np.array([[1, 0]], dtype=int)

    with pytest.raises(
        ValueError,
        match="max_gain must be finite and greater than or equal to 1",
    ):
        _apply_replay_gains(
            rates_hz,
            counts,
            np.array([1.0], dtype=float),
            mode=mode,
            prior_count=0.0,
            max_gain=0.5,
        )

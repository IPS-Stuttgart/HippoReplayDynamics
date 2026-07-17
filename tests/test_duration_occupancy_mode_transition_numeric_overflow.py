from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.duration_occupancy as duration_occupancy
from hipporeplayimm import apply_runtime_patches


def test_custom_duration_imm_transition_normalizes_numeric_overflow() -> None:
    apply_runtime_patches()
    transition = np.eye(3, dtype=object)
    transition[0, 0] = 10**400

    with pytest.raises(
        ValueError,
        match="mode transition matrix 0 must contain numeric probabilities",
    ):
        duration_occupancy._resolve_mode_transitions(
            object(),
            3,
            0.9,
            [transition],
            1,
        )

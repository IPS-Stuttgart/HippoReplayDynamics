from __future__ import annotations

import pytest

from hipporeplayimm.duration_occupancy_mode_transition_validation import (
    _validate_mode_transition_sequence,
)


def test_custom_duration_imm_mode_transition_rejects_ragged_matrix() -> None:
    ragged_transition = [
        [0.8, 0.2],
        [0.1],
    ]

    with pytest.raises(ValueError, match="rectangular numeric probability matrix"):
        _validate_mode_transition_sequence(
            [ragged_transition],
            n_modes=2,
            n_transitions=1,
        )

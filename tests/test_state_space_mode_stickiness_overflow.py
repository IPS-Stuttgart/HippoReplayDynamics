from __future__ import annotations

import pytest

from hipporeplayimm.state_space import _mode_transition_matrix


def test_mode_transition_matrix_normalizes_stickiness_numeric_overflow() -> None:
    with pytest.raises(
        ValueError,
        match=r"mode_stickiness must be in \[0, 1\]",
    ):
        _mode_transition_matrix(3, 10**400)

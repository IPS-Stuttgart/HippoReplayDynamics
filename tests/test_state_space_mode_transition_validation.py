from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_utils import _mode_transition_matrix


def test_mode_transition_matrix_rejects_zero_modes() -> None:
    with pytest.raises(ValueError, match="n_modes must be positive"):
        _mode_transition_matrix(0, 0.9)


def test_mode_transition_matrix_single_mode_validates_stickiness() -> None:
    with pytest.raises(ValueError, match="mode_stickiness"):
        _mode_transition_matrix(1, 1.2)

    np.testing.assert_allclose(_mode_transition_matrix(1, 0.25), np.ones((1, 1)))


@pytest.mark.parametrize(
    "stickiness",
    [
        True,
        False,
        np.bool_(True),
        np.array(False, dtype=object),
    ],
)
def test_mode_transition_matrix_rejects_boolean_stickiness(stickiness: object) -> None:
    with pytest.raises(TypeError, match="not boolean"):
        _mode_transition_matrix(3, stickiness)

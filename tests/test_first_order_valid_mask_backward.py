from __future__ import annotations

import numpy as np

from hipporeplayimm.state_space_first_order import _apply_transition_backward


def test_fragmented_backward_transition_respects_valid_bin_mask() -> None:
    values = np.array([2.0, 100.0, 4.0, 200.0], dtype=float)
    valid_bin_mask = np.array([True, False, True, False], dtype=bool)

    backward = _apply_transition_backward(
        None,
        values,
        valid_bin_mask=valid_bin_mask,
    )

    np.testing.assert_allclose(backward, np.array([3.0, 0.0, 3.0, 0.0], dtype=float))

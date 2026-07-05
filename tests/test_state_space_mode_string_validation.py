from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.state_space import _mode_transition_matrix as public_mode_transition_matrix
from hipporeplayimm.state_space_first_order import _mode_transition_matrix as first_order_mode_transition_matrix
from hipporeplayimm.state_space_utils import _mode_transition_matrix as utils_mode_transition_matrix


@pytest.mark.parametrize(
    "stickiness",
    [
        "0.95",
        b"0.95",
        np.str_("0.95"),
        np.asarray("0.95"),
        np.asarray(b"0.95"),
    ],
)
def test_state_space_mode_transition_rejects_string_stickiness_aliases(stickiness: object) -> None:
    hipporeplayimm.apply_runtime_patches()

    for helper in (
        public_mode_transition_matrix,
        first_order_mode_transition_matrix,
        utils_mode_transition_matrix,
    ):
        with pytest.raises(TypeError, match="mode_stickiness"):
            helper(3, stickiness)

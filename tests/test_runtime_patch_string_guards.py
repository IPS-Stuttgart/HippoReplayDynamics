from __future__ import annotations

import numpy as np
import pytest


def test_repeated_runtime_patches_preserve_state_space_numeric_string_guards() -> None:
    import hipporeplayimm
    import hipporeplayimm.state_space as state_space

    hipporeplayimm.apply_runtime_patches()
    hipporeplayimm.apply_runtime_patches()

    log_emission = np.log(np.array([0.6, 0.3, 0.1], dtype=float))

    with pytest.raises(TypeError, match="top_k.*string"):
        state_space._top_candidate_indices(log_emission, "2")

    with pytest.raises(TypeError, match="mass_threshold.*string"):
        state_space._mass_retaining_candidate_indices(log_emission, "0.75")

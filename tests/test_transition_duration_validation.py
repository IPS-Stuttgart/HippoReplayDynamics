from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space import StateSpaceDecoderConfig
from hipporeplayimm.state_space_displacement_imm import (
    _coerce_transition_durations as displacement_imm_transition_durations,
    _duration_adjusted_decays as displacement_imm_duration_decays,
)
from hipporeplayimm.state_space_displacement_momentum import (
    _coerce_transition_durations as displacement_transition_durations,
    _duration_adjusted_decays as displacement_duration_decays,
)
from hipporeplayimm.state_space_sparse_momentum import (
    _coerce_transition_durations as sparse_transition_durations,
    _duration_adjusted_decays as sparse_duration_decays,
)
from hipporeplayimm.state_space_trajectory_imm import (
    _coerce_transition_durations as trajectory_imm_transition_durations,
    _duration_adjusted_decays as trajectory_imm_duration_decays,
)


@pytest.mark.parametrize(
    "coerce_durations",
    [
        sparse_transition_durations,
        trajectory_imm_transition_durations,
        displacement_transition_durations,
        displacement_imm_transition_durations,
    ],
)
def test_exact_model_transition_duration_helpers_reject_invalid_values(coerce_durations) -> None:
    bad_rows = (
        [0.01, np.nan],
        [0.01, 0.0],
        [0.01, -0.02],
    )
    for values in bad_rows:
        with pytest.raises(ValueError, match="transition durations"):
            coerce_durations(values, n_time=3, fallback_dt=0.01)

    np.testing.assert_allclose(
        coerce_durations([], n_time=3, fallback_dt=0.02),
        np.array([0.02, 0.02], dtype=float),
    )
    with pytest.raises(ValueError, match="fallback dt"):
        coerce_durations([], n_time=3, fallback_dt=np.nan)


@pytest.mark.parametrize(
    "duration_decays",
    [
        sparse_duration_decays,
        trajectory_imm_duration_decays,
        displacement_duration_decays,
        displacement_imm_duration_decays,
    ],
)
def test_exact_model_duration_decay_helpers_reject_invalid_durations(duration_decays) -> None:
    config = StateSpaceDecoderConfig()
    bad_arrays = (
        np.array([0.01, np.nan], dtype=float),
        np.array([0.01, 0.0], dtype=float),
        np.array([0.01, -0.02], dtype=float),
    )
    for durations in bad_arrays:
        with pytest.raises(ValueError, match="transition durations"):
            duration_decays(config, durations, 0.01)

    assert duration_decays(config, np.empty(0, dtype=float), 0.01).size == 0

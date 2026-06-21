from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_model import StateSpaceDecoderConfig
from hipporeplayimm.state_space_sparse_momentum import _duration_adjusted_decays, _time_scales


def test_sparse_momentum_duration_helpers_reject_invalid_transition_durations() -> None:
    config = StateSpaceDecoderConfig()
    bad_duration_sets = (
        np.array([0.01, np.nan], dtype=float),
        np.array([0.01, 0.0], dtype=float),
        np.array([0.01, -0.02], dtype=float),
    )

    for durations in bad_duration_sets:
        with pytest.raises(ValueError, match="transition durations"):
            _duration_adjusted_decays(config, durations, 0.01)
        with pytest.raises(ValueError, match="transition durations"):
            _time_scales(durations)


def test_sparse_momentum_duration_helpers_preserve_valid_outputs() -> None:
    durations = np.array([0.01, 0.02, 0.01], dtype=float)

    decays = _duration_adjusted_decays(StateSpaceDecoderConfig(momentum_velocity_decay=0.9), durations, 0.01)
    scales = _time_scales(durations)

    np.testing.assert_allclose(decays, np.array([0.9, 0.81, 0.9], dtype=float))
    np.testing.assert_allclose(scales, np.array([1.0, 2.0, 0.5], dtype=float))

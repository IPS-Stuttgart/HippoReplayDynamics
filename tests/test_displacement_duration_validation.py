from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.state_space_displacement_momentum import (
    _coerce_transition_durations,
    _duration_adjusted_decays,
    _duration_scale_at,
    _time_scales,
)
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig


def test_displacement_transition_duration_coercion_rejects_invalid_values() -> None:
    bad_duration_sets = (
        [0.01, np.nan],
        [0.01, 0.0],
        [0.01, -0.02],
    )

    for durations in bad_duration_sets:
        with pytest.raises(ValueError, match="transition durations"):
            _coerce_transition_durations(durations, n_time=3, fallback_dt=0.01)

    np.testing.assert_allclose(
        _coerce_transition_durations([], n_time=3, fallback_dt=0.02),
        np.array([0.02, 0.02], dtype=float),
    )
    with pytest.raises(ValueError, match="fallback dt"):
        _coerce_transition_durations([], n_time=3, fallback_dt=np.nan)


def test_displacement_duration_helpers_reject_invalid_transition_durations() -> None:
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
        with pytest.raises(ValueError, match="transition durations"):
            _duration_scale_at(durations, 0, 0.01)

    with pytest.raises(ValueError, match="reference dt"):
        _duration_scale_at(np.array([0.01], dtype=float), 0, 0.0)


def test_displacement_duration_helpers_preserve_valid_outputs() -> None:
    durations = np.array([0.01, 0.02, 0.01], dtype=float)

    decays = _duration_adjusted_decays(StateSpaceDecoderConfig(momentum_velocity_decay=0.9), durations, 0.01)
    scales = _time_scales(durations)

    np.testing.assert_allclose(decays, np.array([0.9, 0.81, 0.9], dtype=float))
    np.testing.assert_allclose(scales, np.array([1.0, 2.0, 0.5], dtype=float))
    assert _duration_scale_at(durations, 1, 0.01) == pytest.approx(2.0)

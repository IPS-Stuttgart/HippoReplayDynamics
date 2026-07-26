from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import duration_occupancy
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.sparse_momentum_duration_validation import _valid_transition_durations
from hipporeplayimm.state_space_displacement_momentum import (
    _duration_scale_at as displacement_duration_scale_at,
)
from hipporeplayimm.state_space_displacement_momentum import (
    _time_scales as displacement_time_scales,
)
from hipporeplayimm.state_space_model import StateSpaceDecoderConfig
from hipporeplayimm.state_space_sparse_momentum import (
    _duration_adjusted_decays,
    _score_sparse_momentum_exact,
    _time_scales,
)


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


def test_sparse_momentum_duration_validator_rejects_non_1d_arrays() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        _valid_transition_durations(np.array([[0.01, 0.02]], dtype=float))

    with pytest.raises(ValueError, match="one-dimensional"):
        _valid_transition_durations(np.empty((0, 1), dtype=float))


def test_sparse_momentum_duration_helpers_preserve_valid_outputs() -> None:
    durations = np.array([0.01, 0.02, 0.01], dtype=float)

    decays = _duration_adjusted_decays(StateSpaceDecoderConfig(momentum_velocity_decay=0.9), durations, 0.01)
    scales = _time_scales(durations)

    np.testing.assert_allclose(decays, np.array([0.9, 0.81, 0.9], dtype=float))
    np.testing.assert_allclose(scales, np.array([1.0, 2.0, 0.5], dtype=float))


@pytest.mark.parametrize(
    "time_scale_helper",
    (
        _time_scales,
        displacement_time_scales,
        duration_occupancy._time_scales,
    ),
)
def test_momentum_time_scale_helpers_reject_unrepresentable_duration_ratios(time_scale_helper) -> None:
    durations = np.array([np.finfo(float).tiny, np.finfo(float).max], dtype=float)

    with pytest.raises(ValueError, match="momentum time scales must be finite and positive"):
        time_scale_helper(durations)


def test_displacement_duration_scale_rejects_unrepresentable_ratio() -> None:
    with pytest.raises(ValueError, match="momentum duration scale must be finite and positive"):
        displacement_duration_scale_at(
            np.array([np.finfo(float).max], dtype=float),
            0,
            np.finfo(float).tiny,
        )


def test_sparse_momentum_exact_rejects_nonfinite_bin_centers() -> None:
    emissions = LogEmissionTensor(
        log_likelihood=np.log(np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)),
        spike_counts=np.zeros((2, 1), dtype=int),
        times=np.array([0.0, 0.01]),
        dt=0.01,
        cell_ids=np.array([1]),
        n_spikes=0,
    )
    centers = np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="bin_centers must be finite"):
        _score_sparse_momentum_exact(
            emissions,
            centers,
            StateSpaceDecoderConfig(),
            emissions.transition_durations,
        )

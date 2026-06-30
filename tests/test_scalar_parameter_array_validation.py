from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.encoding import (
    EncodingConfig,
    _poisson_log_emissions,
    _time_bin_edges,
    _validate_encoding_config,
)
from hipporeplayimm.model_parameter_validation import _validate_replay_calibration_max_gain
from hipporeplayimm.models import CandidateKinematicModel, DiffusionModel


@dataclass(frozen=True)
class _Calibration:
    max_gain: object


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_cm": np.array([12.0])},
        {"max_step_sigma": np.array([3.0])},
    ],
)
def test_diffusion_model_rejects_array_shaped_scalar_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="numeric scalar"):
        DiffusionModel(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diffusion_sigma_cm": np.array([12.0])},
        {"velocity_decay": np.array([0.95])},
        {"mode_stickiness": np.array([0.94])},
    ],
)
def test_candidate_kinematic_model_rejects_array_shaped_scalar_parameters(kwargs: dict[str, object]) -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="numeric scalar"):
        CandidateKinematicModel(**kwargs)  # type: ignore[arg-type]


def test_encoding_config_rejects_array_shaped_scalar_parameter() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=r"bin_size_cm.*numeric scalar"):
        _validate_encoding_config(EncodingConfig(bin_size_cm=np.array([4.0])))  # type: ignore[arg-type]


def test_time_bin_edges_rejects_array_shaped_scalar_bin_size() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=r"time_bin_s.*numeric scalar"):
        _time_bin_edges(0.0, 1.0, np.array([0.02]))


def test_poisson_log_emissions_rejects_array_shaped_scalar_calibration() -> None:
    hipporeplayimm.apply_runtime_patches()
    spike_counts = np.array([[0]], dtype=int)
    rates_hz = np.array([[1.0, 2.0]], dtype=float)

    with pytest.raises(TypeError, match=r"spike_rate_scale.*numeric scalar"):
        _poisson_log_emissions(spike_counts, rates_hz, 0.02, spike_rate_scale=np.array([1.0]))


def test_replay_calibration_rejects_array_shaped_max_gain() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match=r"max_gain.*numeric scalar"):
        _validate_replay_calibration_max_gain(_Calibration(max_gain=np.array([2.0])))

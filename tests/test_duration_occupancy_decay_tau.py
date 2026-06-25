from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.duration_occupancy import _duration_adjusted_decays_from_config
from hipporeplayimm.encoding import LogEmissionTensor
from hipporeplayimm.state_space import StateSpaceDecoderConfig, StateSpaceReplayModel


def test_duration_adjusted_decays_from_tau_use_physical_time() -> None:
    config = SimpleNamespace(
        momentum_velocity_decay=0.95,
        momentum_velocity_decay_tau_s=0.060,
    )
    durations = np.asarray([0.003, 0.006, 0.012], dtype=float)

    decays = _duration_adjusted_decays_from_config(config, durations, 0.003)

    assert np.allclose(decays, np.exp(-durations / 0.060))


def test_duration_adjusted_decays_keep_legacy_path_when_tau_disabled() -> None:
    config = SimpleNamespace(
        momentum_velocity_decay=0.95,
        momentum_velocity_decay_tau_s=0.0,
    )
    durations = np.asarray([0.003, 0.006], dtype=float)

    decays = _duration_adjusted_decays_from_config(config, durations, 0.003)

    assert np.allclose(decays, np.asarray([0.95, 0.95**2]))


@pytest.mark.parametrize("bad_tau", [float("nan"), float("inf"), -0.001])
def test_duration_adjusted_decays_reject_invalid_tau(bad_tau: float) -> None:
    config = SimpleNamespace(
        momentum_velocity_decay=0.95,
        momentum_velocity_decay_tau_s=bad_tau,
    )

    with pytest.raises(ValueError, match="momentum_velocity_decay_tau_s"):
        _duration_adjusted_decays_from_config(config, np.asarray([0.003]), 0.003)


def test_duration_adjusted_decays_reject_negative_tau() -> None:
    config = SimpleNamespace(momentum_velocity_decay=0.95, momentum_velocity_decay_tau_s=-1.0)

    with pytest.raises(ValueError, match="momentum_velocity_decay_tau_s"):
        _duration_adjusted_decays_from_config(config, np.asarray([0.003]), 0.003)


@pytest.mark.parametrize("mode", ["momentum", "imm"])
def test_duration_momentum_diagnostics_report_per_transition_values(mode: str) -> None:
    durations = np.asarray([0.003, 0.006], dtype=float)
    emissions = LogEmissionTensor(
        log_likelihood=np.log(
            np.asarray(
                [
                    [0.70, 0.20, 0.10],
                    [0.10, 0.70, 0.20],
                    [0.10, 0.20, 0.70],
                ],
                dtype=float,
            )
        ),
        spike_counts=np.zeros((3, 1), dtype=int),
        times=np.asarray([0.0, 0.003, 0.009], dtype=float),
        dt=0.003,
        cell_ids=np.asarray([1], dtype=int),
        n_spikes=0,
        bin_durations=np.asarray([0.003, 0.003, 0.006], dtype=float),
        transition_durations=durations,
    )
    bin_centers = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    config = StateSpaceDecoderConfig(
        mode=mode,
        momentum_sigma_cm_sqrt_s=10.0,
        momentum_initial_sigma_cm_sqrt_s=20.0,
        momentum_velocity_decay=0.9,
        momentum_candidate_top_k=0,
    )

    score = StateSpaceReplayModel(mode=mode, config=config).score(emissions, bin_centers)

    diagnostics = score.diagnostics
    assert np.allclose(_float_series(diagnostics["state_space_transition_durations_s"]), durations)
    assert np.allclose(
        _float_series(diagnostics["state_space_momentum_transition_sigma_cm_per_step"]),
        10.0 * np.sqrt(durations),
    )
    assert np.allclose(
        _float_series(diagnostics["state_space_momentum_initial_transition_sigma_cm_per_step"]),
        20.0 * np.sqrt(durations),
    )
    assert np.allclose(
        _float_series(diagnostics["state_space_momentum_velocity_decay_per_step"]),
        np.asarray([0.9, 0.9**2], dtype=float),
    )
    assert np.isclose(float(diagnostics["state_space_momentum_transition_sigma_cm"]), 10.0 * np.sqrt(np.median(durations)))
    assert np.isclose(float(diagnostics["state_space_momentum_initial_transition_sigma_cm"]), 20.0 * np.sqrt(durations[0]))
    assert np.isclose(float(diagnostics["state_space_momentum_velocity_decay_effective"]), np.median([0.9, 0.9**2]))


def _float_series(value: object) -> np.ndarray:
    parts = [part for part in str(value).split(",") if part]
    return np.asarray([float(part) for part in parts], dtype=float)

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.duration_occupancy import _duration_adjusted_decays_from_config


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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipporeplayimm.duration_dynamics import _decays


def test_duration_decays_reject_invalid_transition_durations() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        _decays(0.95, np.ones((1, 1), dtype=float), 0.02)

    for durations in (
        np.array([0.02, np.nan], dtype=float),
        np.array([0.02, 0.0], dtype=float),
        np.array([0.02, -0.01], dtype=float),
    ):
        with pytest.raises(ValueError, match="finite positive durations"):
            _decays(0.95, durations, 0.02)


def test_duration_decays_reject_invalid_reference_dt_and_decay() -> None:
    durations = np.array([0.02, 0.04], dtype=float)

    for reference_dt in (float("nan"), float("inf"), 0.0, -0.02):
        with pytest.raises(ValueError, match="reference dt"):
            _decays(0.95, durations, reference_dt)

    for decay in (float("nan"), float("inf"), -0.1):
        with pytest.raises(ValueError, match="momentum_velocity_decay"):
            _decays(decay, durations, 0.02)


def test_duration_decays_preserve_zero_decay_without_flooring() -> None:
    durations = np.array([0.02, 0.04], dtype=float)

    np.testing.assert_allclose(_decays(0.0, durations, 0.02), np.zeros(2, dtype=float))


def test_duration_decays_honor_physical_time_tau_config() -> None:
    durations = np.array([0.02, 0.04], dtype=float)
    config = SimpleNamespace(
        momentum_velocity_decay=0.25,
        momentum_velocity_decay_tau_s=0.04,
    )

    np.testing.assert_allclose(_decays(config, durations, 0.02), np.exp(-durations / 0.04))


def test_duration_decays_reject_invalid_physical_time_tau_config() -> None:
    durations = np.array([0.02, 0.04], dtype=float)

    for tau_s in (float("nan"), float("inf"), -0.1):
        config = SimpleNamespace(
            momentum_velocity_decay=0.25,
            momentum_velocity_decay_tau_s=tau_s,
        )
        with pytest.raises(ValueError, match="momentum_velocity_decay_tau_s"):
            _decays(config, durations, 0.02)

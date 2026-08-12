from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hipporeplayimm.duration_dynamics as duration_dynamics
import hipporeplayimm.encoding as encoding
import hipporeplayimm.kd_reference as kd_reference
import hipporeplayimm.state_space as state_space


def _bare_emission() -> SimpleNamespace:
    return SimpleNamespace(
        n_time=2,
        dt=0.02,
        times=np.array([0.01, 0.025]),
        transition_durations=None,
    )


def test_duration_patch_refreshes_stale_emission_builders(monkeypatch) -> None:
    assert getattr(state_space, "_duration_dynamics_patch_applied", False)

    def stale_encoding_builder(*args, **kwargs):
        return _bare_emission()

    def stale_kd_builder(*args, **kwargs):
        return _bare_emission()

    monkeypatch.setattr(encoding, "build_emissions", stale_encoding_builder)
    monkeypatch.setattr(kd_reference, "build_kd_emissions", stale_kd_builder)

    duration_dynamics.apply_duration_dynamics_patch()

    assert getattr(encoding.build_emissions, "_duration_wrapped", False)
    assert getattr(kd_reference.build_kd_emissions, "_duration_wrapped", False)

    encoding_emissions = encoding.build_emissions()
    kd_emissions = kd_reference.build_kd_emissions(None, None, None, 0.02)
    np.testing.assert_allclose(encoding_emissions.transition_durations, np.array([0.015]))
    np.testing.assert_allclose(kd_emissions.transition_durations, np.array([0.015]))
    assert encoding_emissions.dt == 0.02
    assert kd_emissions.dt == 0.02

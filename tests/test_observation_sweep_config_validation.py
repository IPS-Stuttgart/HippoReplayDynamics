from __future__ import annotations

import pytest

from hipporeplayimm import observation_sweep
from hipporeplayimm import observation_sweep_config_validation


def test_observation_sweep_validation_patch_delegates_to_existing_validator(monkeypatch):
    calls = []

    def original_validate_config(config):
        calls.append(config)
        raise RuntimeError("delegated observation-sweep validator called")

    monkeypatch.setattr(observation_sweep, "_validate_config", original_validate_config)
    monkeypatch.setattr(
        observation_sweep,
        observation_sweep_config_validation._PATCHED_FLAG,
        False,
        raising=False,
    )

    observation_sweep_config_validation.apply_observation_sweep_config_validation_patch()

    with pytest.raises(RuntimeError, match="delegated observation-sweep validator called"):
        observation_sweep._validate_config(observation_sweep.ObservationSweepConfig())

    assert len(calls) == 1

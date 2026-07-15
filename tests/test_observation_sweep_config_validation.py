from __future__ import annotations

from decimal import Decimal

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


@pytest.mark.parametrize("field", ["n_folds", "simulation_events_per_model"])
def test_observation_sweep_rejects_large_fractional_integer_controls(field: str) -> None:
    config = observation_sweep.ObservationSweepConfig(
        **{field: Decimal("9007199254740992.5")}
    )

    with pytest.raises(ValueError, match=rf"{field} must be a positive integer"):
        observation_sweep._validate_config(config)


@pytest.mark.parametrize("field", ["n_folds", "simulation_events_per_model"])
def test_observation_sweep_preserves_exact_large_decimal_integer_controls(field: str) -> None:
    value = Decimal(2**53 + 1)
    config = observation_sweep.ObservationSweepConfig(**{field: value})

    assert observation_sweep_config_validation._positive_integer(field, value) == 2**53 + 1
    observation_sweep._validate_config(config)

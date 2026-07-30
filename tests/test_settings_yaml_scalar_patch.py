from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm.observation_sweep as observation_sweep
import hipporeplayimm.result_improvements as result_improvements
import hipporeplayimm.simulation_recovery as simulation_recovery


_SETTINGS_MODULES = (result_improvements, observation_sweep, simulation_recovery)


@pytest.mark.parametrize("module", _SETTINGS_MODULES)
def test_all_settings_writers_quote_yaml_ambiguous_strings(module) -> None:
    assert module._yaml_scalar("null") == '"null"'
    assert module._yaml_scalar("true") == '"true"'
    assert module._yaml_scalar("0123") == '"0123"'
    assert module._yaml_scalar("2026-07-27") == '"2026-07-27"'
    assert module._yaml_scalar("0x10") == '"0x10"'
    assert module._yaml_scalar("0b101") == '"0b101"'
    assert module._yaml_scalar(" leading") == '" leading"'
    assert module._yaml_scalar("plain-value") == "plain-value"


@pytest.mark.parametrize("module", _SETTINGS_MODULES)
def test_all_settings_writers_preserve_numpy_boolean_types(module) -> None:
    assert module._yaml_scalar(np.bool_(True)) == "true"
    assert module._yaml_scalar(np.bool_(False)) == "false"

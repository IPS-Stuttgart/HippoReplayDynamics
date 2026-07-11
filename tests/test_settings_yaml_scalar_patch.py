from __future__ import annotations

import pytest

import hipporeplayimm.observation_sweep as observation_sweep
import hipporeplayimm.result_improvements as result_improvements
import hipporeplayimm.simulation_recovery as simulation_recovery


@pytest.mark.parametrize(
    "module",
    (result_improvements, observation_sweep, simulation_recovery),
)
def test_all_settings_writers_quote_yaml_ambiguous_strings(module) -> None:
    assert module._yaml_scalar("null") == '"null"'
    assert module._yaml_scalar("true") == '"true"'
    assert module._yaml_scalar("0123") == '"0123"'
    assert module._yaml_scalar(" leading") == '" leading"'
    assert module._yaml_scalar("plain-value") == "plain-value"

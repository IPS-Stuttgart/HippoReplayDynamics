from __future__ import annotations

import pytest

import hipporeplayimm
import hipporeplayimm.simulation_recovery as recovery
import hipporeplayimm.simulation_recovery_runtime_limits as validation
from hipporeplayimm.simulation_recovery import SimulationRecoveryConfig


def test_runtime_limit_patch_refreshes_replaced_helpers(monkeypatch) -> None:
    def stale_validator(config):
        return None

    def stale_run(dataset_root, session_id, config):
        return config

    monkeypatch.setattr(recovery, "_validate_recovery_runtime_limits", stale_validator)
    monkeypatch.setattr(recovery, "run_session_simulation_recovery", stale_run)
    monkeypatch.setattr(recovery, validation._PATCHED_FLAG, True, raising=False)

    hipporeplayimm.apply_runtime_patches()

    invalid = SimulationRecoveryConfig(events_per_model=1.5)
    with pytest.raises(ValueError, match="events_per_model"):
        recovery._validate_recovery_runtime_limits(invalid)
    with pytest.raises(ValueError, match="events_per_model"):
        recovery.run_session_simulation_recovery("/unused", "Rat1/Open1", invalid)

    patched_validator = recovery._validate_recovery_runtime_limits
    patched_run = recovery.run_session_simulation_recovery
    assert getattr(patched_validator, validation._VALIDATOR_WRAPPER_ATTR, False)
    assert getattr(patched_run, validation._RUN_WRAPPER_ATTR, False)
    assert patched_validator.__hipporeplayimm_original__ is stale_validator
    assert patched_run.__hipporeplayimm_original__ is stale_run

    normalized = patched_run(
        "/unused",
        "Rat1/Open1",
        SimulationRecoveryConfig(events_per_model="3.0"),
    )
    assert normalized.events_per_model == 3
    assert isinstance(normalized.events_per_model, int)

    hipporeplayimm.apply_runtime_patches()
    assert recovery._validate_recovery_runtime_limits is patched_validator
    assert recovery.run_session_simulation_recovery is patched_run

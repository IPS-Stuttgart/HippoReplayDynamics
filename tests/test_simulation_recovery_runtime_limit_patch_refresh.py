from __future__ import annotations

import pytest

import hipporeplayimm.simulation_recovery as recovery
import hipporeplayimm.simulation_recovery_runtime_limits as validation
from hipporeplayimm.simulation_recovery import SimulationRecoveryConfig


def test_runtime_limit_patch_refreshes_replaced_helpers(monkeypatch) -> None:
    def stale_validator(config) -> None:
        return None

    def stale_run(dataset_root, session_id, config):
        return config

    monkeypatch.setattr(recovery, "_validate_recovery_runtime_limits", stale_validator)
    monkeypatch.setattr(recovery, "run_session_simulation_recovery", stale_run)
    monkeypatch.setattr(recovery, validation._PATCHED_FLAG, True, raising=False)

    validation.apply_simulation_recovery_runtime_limit_validation_patch()

    patched_validator = recovery._validate_recovery_runtime_limits
    patched_run = recovery.run_session_simulation_recovery
    assert getattr(patched_validator, validation._VALIDATOR_ATTR, False)
    assert getattr(patched_run, validation._RUN_WRAPPER_ATTR, False)
    assert getattr(patched_validator, validation._ORIGINAL_ATTR) is stale_validator
    assert getattr(patched_run, validation._ORIGINAL_ATTR) is stale_run

    with pytest.raises(ValueError, match="events_per_model"):
        patched_validator(SimulationRecoveryConfig(events_per_model=0))
    with pytest.raises(ValueError, match="continue_on_error"):
        patched_run(
            "/unused",
            "Rat1/Open1",
            SimulationRecoveryConfig(continue_on_error="false"),
        )

    validation.apply_simulation_recovery_runtime_limit_validation_patch()

    assert recovery._validate_recovery_runtime_limits is patched_validator
    assert recovery.run_session_simulation_recovery is patched_run

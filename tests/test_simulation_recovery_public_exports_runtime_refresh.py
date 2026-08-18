from __future__ import annotations

import importlib

import hipporeplayimm
import hipporeplayimm.simulation_recovery as simulation_recovery


def test_runtime_patches_refresh_public_simulation_recovery_exports_after_reload() -> None:
    stale_config = hipporeplayimm.SimulationRecoveryConfig
    stale_result = hipporeplayimm.SimulationRecoveryResult
    stale_run = hipporeplayimm.run_session_simulation_recovery

    module = importlib.reload(simulation_recovery)

    assert module.SimulationRecoveryConfig is not stale_config
    assert module.SimulationRecoveryResult is not stale_result
    assert module.run_session_simulation_recovery is not stale_run
    assert hipporeplayimm.SimulationRecoveryConfig is stale_config
    assert hipporeplayimm.SimulationRecoveryResult is stale_result
    assert hipporeplayimm.run_session_simulation_recovery is stale_run

    hipporeplayimm.apply_runtime_patches()

    assert hipporeplayimm.SimulationRecoveryConfig is module.SimulationRecoveryConfig
    assert hipporeplayimm.SimulationRecoveryResult is module.SimulationRecoveryResult
    assert hipporeplayimm.run_session_simulation_recovery is module.run_session_simulation_recovery

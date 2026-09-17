from __future__ import annotations

import importlib

import hipporeplayimm
from hipporeplayimm import simulation_recovery


def test_runtime_patches_refresh_public_recovery_exports_after_reload() -> None:
    recovery = importlib.reload(simulation_recovery)
    try:
        # Reload recreates the source-defined classes and entry point, while the
        # package-level aliases still point at the pre-reload objects.
        assert hipporeplayimm.SimulationRecoveryConfig is not recovery.SimulationRecoveryConfig
        assert hipporeplayimm.SimulationRecoveryResult is not recovery.SimulationRecoveryResult
        assert hipporeplayimm.run_session_simulation_recovery is not recovery.run_session_simulation_recovery

        hipporeplayimm.apply_runtime_patches()

        assert hipporeplayimm.SimulationRecoveryConfig is recovery.SimulationRecoveryConfig
        assert hipporeplayimm.SimulationRecoveryResult is recovery.SimulationRecoveryResult
        assert hipporeplayimm.run_session_simulation_recovery is recovery.run_session_simulation_recovery
    finally:
        hipporeplayimm.apply_runtime_patches()

from __future__ import annotations

import hipporeplayimm


_PUBLIC_EXPORTS = (
    "PyRecEstGoalParticleIMMModel",
    "RandomModel",
    "ReplaySession",
    "SimulationRecoveryConfig",
    "SimulationRecoveryResult",
    "run_open_field_benchmark",
    "run_session_simulation_recovery",
    "load_open_field_sessions",
    "score_model",
    "write_pyrecest_sweep_outputs",
)


def test_package_all_includes_public_api_symbols() -> None:
    for name in _PUBLIC_EXPORTS:
        assert hasattr(hipporeplayimm, name)
        assert name in hipporeplayimm.__all__


def test_package_all_symbols_resolve_to_public_attributes() -> None:
    for name in _PUBLIC_EXPORTS:
        assert getattr(hipporeplayimm, name) is not None

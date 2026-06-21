from __future__ import annotations

import hipporeplayimm


def test_public_exports_include_runtime_api_symbols() -> None:
    expected = {
        "ReplaySession",
        "RandomModel",
        "StationaryModel",
        "SimulationRecoveryConfig",
        "SimulationRecoveryResult",
        "run_open_field_benchmark",
        "run_session_simulation_recovery",
        "compare_scores_to_ground_truth",
        "load_open_field_sessions",
        "score_model",
    }

    exported = set(hipporeplayimm.__all__)

    assert expected <= exported
    for name in expected:
        assert hasattr(hipporeplayimm, name)

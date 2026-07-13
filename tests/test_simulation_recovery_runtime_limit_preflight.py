import numpy as np
import pytest

import hipporeplayimm.simulation_recovery as recovery
from hipporeplayimm.simulation_recovery import SimulationRecoveryConfig
from hipporeplayimm.simulation_recovery_runtime_limits import _normalized_runtime_config


def test_runtime_limit_preflight_rejects_before_dataset_resolution(monkeypatch):
    dataset_resolution_attempted = False

    def unexpected_dataset_resolution(*args, **kwargs):
        nonlocal dataset_resolution_attempted
        dataset_resolution_attempted = True
        raise AssertionError("dataset resolution must not run before runtime-limit validation")

    monkeypatch.setattr(recovery, "_session_path", unexpected_dataset_resolution)

    with pytest.raises(ValueError, match="max_runtime_s"):
        recovery.run_session_simulation_recovery(
            "/unused",
            "Rat1/Open1",
            SimulationRecoveryConfig(max_runtime_s=True),
        )

    assert not dataset_resolution_attempted


def test_runtime_limit_preflight_canonicalizes_accepted_scalars():
    normalized = _normalized_runtime_config(
        SimulationRecoveryConfig(
            events_per_model="3.0",
            max_template_events=np.array(2.0),
            max_synthetic_events="4",
            max_runtime_s=np.array(1.25),
            score_with_occupancy=np.bool_(True),
            oracle_candidate_support=np.bool_(False),
            continue_on_error=np.array(False),
            progress_log=np.array(True),
        )
    )

    integer_limits = (
        normalized.events_per_model,
        normalized.max_template_events,
        normalized.max_synthetic_events,
    )
    assert integer_limits == (3, 2, 4)
    assert all(isinstance(value, int) and not isinstance(value, bool) for value in integer_limits)
    assert normalized.max_runtime_s == 1.25
    assert isinstance(normalized.max_runtime_s, float)
    assert normalized.score_with_occupancy is True
    assert normalized.oracle_candidate_support is False
    assert normalized.continue_on_error is False
    assert normalized.progress_log is True

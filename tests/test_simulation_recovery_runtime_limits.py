import numpy as np
import pytest

from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    _validate_recovery_runtime_limits,
)


@pytest.mark.parametrize("bad_limit", [0, -1, 1.5, np.nan, np.inf, True])
def test_simulation_recovery_rejects_invalid_max_synthetic_events(bad_limit):
    with pytest.raises(ValueError, match="max_synthetic_events"):
        _validate_recovery_runtime_limits(
            SimulationRecoveryConfig(max_synthetic_events=bad_limit)
        )


@pytest.mark.parametrize("bad_limit", [0.0, -0.1, np.nan, np.inf, True])
def test_simulation_recovery_rejects_invalid_max_runtime_s(bad_limit):
    with pytest.raises(ValueError, match="max_runtime_s"):
        _validate_recovery_runtime_limits(SimulationRecoveryConfig(max_runtime_s=bad_limit))


def test_simulation_recovery_accepts_valid_runtime_limits():
    _validate_recovery_runtime_limits(
        SimulationRecoveryConfig(max_synthetic_events=3, max_runtime_s=1.25)
    )

import pytest

from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    _validate_recovery_runtime_limits,
)
from hipporeplayimm.simulation_recovery_runtime_limits import _positive_integer_value


def test_runtime_limit_validation_rejects_decimal_string_synthetic_event_limit():
    config = SimulationRecoveryConfig(max_synthetic_events="1.0")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_synthetic_events"):
        _validate_recovery_runtime_limits(config)


def test_runtime_limit_validation_keeps_supported_integer_like_values():
    assert _positive_integer_value("max_synthetic_events", "1") == 1
    assert _positive_integer_value("max_synthetic_events", 1.0) == 1

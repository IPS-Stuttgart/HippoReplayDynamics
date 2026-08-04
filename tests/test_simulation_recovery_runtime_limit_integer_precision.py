from __future__ import annotations

from decimal import Decimal

import pytest

from hipporeplayimm.simulation_recovery import (
    SimulationRecoveryConfig,
    _validate_recovery_runtime_limits,
)
from hipporeplayimm.simulation_recovery_runtime_limits import (
    _normalized_runtime_config,
)


@pytest.mark.parametrize(
    "field_name",
    ["events_per_model", "max_template_events", "max_synthetic_events"],
)
@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.0000000000000000000000000000000001"),
        "9007199254740993.5",
    ],
)
def test_runtime_integer_limits_reject_fractional_values_before_float_conversion(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _validate_recovery_runtime_limits(
            SimulationRecoveryConfig(**{field_name: value})
        )


def test_runtime_integer_limit_preflight_preserves_large_exact_values() -> None:
    above_binary64_precision = 2**53 + 1
    above_binary64_range = 10**400

    normalized = _normalized_runtime_config(
        SimulationRecoveryConfig(
            events_per_model=str(above_binary64_precision),
            max_template_events=Decimal(str(above_binary64_precision)),
            max_synthetic_events=above_binary64_range,
        )
    )

    assert normalized.events_per_model == above_binary64_precision
    assert normalized.max_template_events == above_binary64_precision
    assert normalized.max_synthetic_events == above_binary64_range
    assert isinstance(normalized.events_per_model, int)
    assert isinstance(normalized.max_template_events, int)
    assert isinstance(normalized.max_synthetic_events, int)

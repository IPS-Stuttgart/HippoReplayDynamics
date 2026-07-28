from decimal import Decimal

import numpy as np
import pytest

from hipporeplayimm.latent_path_validation import (
    _config_with_validated_event_counts,
    _integer_valued_scalar,
)
from hipporeplayimm.simulation_recovery import SimulationRecoveryConfig


@pytest.mark.parametrize(
    "value",
    [
        2**53 + 1,
        np.uint64(2**63 + 1),
        Decimal("9007199254740993"),
    ],
)
def test_integer_valued_scalar_preserves_large_integer_precision(value):
    assert _integer_valued_scalar("count", value) == int(value)


def test_simulation_event_count_normalization_preserves_large_integers():
    config = SimulationRecoveryConfig(
        events_per_model=2**53 + 1,
        max_template_events=np.uint64(2**63 + 1),
        max_synthetic_events=Decimal("9007199254740995"),
    )

    normalized = _config_with_validated_event_counts(config)

    assert normalized.events_per_model == 2**53 + 1
    assert normalized.max_template_events == 2**63 + 1
    assert normalized.max_synthetic_events == 9007199254740995


def test_integer_valued_scalar_rejects_fractional_decimal_above_float_precision():
    with pytest.raises(ValueError, match="count must be positive integer-valued"):
        _integer_valued_scalar("count", Decimal("9007199254740993.5"))

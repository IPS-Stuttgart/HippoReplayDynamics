import warnings

import numpy as np
import pytest

from hipporeplayimm.latent_path_validation import (
    _config_with_validated_event_counts,
    _finite_nonnegative_value,
    _integer_valued_scalar,
)
from hipporeplayimm.simulation_recovery import SimulationRecoveryConfig


def _nested_zero_dimensional_object(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


_COMPLEX_SCALARS = (
    np.complex128(3.0 + 0.0j),
    np.clongdouble(3.0 + 0.0j),
    _nested_zero_dimensional_object(np.complex128(3.0 + 0.0j)),
    _nested_zero_dimensional_object(np.clongdouble(3.0 + 2.0j)),
)


@pytest.mark.parametrize("value", _COMPLEX_SCALARS)
def test_integer_valued_scalar_rejects_complex_values_without_coercion(value) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="positive integer-valued, not complex"):
            _integer_valued_scalar("count", value)


@pytest.mark.parametrize(
    "field",
    ["events_per_model", "max_template_events", "max_synthetic_events"],
)
def test_simulation_recovery_config_rejects_complex_count_fields(field) -> None:
    config = SimulationRecoveryConfig(
        **{field: _nested_zero_dimensional_object(np.clongdouble(3.0 + 0.0j))}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=f"{field} must be positive integer-valued"):
            _config_with_validated_event_counts(config)


@pytest.mark.parametrize("value", _COMPLEX_SCALARS)
def test_finite_nonnegative_value_rejects_complex_values_without_coercion(value) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite and nonnegative"):
            _finite_nonnegative_value("sigma", value)


def test_extended_precision_real_scalars_remain_supported() -> None:
    assert _integer_valued_scalar("count", np.longdouble("3.0")) == 3
    assert _finite_nonnegative_value("sigma", np.longdouble("3.5")) == pytest.approx(3.5)

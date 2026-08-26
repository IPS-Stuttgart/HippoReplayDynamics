import numpy as np
import pytest

from hipporeplayimm.advanced_result_place_field_cell_id_validation import (
    _coerce_place_field_numeric_arrays,
)


def test_place_field_rates_reject_boolean_values_before_float_coercion() -> None:
    with pytest.raises(ValueError, match="rates_hz must contain finite nonnegative values"):
        _coerce_place_field_numeric_arrays(
            [[1.0, True], [2.0, 3.0]],
            [1.0, 1.0],
        )


def test_place_field_occupancy_rejects_boolean_values_before_float_coercion() -> None:
    with pytest.raises(ValueError, match="occupancy_s must contain finite nonnegative values"):
        _coerce_place_field_numeric_arrays(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([1.0, False], dtype=object),
        )

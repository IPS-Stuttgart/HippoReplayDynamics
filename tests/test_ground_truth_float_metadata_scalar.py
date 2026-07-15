from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm.ground_truth_float_metadata import _parse_float_metadata_value


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.25], dtype=float),
        np.array([[1.25]], dtype=float),
        [1.25],
    ],
)
def test_ground_truth_float_metadata_rejects_non_scalar_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="finite numeric values"):
        _parse_float_metadata_value("visit_radius_cm", value)


@pytest.mark.parametrize(
    "value",
    [
        "1.25",
        b"1.25",
        np.array(1.25),
    ],
)
def test_ground_truth_float_metadata_keeps_supported_scalar_values(
    value: object,
) -> None:
    assert _parse_float_metadata_value("visit_radius_cm", value) == 1.25

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.spike_rate_metadata import _unique_float_from_columns


def _nested_object_scalar(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "value",
    [
        np.complex128(1.25 + 0.5j),
        np.clongdouble(1.25 + 0.5j),
        True,
        np.bool_(False),
    ],
)
def test_unique_float_rejects_nested_non_real_metadata(value: object) -> None:
    frame = pd.DataFrame(
        {
            "emission_spike_rate_scale": pd.Series(
                [_nested_object_scalar(value)],
                dtype=object,
            )
        }
    )

    with pytest.raises(ValueError, match="finite real numeric metadata values"):
        _unique_float_from_columns(
            frame,
            ("emission_spike_rate_scale",),
            1.0,
        )


def test_unique_float_accepts_nested_real_metadata() -> None:
    frame = pd.DataFrame(
        {
            "emission_spike_rate_scale": pd.Series(
                [_nested_object_scalar(np.float64(1.25))],
                dtype=object,
            )
        }
    )

    value = _unique_float_from_columns(
        frame,
        ("emission_spike_rate_scale",),
        1.0,
    )

    assert value == pytest.approx(1.25)

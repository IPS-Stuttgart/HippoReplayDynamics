from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.spike_rate_metadata import _unique_float_from_columns


@pytest.mark.parametrize(
    "column",
    [
        "emission_time_bin_s",
        "emission_spike_rate_scale",
        "emission_likelihood_temperature",
        "emission_negative_binomial_overdispersion",
    ],
)
def test_spike_rate_metadata_normalizes_numeric_overflow(column: str) -> None:
    frame = pd.DataFrame({column: pd.Series([10**400], dtype=object)})

    with pytest.raises(
        ValueError,
        match=rf"{column} must contain finite numeric metadata values",
    ):
        _unique_float_from_columns(frame, (column,), default=1.0)

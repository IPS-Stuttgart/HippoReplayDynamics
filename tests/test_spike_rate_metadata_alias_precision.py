from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.spike_rate_metadata import _unique_float_from_columns


def test_spike_rate_metadata_rejects_distinct_float64_aliases_within_legacy_tolerance() -> None:
    scores = pd.DataFrame(
        {
            "emission_spike_rate_scale": [1.0],
            "spike_rate_scale": [1.000005],
        }
    )

    with pytest.raises(ValueError, match="contains multiple values"):
        _unique_float_from_columns(
            scores,
            ("emission_spike_rate_scale", "spike_rate_scale"),
            default=1.0,
        )

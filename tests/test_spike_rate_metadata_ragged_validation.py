from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.spike_rate_metadata import _unique_float_from_columns


def test_unique_float_rejects_ragged_metadata_with_field_context() -> None:
    frame = pd.DataFrame(
        {
            "emission_time_bin_s": pd.Series(
                [[1.0, [2.0, 3.0]]],
                dtype=object,
            )
        }
    )

    with pytest.raises(
        ValueError,
        match="emission_time_bin_s.*scalar numeric",
    ):
        _unique_float_from_columns(
            frame,
            ("emission_time_bin_s",),
            default=0.02,
        )

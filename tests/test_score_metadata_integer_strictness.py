from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.score_metadata import _unique_int_from_columns


@pytest.mark.parametrize("value", ["128.0000000001", "127.9999999999"])
def test_score_metadata_rejects_near_integral_integer_metadata(value: str) -> None:
    scores = pd.DataFrame({"state_space_momentum_candidate_top_k": [value]})

    with pytest.raises(ValueError, match="must be an integer"):
        _unique_int_from_columns(
            scores,
            ("state_space_momentum_candidate_top_k",),
            default=128,
        )

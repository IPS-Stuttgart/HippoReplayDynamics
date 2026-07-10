from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from select_momentum_confidence_threshold import (  # noqa: E402
    _constant_thresholds_for_strata,
    _threshold_for_key,
)


def test_threshold_lookup_round_trips_missing_stratum_value() -> None:
    scores = pd.DataFrame(
        {
            "matrix_id": [np.nan, np.nan],
            "event_index": [0, 1],
        }
    )

    thresholds = _constant_thresholds_for_strata(scores, ("matrix_id",), 4.0)

    assert _threshold_for_key(thresholds, ("matrix_id",), (np.nan,)) == 4.0


def test_threshold_lookup_matches_missing_value_in_composite_key() -> None:
    thresholds = pd.DataFrame(
        {
            "emission_likelihood_temperature": [0.5, 0.5],
            "matrix_id": [pd.NA, "known"],
            "selected_margin_threshold": [4.0, 6.0],
        }
    )

    selected = _threshold_for_key(
        thresholds,
        ("emission_likelihood_temperature", "matrix_id"),
        (0.5, pd.NA),
    )

    assert selected == 4.0

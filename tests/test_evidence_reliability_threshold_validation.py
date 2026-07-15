from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.evidence_reliability import (
    add_event_reliability_flags,
    event_reliability_flags,
)


_SCORE_ROW = pd.Series(
    {
        "status": "success",
        "n_spikes": 4,
        "n_time": 3,
        "mean_candidate_log_mass": -10.0,
        "terminal_posterior_entropy": 1.0,
    }
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_spikes": True},
        {"min_spikes": -1},
        {"min_spikes": 1.5},
        {"min_time_bins": np.array([2])},
        {"min_candidate_log_mass": np.nan},
        {"max_terminal_entropy": np.nan},
        {"max_terminal_entropy": -1.0},
    ],
)
def test_event_reliability_rejects_malformed_thresholds(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        event_reliability_flags(_SCORE_ROW, **kwargs)


def test_add_event_reliability_validates_thresholds_for_empty_frames() -> None:
    with pytest.raises(ValueError, match="min_candidate_log_mass"):
        add_event_reliability_flags(
            pd.DataFrame(),
            min_candidate_log_mass=np.nan,
        )


def test_event_reliability_accepts_numpy_scalars_and_infinite_disable_thresholds() -> None:
    flags = event_reliability_flags(
        _SCORE_ROW,
        min_spikes=np.int64(3),
        min_time_bins=np.array(2),
        min_candidate_log_mass=-np.inf,
        max_terminal_entropy=np.inf,
    )

    assert flags["event_reliable"]

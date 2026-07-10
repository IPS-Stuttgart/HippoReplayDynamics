from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def _score_with_n_spikes(value: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": value,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )


@pytest.mark.parametrize(
    "value",
    [
        "not-a-number",
        b"not-a-number",
        object(),
    ],
)
def test_reliability_flags_mark_unparseable_numeric_metrics_malformed(value: object) -> None:
    flagged = add_event_reliability_flags(_score_with_n_spikes(value))

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, "event_low_spike_count"])
    assert flagged.loc[0, "event_reliability_reasons"] == "invalid_numeric_metric"


@pytest.mark.parametrize(
    "value",
    [
        4.0 + 1.0j,
        np.complex128(4.0 + 1.0j),
        np.asarray(4.0 + 1.0j),
        np.array(np.complex128(4.0 + 1.0j), dtype=object),
    ],
)
def test_reliability_flags_reject_complex_numeric_metrics_without_lossy_coercion(value: object) -> None:
    flagged = add_event_reliability_flags(_score_with_n_spikes(value))

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, "event_low_spike_count"])
    assert flagged.loc[0, "event_reliability_reasons"] == "invalid_numeric_metric"


def test_reliability_flags_keep_parseable_numeric_strings_supported() -> None:
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": "4",
                "n_time": "3",
                "mean_candidate_log_mass": "0.0",
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert bool(flagged.loc[0, "event_reliable"])
    assert not bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert flagged.loc[0, "event_reliability_reasons"] == ""

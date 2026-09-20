from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.shuffle_controls import add_shuffle_p_values


def test_shuffle_p_values_keep_negative_infinite_control_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["s"],
            "event_index": [0],
            "model": ["m"],
            "log_evidence": [0.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["s", "s"],
            "event_index": [0, 0],
            "model": ["m", "m"],
            "log_evidence": [float("-inf"), 1.0],
        }
    )

    result = add_shuffle_p_values(real_scores, control_scores)

    row = result.iloc[0]
    assert row["shuffle_count"] == 2
    assert row["shuffle_p_value"] == pytest.approx(2.0 / 3.0)


def test_shuffle_p_values_accept_negative_infinite_real_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["s"],
            "event_index": [0],
            "model": ["m"],
            "log_evidence": [float("-inf")],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["s", "s"],
            "event_index": [0, 0],
            "model": ["m", "m"],
            "log_evidence": [float("-inf"), 0.0],
        }
    )

    result = add_shuffle_p_values(real_scores, control_scores)

    row = result.iloc[0]
    assert row["shuffle_count"] == 2
    assert row["shuffle_p_value"] == pytest.approx(1.0)
    assert np.isneginf(row["log_evidence"])


def test_shuffle_p_values_still_reject_positive_infinite_evidence() -> None:
    real_scores = pd.DataFrame(
        {
            "session": ["s"],
            "event_index": [0],
            "model": ["m"],
            "log_evidence": [0.0],
        }
    )
    control_scores = pd.DataFrame(
        {
            "session": ["s", "s"],
            "event_index": [0, 0],
            "model": ["m", "m"],
            "log_evidence": [float("inf"), 1.0],
        }
    )

    result = add_shuffle_p_values(real_scores, control_scores)

    row = result.iloc[0]
    assert row["shuffle_count"] == 1
    assert row["shuffle_p_value"] == pytest.approx(1.0)

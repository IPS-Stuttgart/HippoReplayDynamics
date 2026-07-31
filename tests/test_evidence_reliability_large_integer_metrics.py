from __future__ import annotations

from decimal import Decimal

import pandas as pd

from hipporeplayimm.evidence_reliability import add_event_reliability_flags


def test_reliability_flags_keep_arbitrary_size_integer_counts_valid() -> None:
    huge_count = 10**400
    scores = pd.DataFrame(
        {
            "model": pd.Series(["diffusion"], dtype=object),
            "status": pd.Series(["success"], dtype=object),
            "n_spikes": pd.Series([huge_count], dtype=object),
            "n_time": pd.Series([huge_count], dtype=object),
            "mean_candidate_log_mass": pd.Series([0.0], dtype=object),
        }
    )

    flagged = add_event_reliability_flags(scores)

    assert bool(flagged.loc[0, "event_reliable"])
    assert not bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, "event_low_spike_count"])
    assert not bool(flagged.loc[0, "event_too_few_time_bins"])
    assert flagged.loc[0, "event_reliability_reasons"] == ""


def test_reliability_flags_reject_fractional_counts_before_float_rounding() -> None:
    rounded_fraction = "9007199254740993.5"
    scores = pd.DataFrame(
        {
            "model": pd.Series(["diffusion", "diffusion"], dtype=object),
            "status": pd.Series(["success", "success"], dtype=object),
            "n_spikes": pd.Series([Decimal(rounded_fraction), 5], dtype=object),
            "n_time": pd.Series([3, rounded_fraction], dtype=object),
            "mean_candidate_log_mass": pd.Series([0.0, 0.0], dtype=object),
        }
    )

    flagged = add_event_reliability_flags(scores)

    assert flagged["event_invalid_numeric_metric"].tolist() == [True, True]
    assert flagged["event_reliable"].tolist() == [False, False]
    assert flagged["event_reliability_reasons"].tolist() == [
        "invalid_numeric_metric",
        "invalid_numeric_metric",
    ]


def test_reliability_flags_keep_exact_large_decimal_counts_valid() -> None:
    scores = pd.DataFrame(
        {
            "model": pd.Series(["diffusion"], dtype=object),
            "status": pd.Series(["success"], dtype=object),
            "n_spikes": pd.Series([Decimal("9007199254740993")], dtype=object),
            "n_time": pd.Series(["9.007199254740993e15"], dtype=object),
            "mean_candidate_log_mass": pd.Series([0.0], dtype=object),
        }
    )

    flagged = add_event_reliability_flags(scores)

    assert bool(flagged.loc[0, "event_reliable"])
    assert not bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert flagged.loc[0, "event_reliability_reasons"] == ""

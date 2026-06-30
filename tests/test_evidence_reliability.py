from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reliability import add_event_reliability_flags
from hipporeplayimm.evidence_reporting import DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT


def _valid_score_rows(status_values: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": f"model-{index}",
                "status": status,
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
            }
            for index, status in enumerate(status_values)
        ]
    )


def test_add_event_reliability_flags_replaces_existing_columns_on_rerun():
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )

    once = add_event_reliability_flags(scores)
    assert list(once.columns).count("event_reliable") == 1

    # Simulate rerunning the augmentation on an already augmented CSV.  The
    # stale flag value must be replaced rather than kept beside a duplicate
    # event_reliable column, otherwise downstream pandas aggregations can fail
    # or read the wrong column shape.
    once["event_reliable"] = False
    twice = add_event_reliability_flags(once)

    assert list(twice.columns).count("event_reliable") == 1
    assert bool(twice.loc[0, "event_reliable"])

    reliability = twice.groupby("model", as_index=False).agg(
        reliable_rows=("event_reliable", "sum"),
    )
    assert int(reliability.loc[0, "reliable_rows"]) == 1


def test_add_event_reliability_flags_treats_legacy_missing_status_as_success():
    scores = _valid_score_rows(["", " ", None, pd.NA, float("nan")])

    flagged = add_event_reliability_flags(scores)

    assert flagged["event_reliable"].tolist() == [True] * len(scores)
    assert flagged["event_reliability_reasons"].tolist() == [""] * len(scores)


def test_add_event_reliability_flags_keeps_explicit_failed_status_unreliable():
    scores = _valid_score_rows(["failure", "unsupported"])

    flagged = add_event_reliability_flags(scores)

    assert flagged["event_reliable"].tolist() == [False, False]
    assert flagged["event_reliability_reasons"].tolist() == ["score_failure", "score_failure"]


def test_add_event_reliability_flags_falls_back_to_test_spikes_when_n_spikes_missing():
    scores = pd.DataFrame(
        [
            {
                "model": "heldout",
                "status": "success",
                "n_spikes": float("nan"),
                "test_spikes": 1,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_low_spike_count"])
    assert flagged.loc[0, "event_reliability_reasons"] == "low_spike_count"


def test_add_event_reliability_flags_infers_degenerate_support_from_diagnostics():
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": 4,
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
                "diagnostic_candidate_evidence_support": DEGENERATE_SINGLE_BIN_EVIDENCE_SUPPORT,
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert flagged.loc[0, "event_reliability_reasons"] == "degenerate_single_bin"


def test_add_event_reliability_flags_marks_array_shaped_numeric_metrics_malformed():
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": np.array([4]),
                "n_time": 3,
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, "event_low_spike_count"])
    assert flagged.loc[0, "event_reliability_reasons"] == "invalid_numeric_metric"


def test_add_event_reliability_flags_marks_boolean_numeric_metrics_malformed():
    scores = pd.DataFrame(
        [
            {
                "model": "diffusion",
                "status": "success",
                "n_spikes": 4,
                "n_time": np.array(True, dtype=object),
                "mean_candidate_log_mass": 0.0,
            }
        ]
    )

    flagged = add_event_reliability_flags(scores)

    assert not bool(flagged.loc[0, "event_reliable"])
    assert bool(flagged.loc[0, "event_invalid_numeric_metric"])
    assert not bool(flagged.loc[0, "event_too_few_time_bins"])
    assert flagged.loc[0, "event_reliability_reasons"] == "invalid_numeric_metric"

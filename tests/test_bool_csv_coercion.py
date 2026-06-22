from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.evidence_reporting import simulation_event_best_rows
from hipporeplayimm.result_quality_gates import _candidate_good_fraction, event_quality_summary


def test_event_quality_summary_treats_string_false_as_false() -> None:
    frame = pd.DataFrame(
        {
            "session": ["session-a", "session-a"],
            "event_index": [0, 0],
            "model": ["model-a", "model-b"],
            "status": ["success", "success"],
            "log_evidence": [0.0, -1.0],
            "diagnostic_candidate_evidence_support": ["exact_full_grid", "exact_full_grid"],
            "event_reliable": ["False", "0"],
        }
    )

    summary = event_quality_summary(frame)

    assert summary.loc[0, "event_reliable_fraction"] == pytest.approx(0.0)


def test_event_quality_summary_treats_blank_status_as_legacy_success() -> None:
    frame = pd.DataFrame(
        {
            "session": ["session-a", "session-a", "session-a"],
            "event_index": [0, 0, 0],
            "model": ["missing-status", "blank-status", "failed-status"],
            "status": [pd.NA, "", "failed"],
            "log_evidence": [3.0, 1.0, 100.0],
            "diagnostic_candidate_evidence_support": [
                "exact_full_grid",
                "exact_full_grid",
                "exact_full_grid",
            ],
        }
    )

    summary = event_quality_summary(frame)

    assert summary.loc[0, "successful_rows"] == 2
    assert summary.loc[0, "exact_comparable_models"] == 2
    assert summary.loc[0, "exact_best_model"] == "missing-status"
    assert summary.loc[0, "exact_log_evidence_margin"] == pytest.approx(2.0)


def test_candidate_good_fraction_treats_string_false_as_false() -> None:
    frame = pd.DataFrame(
        {
            "candidate_support_quality_good": ["False", "True"],
        }
    )

    assert _candidate_good_fraction(frame) == pytest.approx(0.5)


def test_simulation_event_best_rows_ignores_string_false_best_flags() -> None:
    event_scores = pd.DataFrame(
        {
            "session": ["session-a", "session-a"],
            "event_index": [0, 0],
            "model": ["low-evidence", "high-evidence"],
            "status": ["success", "success"],
            "log_evidence": [0.0, 3.0],
            "diagnostic_candidate_evidence_support": ["exact_full_grid", "exact_full_grid"],
            "is_best_model": ["False", "False"],
        }
    )

    best = simulation_event_best_rows(event_scores)

    assert best["model"].tolist() == ["high-evidence"]


def test_simulation_event_best_rows_honors_string_true_best_flag() -> None:
    event_scores = pd.DataFrame(
        {
            "session": ["session-a", "session-a"],
            "event_index": [0, 0],
            "model": ["explicit-best", "higher-evidence"],
            "status": ["success", "success"],
            "log_evidence": [0.0, 3.0],
            "diagnostic_candidate_evidence_support": ["exact_full_grid", "exact_full_grid"],
            "is_best_model": ["True", "False"],
        }
    )

    best = simulation_event_best_rows(event_scores)

    assert best["model"].tolist() == ["explicit-best"]

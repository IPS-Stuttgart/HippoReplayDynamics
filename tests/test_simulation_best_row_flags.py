from __future__ import annotations

import pandas as pd

from hipporeplayimm.evidence_reporting import simulation_event_best_rows


def _row(event_index: int, model: str, log_evidence: float, is_best_model: str) -> dict[str, object]:
    return {
        "session": "session-a",
        "event_index": event_index,
        "model": model,
        "status": "success",
        "log_evidence": log_evidence,
        "diagnostic_candidate_evidence_support": "exact_full_grid",
        "is_best_model": is_best_model,
    }


def test_simulation_event_best_rows_handles_empty_score_frame() -> None:
    best = simulation_event_best_rows(pd.DataFrame())

    assert best.empty


def test_simulation_event_best_rows_recomputes_unflagged_events() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "explicit-best", 0.0, "True"),
            _row(0, "higher-but-not-flagged", 3.0, "False"),
            _row(1, "event-one-low", 0.0, "False"),
            _row(1, "event-one-high", 2.0, "False"),
        ]
    )

    best = simulation_event_best_rows(rows).sort_values("event_index").reset_index(drop=True)

    assert best["model"].tolist() == ["explicit-best", "event-one-high"]


def test_simulation_event_best_rows_recomputes_duplicate_flags() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "stale-flag-a", 0.0, "True"),
            _row(0, "stale-flag-b", 1.0, "True"),
            _row(0, "true-highest", 4.0, "False"),
        ]
    )

    best = simulation_event_best_rows(rows)

    assert best["model"].tolist() == ["true-highest"]


def test_simulation_event_best_rows_ignores_nonfinite_flagged_rows() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "stale-nan-flag", float("nan"), "True"),
            _row(0, "finite-highest", 4.0, "False"),
        ]
    )

    best = simulation_event_best_rows(rows)

    assert best["model"].tolist() == ["finite-highest"]
    assert best["log_evidence"].tolist() == [4.0]


def test_simulation_event_best_rows_treats_missing_status_as_success() -> None:
    rows = pd.DataFrame(
        [
            _row(0, "legacy-low", 0.0, "False"),
            _row(0, "legacy-high", 5.0, "False"),
        ]
    )
    rows["status"] = [pd.NA, float("nan")]

    best = simulation_event_best_rows(rows)

    assert best["model"].tolist() == ["legacy-high"]

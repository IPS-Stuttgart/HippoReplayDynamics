import numpy as np
import pandas as pd

from hipporeplayimm.evidence_reporting import (
    ensure_evidence_support_columns,
    simulation_add_evidence_columns,
    simulation_event_best_rows,
)


def _row(model: str, log_evidence: float, status: object) -> dict[str, object]:
    return {
        "session": "RatX/Open1",
        "event_index": 0,
        "model": model,
        "status": status,
        "log_evidence": log_evidence,
        "diagnostic_candidate_evidence_support": "exact_full_grid",
    }


def test_missing_status_remains_exact_comparable_for_legacy_scores() -> None:
    rows = pd.DataFrame(
        [
            _row("legacy-low", 0.0, ""),
            _row("legacy-high", 5.0, np.nan),
        ]
    )

    supported = ensure_evidence_support_columns(rows)
    assert supported["evidence_comparable"].tolist() == [True, True]

    scored = simulation_add_evidence_columns(rows)
    assert scored["best_model"].unique().tolist() == ["legacy-high"]
    assert scored.loc[scored["model"] == "legacy-high", "is_best_model"].item() is True

    best = simulation_event_best_rows(rows)
    assert best["model"].tolist() == ["legacy-high"]


def test_non_success_status_remains_excluded_after_status_normalization() -> None:
    rows = pd.DataFrame(
        [
            _row("failed-high", 10.0, "failure"),
            _row("success-low", 0.0, " Success "),
        ]
    )

    supported = ensure_evidence_support_columns(rows)
    assert supported.loc[supported["model"] == "failed-high", "evidence_comparable"].item() is False
    assert supported.loc[supported["model"] == "success-low", "evidence_comparable"].item() is True

    scored = simulation_add_evidence_columns(rows)
    assert scored["best_model"].unique().tolist() == ["success-low"]

    best = simulation_event_best_rows(rows)
    assert best["model"].tolist() == ["success-low"]

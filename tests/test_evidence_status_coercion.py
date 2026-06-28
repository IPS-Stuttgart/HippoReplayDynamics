import importlib

import numpy as np
import pandas as pd

import hipporeplayimm
import hipporeplayimm.evidence_reporting as reporting
from hipporeplayimm.evidence_reporting import (
    ensure_evidence_support_columns,
    simulation_add_evidence_columns,
    simulation_event_best_rows,
)


def _row(model: str, log_evidence: float, status: object, *, event_index: int = 0) -> dict[str, object]:
    return {
        "session": "RatX/Open1",
        "event_index": event_index,
        "model": model,
        "status": status,
        "log_evidence": log_evidence,
        "diagnostic_candidate_evidence_support": "exact_full_grid",
    }


def test_reloaded_core_evidence_reporting_keeps_legacy_status_semantics() -> None:
    reloaded = importlib.reload(reporting)
    try:
        rows = pd.DataFrame(
            [
                _row("status-success", 0.0, " Success "),
                _row("status-missing", 5.0, np.nan),
                _row("status-failed", 10.0, "failure"),
            ]
        )

        supported = reloaded.ensure_evidence_support_columns(rows)
        assert supported.loc[supported["model"].eq("status-success"), "evidence_comparable"].item() is True
        assert supported.loc[supported["model"].eq("status-missing"), "evidence_comparable"].item() is True
        assert supported.loc[supported["model"].eq("status-failed"), "evidence_comparable"].item() is False

        scored = reloaded.simulation_add_evidence_columns(rows)
        assert scored["best_model"].unique().tolist() == ["status-missing"]

        best = reloaded.simulation_event_best_rows(rows)
        assert best["model"].tolist() == ["status-missing"]
    finally:
        hipporeplayimm.apply_runtime_patches()


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


def test_nonfinite_log_evidence_is_not_exact_comparable_or_event_best() -> None:
    rows = pd.DataFrame(
        [
            _row("nan-exact", np.nan, "success"),
            _row("finite-exact", -2.0, "success"),
            _row("only-nan-exact", np.nan, "success", event_index=1),
        ]
    )

    supported = ensure_evidence_support_columns(rows)

    nan_rows = supported[supported["model"].str.contains("nan")]
    assert not nan_rows["evidence_comparable"].any()
    assert supported.loc[supported["model"] == "finite-exact", "evidence_comparable"].item() is True

    best = simulation_event_best_rows(supported)
    assert best[["event_index", "model"]].values.tolist() == [[0, "finite-exact"]]


def test_missing_support_with_explicit_false_comparable_stays_noncomparable() -> None:
    rows = pd.DataFrame(
        [
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "model": "exact-low",
                "status": "success",
                "log_evidence": 0.0,
                "evidence_support": "exact_full_grid",
                "evidence_comparable": True,
            },
            {
                "session": "RatX/Open1",
                "event_index": 0,
                "model": "legacy-noncomparable-high",
                "status": "success",
                "log_evidence": 100.0,
                "evidence_support": "",
                "evidence_comparable": "False",
            },
        ]
    )

    supported = ensure_evidence_support_columns(rows)
    legacy = supported[supported["model"] == "legacy-noncomparable-high"].iloc[0]
    assert not bool(legacy["evidence_comparable"])
    assert legacy["evidence_support"] == "unknown_noncomparable"
    assert legacy["evidence_comparison"] == "unknown_noncomparable"

    scored = simulation_add_evidence_columns(rows)
    assert scored["best_model"].unique().tolist() == ["exact-low"]
    assert scored.loc[scored["model"] == "exact-low", "is_best_model"].item() is True
    assert pd.isna(scored.loc[scored["model"] == "legacy-noncomparable-high", "model_probability"].item())

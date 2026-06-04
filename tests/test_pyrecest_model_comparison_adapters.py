from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    paired_model_margin_decisions,
    paired_model_margin_summary,
    paired_model_margin_threshold_sweep,
    select_paired_model_margin_threshold,
)


def test_advanced_result_diagnostics_uses_pyrecest_paired_margin_helpers():
    scores = pd.DataFrame(
        [
            _score("Rat1/Open1", 0, "positive", 9.0, true_model="positive"),
            _score("Rat1/Open1", 0, "reference", 1.0, true_model="positive"),
            _score("Rat1/Open1", 1, "positive", 3.0, true_model="reference"),
            _score("Rat1/Open1", 1, "reference", 6.0, true_model="reference"),
        ]
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="positive",
        reference_model="reference",
        margin_threshold=2.0,
        true_model_col="true_model",
        positive_true_label="positive",
    )
    assert decisions["margin_decision"].tolist() == ["positive", "reference"]

    summary = paired_model_margin_summary(decisions, true_model_col="true_model")
    assert int(summary.iloc[0]["positive_model_claims"]) == 1
    assert int(summary.iloc[0]["false_positive_claims"]) == 0

    sweep = paired_model_margin_threshold_sweep(
        scores,
        positive_model="positive",
        reference_model="reference",
        thresholds=(0.0, 2.0, 5.0),
        group_cols=("session", "event_index"),
        true_model_col="true_model",
        positive_true_label="positive",
    )
    selected = select_paired_model_margin_threshold(sweep, max_false_positive_claims=0)
    assert selected.iloc[0]["selection_status"] == "passed_specificity_gate"


def test_all_session_aggregate_uses_pyrecest_grouped_and_bootstrap_summaries():
    module = _load_aggregate_module()
    decisions = pd.DataFrame(
        [
            _decision("Rat1/Open1", 0, 5.0, True),
            _decision("Rat1/Open2", 1, 4.0, True),
            _decision("Rat2/Open1", 2, -1.0, False),
            _decision("Rat2/Open2", 3, 6.0, True),
        ]
    )

    session = module._paired_margin_summary(decisions, group_cols=("session",))
    assert session["session"].tolist() == ["Rat1/Open1", "Rat1/Open2", "Rat2/Open1", "Rat2/Open2"]
    assert session["positive_model_claims"].sum() == 3

    leave_one = module._leave_one_rat_out_summary(
        decisions,
        lambda frame: module._paired_margin_summary(frame, group_cols=()),
    )
    assert set(leave_one["held_out_rat"]) == {"Rat1", "Rat2"}
    assert (leave_one["events"] == 2).all()

    bootstrap = module._rat_bootstrap_margin_summary(
        decisions,
        delta_col="positive_minus_reference_log_evidence",
        positive_claim_col="positive_model_claimed",
        n_bootstrap=50,
        random_seed=1,
    )
    assert bootstrap.iloc[0]["bootstrap_unit"] == "rat"
    assert int(bootstrap.iloc[0]["observed_rats"]) == 2
    assert np.isfinite(float(bootstrap.iloc[0]["observed_mean_delta"]))


def _load_aggregate_module():
    scripts = str(Path("scripts").resolve())
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = Path("scripts/aggregate_all_session_model_evidence.py")
    spec = importlib.util.spec_from_file_location("aggregate_all_session_model_evidence_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(session: str, event_index: int, model: str, log_evidence: float, *, true_model: str) -> dict[str, object]:
    return {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": True,
        "true_model": true_model,
    }


def _decision(session: str, event_index: int, delta: float, positive_claimed: bool) -> dict[str, object]:
    return {
        "session": session,
        "event_index": event_index,
        "positive_model": "positive",
        "reference_model": "reference",
        "positive_minus_reference_log_evidence": float(delta),
        "positive_model_claimed": bool(positive_claimed),
        "margin_decision": "positive" if positive_claimed else "ambiguous",
        "margin_threshold": 2.0,
    }

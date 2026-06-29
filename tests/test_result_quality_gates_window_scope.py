from __future__ import annotations

import pandas as pd
import pytest

from hipporeplayimm.result_quality_gates import (
    add_evidence_margin_columns,
    event_group_columns,
    event_quality_summary,
)


def _score_row(*, start_s: float, end_s: float, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 10,
        "window_role": "matched_null",
        "window_start_s": float(start_s),
        "window_end_s": float(end_s),
        "window_duration_s": float(end_s - start_s),
        "model": str(model),
        "log_evidence": float(log_evidence),
        "evidence_comparable": True,
        "evidence_support": "exact_full_grid",
    }


def _cell_split_score_row(*, split_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 10,
        "cell_split_index": int(split_index),
        "cell_split_seed": 100 + int(split_index),
        "test_cell_fraction": 0.5,
        "model": str(model),
        "log_evidence": float(log_evidence),
        "evidence_comparable": True,
        "evidence_support": "exact_full_grid",
    }


def test_result_quality_gates_scope_explicit_window_time_metadata() -> None:
    scores = pd.DataFrame(
        [
            _score_row(start_s=1.0, end_s=1.05, model="stationary", log_evidence=0.0),
            _score_row(start_s=1.0, end_s=1.05, model="diffusion", log_evidence=5.0),
            _score_row(start_s=2.0, end_s=2.05, model="stationary", log_evidence=10.0),
            _score_row(start_s=2.0, end_s=2.05, model="diffusion", log_evidence=6.0),
        ]
    )

    columns = event_group_columns(scores)

    assert columns[:2] == ["session", "event_index"]
    assert "window_start_s" in columns
    assert "window_end_s" in columns
    assert "window_duration_s" in columns

    with_margins = add_evidence_margin_columns(scores)
    summary = event_quality_summary(scores).sort_values("window_start_s").reset_index(drop=True)

    assert with_margins["exact_model_best_model"].tolist() == [
        "diffusion",
        "diffusion",
        "stationary",
        "stationary",
    ]
    assert summary["window_start_s"].tolist() == [1.0, 2.0]
    assert summary["exact_best_model"].tolist() == ["diffusion", "stationary"]
    assert summary["exact_log_evidence_margin"].tolist() == pytest.approx([5.0, 4.0])


def test_result_quality_gates_scope_cell_split_heldout_metadata() -> None:
    scores = pd.DataFrame(
        [
            _cell_split_score_row(split_index=0, model="stationary", log_evidence=0.0),
            _cell_split_score_row(split_index=0, model="diffusion", log_evidence=5.0),
            _cell_split_score_row(split_index=1, model="stationary", log_evidence=10.0),
            _cell_split_score_row(split_index=1, model="diffusion", log_evidence=6.0),
        ]
    )

    columns = event_group_columns(scores)

    assert columns[:2] == ["session", "event_index"]
    assert "cell_split_index" in columns
    assert "cell_split_seed" in columns
    assert "test_cell_fraction" in columns

    with_margins = add_evidence_margin_columns(scores)
    summary = event_quality_summary(scores).sort_values("cell_split_index").reset_index(drop=True)

    assert with_margins["exact_model_best_model"].tolist() == [
        "diffusion",
        "diffusion",
        "stationary",
        "stationary",
    ]
    assert summary["cell_split_index"].astype(int).tolist() == [0, 1]
    assert summary["exact_best_model"].tolist() == ["diffusion", "stationary"]
    assert summary["exact_log_evidence_margin"].tolist() == pytest.approx([5.0, 4.0])

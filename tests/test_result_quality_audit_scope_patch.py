from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_audit import event_group_columns, write_result_quality_audit


def _score_row(*, null_index: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "Rat1/Open1",
        "event_index": 10,
        "window_role": "matched_null",
        "null_index": int(null_index),
        "model": str(model),
        "model_family": "trajectory" if model == "diffusion" else "nontrajectory",
        "log_evidence": float(log_evidence),
        "n_time": 3,
        "n_spikes": 5,
        "runtime_s": 0.01,
        "error": "",
        "evidence_comparable": True,
        "evidence_support": "exact_full_grid",
        "relative_log_evidence": float(log_evidence),
    }


def test_result_quality_audit_group_columns_include_matched_null_scope() -> None:
    scores = pd.DataFrame(
        [
            _score_row(null_index=0, model="stationary", log_evidence=0.0),
            _score_row(null_index=0, model="diffusion", log_evidence=5.0),
        ]
    )

    columns = event_group_columns(scores)

    assert columns[:2] == ["session", "event_index"]
    assert "window_role" in columns
    assert "null_index" in columns


def test_result_quality_audit_margins_do_not_mix_matched_null_windows(tmp_path) -> None:
    scores = pd.DataFrame(
        [
            _score_row(null_index=0, model="stationary", log_evidence=0.0),
            _score_row(null_index=0, model="diffusion", log_evidence=5.0),
            _score_row(null_index=1, model="stationary", log_evidence=10.0),
            _score_row(null_index=1, model="diffusion", log_evidence=6.0),
        ]
    )

    write_result_quality_audit(scores, tmp_path)

    margins = pd.read_csv(tmp_path / "evidence_margins.csv").sort_values("null_index").reset_index(drop=True)
    assert margins["null_index"].astype(int).tolist() == [0, 1]
    assert margins["best_model_by_evidence"].tolist() == ["diffusion", "stationary"]
    assert margins["evidence_margin_to_second_best"].tolist() == [5.0, 4.0]

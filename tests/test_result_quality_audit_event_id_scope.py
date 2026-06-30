from __future__ import annotations

import pandas as pd

from hipporeplayimm.result_quality_audit import event_group_columns, write_result_quality_audit


def _score_row(*, event_id: int, model: str, log_evidence: float) -> dict[str, object]:
    return {
        "status": "success",
        "session": "RatX/OpenY",
        "event_id": int(event_id),
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


def test_result_quality_audit_uses_event_id_when_event_index_is_missing() -> None:
    scores = pd.DataFrame(
        [
            _score_row(event_id=0, model="stationary", log_evidence=0.0),
            _score_row(event_id=0, model="diffusion", log_evidence=5.0),
        ]
    )

    assert event_group_columns(scores) == ["session", "event_id"]


def test_result_quality_audit_margins_do_not_mix_event_id_only_tables(tmp_path) -> None:
    scores = pd.DataFrame(
        [
            _score_row(event_id=0, model="stationary", log_evidence=0.0),
            _score_row(event_id=0, model="diffusion", log_evidence=5.0),
            _score_row(event_id=1, model="stationary", log_evidence=10.0),
            _score_row(event_id=1, model="diffusion", log_evidence=6.0),
        ]
    )

    dashboard = write_result_quality_audit(scores, tmp_path)

    margins = pd.read_csv(tmp_path / "evidence_margins.csv").sort_values("event_id").reset_index(drop=True)
    assert margins["event_id"].astype(int).tolist() == [0, 1]
    assert margins["best_model_by_evidence"].tolist() == ["diffusion", "stationary"]
    assert margins["evidence_margin_to_second_best"].tolist() == [5.0, 4.0]
    assert "Events: 2\n" in dashboard.read_text(encoding="utf-8")


def test_result_quality_audit_prefers_event_index_over_event_id() -> None:
    scores = pd.DataFrame(
        {
            "session": ["RatX/OpenY", "RatX/OpenY"],
            "event_index": [4, 4],
            "event_id": [99, 100],
            "model": ["diffusion", "stationary"],
            "model_family": ["trajectory", "nontrajectory"],
            "log_evidence": [2.0, 1.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
            "evidence_support": ["exact_full_grid", "exact_full_grid"],
        }
    )

    assert event_group_columns(scores) == ["session", "event_index"]

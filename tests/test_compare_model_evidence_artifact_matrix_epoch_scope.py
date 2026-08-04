from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_model_evidence_artifacts import compare_artifacts  # noqa: E402


def _write_event_scores(path: Path, filename: str, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / filename, index=False)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "session": "Rat1/Open1",
            "matrix_id": "matrix-a",
            "benchmark_event_epoch": "run",
            "event_index": 0,
            "model": "momentum",
            "log_evidence": 10.0,
        },
        {
            "session": "Rat1/Open1",
            "matrix_id": "matrix-a",
            "benchmark_event_epoch": "run",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": 0.0,
        },
        {
            "session": "Rat1/Open1",
            "matrix_id": "matrix-b",
            "benchmark_event_epoch": "sleep",
            "event_index": 0,
            "model": "momentum",
            "log_evidence": 0.0,
        },
        {
            "session": "Rat1/Open1",
            "matrix_id": "matrix-b",
            "benchmark_event_epoch": "sleep",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": 10.0,
        },
    ]


def test_compare_artifacts_scopes_reused_events_by_matrix_and_epoch(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_event_scores(left, "event_model_evidence.csv", _rows())
    _write_event_scores(right, "all_sessions_event_model_evidence.csv", _rows())

    tables = compare_artifacts(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "comparison",
    )

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 2
    assert summary["right_events"] == 2
    assert summary["matched_events"] == 2
    assert summary["canonical_best_agreements"] == 2

    events = tables["best_comparison"]
    assert set(zip(events["matrix_id"], events["benchmark_event_epoch"], strict=True)) == {
        ("matrix-a", "run"),
        ("matrix-b", "sleep"),
    }
    assert set(events["left_canonical_best_model"]) == {"momentum", "diffusion"}
    assert set(events["right_canonical_best_model"]) == {"momentum", "diffusion"}

    relative = tables["relative"]
    assert len(relative) == 4
    assert set(zip(relative["matrix_id"], relative["benchmark_event_epoch"], strict=True)) == {
        ("matrix-a", "run"),
        ("matrix-b", "sleep"),
    }
    assert relative["right_minus_left_relative_log_evidence"].eq(0.0).all()

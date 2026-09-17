from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_model_evidence_runs import compare_runs  # noqa: E402


def _write_event_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "event_model_evidence.csv", index=False)


def test_compare_runs_retains_negative_infinite_model_evidence_with_finite_reference(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    rows = [
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "momentum",
            "log_evidence": -2.0,
            "status": "success",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": float("-inf"),
            "status": "success",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 1,
            "model": "diffusion",
            "log_evidence": float("-inf"),
            "status": "success",
        },
    ]
    _write_event_scores(left, rows)
    _write_event_scores(right, rows)

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 1
    assert summary["right_events"] == 1
    assert summary["matched_events"] == 1
    assert tables["event_comparison"].iloc[0]["left_canonical_best_model"] == "momentum"

    relative = tables["relative"].set_index("canonical_model")
    assert set(relative.index) == {"diffusion", "momentum"}
    assert relative.loc["diffusion", "left_relative_log_evidence"] == float("-inf")
    assert relative.loc["diffusion", "right_relative_log_evidence"] == float("-inf")
    assert relative.loc["momentum", "left_relative_log_evidence"] == 0.0
    assert relative.loc["momentum", "right_relative_log_evidence"] == 0.0

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_model_evidence_runs import compare_runs  # noqa: E402


def _write_event_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "event_model_evidence.csv", index=False)


def test_compare_runs_scopes_reused_event_indices_by_simulation_seed(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    rows = [
        {
            "session": "Rat1/Open1",
            "simulation_random_seed": 1,
            "event_index": 0,
            "model": "momentum",
            "log_evidence": 10.0,
        },
        {
            "session": "Rat1/Open1",
            "simulation_random_seed": 1,
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": 0.0,
        },
        {
            "session": "Rat1/Open1",
            "simulation_random_seed": 2,
            "event_index": 0,
            "model": "momentum",
            "log_evidence": 0.0,
        },
        {
            "session": "Rat1/Open1",
            "simulation_random_seed": 2,
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": 10.0,
        },
    ]
    _write_event_scores(left, rows)
    _write_event_scores(right, rows)

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 2
    assert summary["right_events"] == 2
    assert summary["matched_events"] == 2
    assert summary["canonical_best_agreements"] == 2
    assert set(tables["event_comparison"]["simulation_random_seed"]) == {1, 2}
    assert set(tables["event_comparison"]["left_canonical_best_model"]) == {"momentum", "diffusion"}

    counts = tables["counts"].set_index(["run_label", "canonical_model"])["events"].to_dict()
    assert counts == {
        ("left", "momentum"): 1,
        ("left", "diffusion"): 1,
        ("right", "momentum"): 1,
        ("right", "diffusion"): 1,
    }
    assert len(tables["relative"]) == 4
    assert set(tables["relative"]["simulation_random_seed"]) == {1, 2}

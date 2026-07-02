from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_model_evidence_runs.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("compare_model_evidence_runs", _SCRIPT)
assert _SPEC is not None
compare_model_evidence_runs = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_model_evidence_runs"] = compare_model_evidence_runs
assert _SPEC.loader is not None
_SPEC.loader.exec_module(compare_model_evidence_runs)


def _score_row(model: str, log_evidence: float, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "session": "Rat1/Open1",
        "event_index": 7,
        "model": model,
        "log_evidence": log_evidence,
    }
    row.update(extra)
    return row


def _write_scores(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "event_model_evidence.csv", index=False)


def test_compare_runs_matches_shared_event_keys_when_only_one_run_has_extra_metadata(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scores(
        left,
        [
            _score_row("sorted-spike-state-space-diffusion", -10.0, null_random_seed=123),
            _score_row("sorted-spike-state-space-momentum-exact-sparse", -1.0, null_random_seed=123),
        ],
    )
    _write_scores(
        right,
        [
            _score_row("sorted-spike-state-space-diffusion", -11.0),
            _score_row("sorted-spike-state-space-momentum-exact-sparse", -2.0),
        ],
    )

    tables = compare_model_evidence_runs.compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "comparison",
    )

    assert int(tables["summary"].loc[0, "matched_events"]) == 1
    assert tables["event_comparison"]["canonical_best_agree"].tolist() == [True]
    assert tables["event_comparison"]["left_canonical_best_model"].tolist() == ["momentum"]
    assert set(tables["relative"]["canonical_model"]) == {"diffusion", "momentum"}


def test_compare_runs_ignores_event_key_column_when_one_run_only_has_missing_metadata(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scores(
        left,
        [
            _score_row("sorted-spike-state-space-diffusion", -10.0, null_random_seed=123),
            _score_row("sorted-spike-state-space-momentum-exact-sparse", -1.0, null_random_seed=123),
        ],
    )
    _write_scores(
        right,
        [
            _score_row("sorted-spike-state-space-diffusion", -11.0, null_random_seed=pd.NA),
            _score_row("sorted-spike-state-space-momentum-exact-sparse", -2.0, null_random_seed=pd.NA),
        ],
    )

    tables = compare_model_evidence_runs.compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "comparison-missing-column",
    )

    assert int(tables["summary"].loc[0, "matched_events"]) == 1
    assert tables["event_comparison"]["canonical_best_agree"].tolist() == [True]
    assert set(tables["relative"]["canonical_model"]) == {"diffusion", "momentum"}

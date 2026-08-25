from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_model_evidence_runs.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("compare_model_evidence_runs_large_integer_keys", _SCRIPT)
assert _SPEC is not None
compare_model_evidence_runs = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_model_evidence_runs_large_integer_keys"] = compare_model_evidence_runs
assert _SPEC.loader is not None
_SPEC.loader.exec_module(compare_model_evidence_runs)


def _score_rows(event_index: int, null_random_seed: object, *, diffusion: float, momentum: float) -> list[dict[str, object]]:
    return [
        {
            "session": "Rat1/Open1",
            "event_index": event_index,
            "null_random_seed": null_random_seed,
            "model": "state-space-diffusion",
            "log_evidence": diffusion,
        },
        {
            "session": "Rat1/Open1",
            "event_index": event_index,
            "null_random_seed": null_random_seed,
            "model": "state-space-momentum-exact-sparse",
            "log_evidence": momentum,
        },
    ]


def _write_scores(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir()
    pd.DataFrame(rows).to_csv(root / "event_model_evidence.csv", index=False)


def test_compare_runs_preserves_nullable_large_integer_scope_keys(tmp_path: Path) -> None:
    lower_seed = 2**53
    upper_seed = lower_seed + 1
    rows = [
        *_score_rows(7, lower_seed, diffusion=5.0, momentum=1.0),
        *_score_rows(7, upper_seed, diffusion=1.0, momentum=6.0),
        *_score_rows(8, pd.NA, diffusion=4.0, momentum=2.0),
    ]
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scores(left, rows)
    _write_scores(right, rows)

    # With pandas' default CSV inference the nullable seed column becomes
    # float64, and 2**53 + 1 is rounded down to 2**53. The comparison must keep
    # these as two distinct null-control scopes.
    inferred = pd.read_csv(left / "event_model_evidence.csv")
    inferred_nonmissing = inferred["null_random_seed"].dropna().unique()
    assert len(inferred_nonmissing) == 1

    tables = compare_model_evidence_runs.compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "comparison",
    )

    summary = tables["summary"].iloc[0]
    assert int(summary["left_events"]) == 3
    assert int(summary["right_events"]) == 3
    assert int(summary["matched_events"]) == 3

    comparison = tables["event_comparison"]
    preserved = {
        int(value)
        for value in comparison.loc[
            comparison["event_index"] == 7,
            "null_random_seed",
        ]
        if not pd.isna(value)
    }
    assert preserved == {lower_seed, upper_seed}
    best_by_seed = {
        int(row.null_random_seed): row.left_canonical_best_model
        for row in comparison.loc[comparison["event_index"] == 7].itertuples()
    }
    assert best_by_seed == {
        lower_seed: "diffusion",
        upper_seed: "momentum",
    }

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from compare_model_evidence_artifacts import compare_artifacts  # noqa: E402
from compare_model_evidence_runs import compare_runs  # noqa: E402


def _write_scores(path: Path, filename: str, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / filename, index=False)


def _mixed_support_rows(*, momentum_model: str = "momentum") -> list[dict[str, object]]:
    return [
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": -2.0,
            # Deliberately stale value to ensure loaders recompute relative
            # evidence after applying the evidence-support filter.
            "relative_log_evidence": -123.0,
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": momentum_model,
            "log_evidence": 0.0,
            "relative_log_evidence": 0.0,
            "diagnostic_state_space_momentum_evidence_support": "truncated_full_grid",
        },
    ]


def test_compare_runs_defaults_to_exact_only_and_recomputes_relative_evidence(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_scores(left, "event_model_evidence.csv", _mixed_support_rows())
    _write_scores(
        right,
        "event_model_evidence.csv",
        _mixed_support_rows(momentum_model="state-space-momentum"),
    )

    tables = compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "exact",
    )

    best = tables["event_comparison"].iloc[0]
    assert best["left_canonical_best_model"] == "diffusion"
    assert best["right_canonical_best_model"] == "diffusion"
    assert best["left_best_relative_log_evidence"] == 0.0
    assert best["right_best_relative_log_evidence"] == 0.0
    assert bool(tables["summary"].iloc[0]["exact_only"])

    lower_bound_tables = compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "with-lower-bounds",
        exact_only=False,
    )
    lower_bound_best = lower_bound_tables["event_comparison"].iloc[0]
    assert lower_bound_best["left_canonical_best_model"] == "momentum"
    assert lower_bound_best["right_canonical_best_model"] == "momentum"


def test_compare_artifacts_defaults_to_exact_only_and_can_include_lower_bounds(tmp_path):
    left = tmp_path / "artifact-left"
    right = tmp_path / "artifact-right"
    _write_scores(left, "event_model_evidence.csv", _mixed_support_rows())
    _write_scores(
        right,
        "all_sessions_event_model_evidence.csv",
        _mixed_support_rows(momentum_model="sorted-spike-state-space-momentum"),
    )

    tables = compare_artifacts(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "artifact-exact",
    )

    best = tables["best_comparison"].iloc[0]
    assert best["left_canonical_best_model"] == "diffusion"
    assert best["right_canonical_best_model"] == "diffusion"
    assert best["left_best_relative_log_evidence"] == 0.0
    assert best["right_best_relative_log_evidence"] == 0.0
    assert bool(tables["summary"].iloc[0]["exact_only"])

    lower_bound_tables = compare_artifacts(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "artifact-with-lower-bounds",
        exact_only=False,
    )
    lower_bound_best = lower_bound_tables["best_comparison"].iloc[0]
    assert lower_bound_best["left_canonical_best_model"] == "momentum"
    assert lower_bound_best["right_canonical_best_model"] == "momentum"

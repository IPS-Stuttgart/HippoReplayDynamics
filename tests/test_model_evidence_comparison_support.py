from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from audit_model_evidence_support import TRUNCATED_EVIDENCE_SUPPORT  # noqa: E402
from compare_model_evidence_artifacts import compare_artifacts, load_scores  # noqa: E402
from compare_model_evidence_runs import compare_runs  # noqa: E402


def _toy_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "status": "success",
                "log_evidence": -10.0,
                # Deliberately stale/mixed-support value: loaders should recompute
                # relative evidence after exact-only filtering.
                "relative_log_evidence": -1.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "status": "success",
                "log_evidence": -9.0,
                "relative_log_evidence": 0.0,
                "diagnostic_state_space_momentum_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )


def _write_score_dir(tmp_path: Path, name: str) -> Path:
    score_dir = tmp_path / name
    score_dir.mkdir()
    _toy_scores().to_csv(score_dir / "event_model_evidence.csv", index=False)
    return score_dir


def test_artifact_comparison_defaults_to_exact_comparable_evidence(tmp_path):
    left = _write_score_dir(tmp_path, "left")
    right = _write_score_dir(tmp_path, "right")

    tables = compare_artifacts(left, right, output=tmp_path / "out")

    best = tables["best_comparison"]
    summary = tables["summary"].iloc[0]
    assert bool(summary["exact_only"])
    assert best["left_canonical_best_model"].tolist() == ["diffusion"]
    assert best["right_canonical_best_model"].tolist() == ["diffusion"]


def test_artifact_comparison_can_include_lower_bounds_for_diagnostics(tmp_path):
    left = _write_score_dir(tmp_path, "left")
    right = _write_score_dir(tmp_path, "right")

    tables = compare_artifacts(left, right, output=tmp_path / "out", exact_only=False)

    best = tables["best_comparison"]
    summary = tables["summary"].iloc[0]
    assert not bool(summary["exact_only"])
    assert best["left_canonical_best_model"].tolist() == ["momentum"]
    assert best["right_canonical_best_model"].tolist() == ["momentum"]


def test_artifact_loader_recomputes_relative_evidence_after_filtering(tmp_path):
    score_dir = _write_score_dir(tmp_path, "scores")

    scores = load_scores(score_dir, "left")

    assert scores["canonical_model"].tolist() == ["diffusion"]
    assert scores["relative_log_evidence"].tolist() == [0.0]


def test_legacy_run_comparison_defaults_to_exact_comparable_evidence(tmp_path):
    left = _write_score_dir(tmp_path, "left")
    right = _write_score_dir(tmp_path, "right")

    tables = compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "out",
    )

    event_comparison = tables["event_comparison"]
    summary = tables["summary"].iloc[0]
    assert bool(summary["exact_only"])
    assert event_comparison["left_canonical_best_model"].tolist() == ["diffusion"]
    assert event_comparison["right_canonical_best_model"].tolist() == ["diffusion"]


def test_legacy_run_comparison_can_include_lower_bounds_for_diagnostics(tmp_path):
    left = _write_score_dir(tmp_path, "left")
    right = _write_score_dir(tmp_path, "right")

    tables = compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=tmp_path / "out",
        exact_only=False,
    )

    event_comparison = tables["event_comparison"]
    summary = tables["summary"].iloc[0]
    assert not bool(summary["exact_only"])
    assert event_comparison["left_canonical_best_model"].tolist() == ["momentum"]
    assert event_comparison["right_canonical_best_model"].tolist() == ["momentum"]

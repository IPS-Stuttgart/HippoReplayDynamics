from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from compare_model_evidence_artifacts import canonical_model_name, compare_artifacts  # noqa: E402


def _write_scores(path: Path, filename: str, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / filename, index=False)


def test_canonical_model_name_maps_state_space_aliases():
    assert canonical_model_name("sorted-spike-state-space-momentum") == "momentum"
    assert canonical_model_name("clusterless-state-space-momentum") == "momentum"
    assert canonical_model_name("state-space-diffusion") == "diffusion"
    assert canonical_model_name("jump") == "fragmented"


def test_compare_artifacts_accepts_single_and_all_session_score_filenames(tmp_path):
    left = tmp_path / "old"
    right = tmp_path / "new"
    output = tmp_path / "comparison"
    _write_scores(
        left,
        "event_model_evidence.csv",
        [
            {"session": "Rat1/Open1", "event_index": 0, "model": "momentum", "log_evidence": -1.0},
            {"session": "Rat1/Open1", "event_index": 0, "model": "diffusion", "log_evidence": -3.0},
            {"session": "Rat1/Open1", "event_index": 1, "model": "momentum", "log_evidence": -4.0},
            {"session": "Rat1/Open1", "event_index": 1, "model": "diffusion", "log_evidence": -2.0},
        ],
    )
    _write_scores(
        right,
        "all_sessions_event_model_evidence.csv",
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": -2.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": -4.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -1.0,
            },
        ],
    )

    tables = compare_artifacts(left, right, left_label="old", right_label="new", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["matched_events"] == 2
    assert summary["canonical_best_agreements"] == 1
    assert summary["canonical_best_agreement_fraction"] == 0.5
    assert str(summary["right_score_file"]).endswith("all_sessions_event_model_evidence.csv")

    assert (output / "event_best_model_comparison.csv").is_file()
    assert (output / "canonical_best_model_crosstab.csv").is_file()
    assert (output / "evidence_support_counts.csv").is_file()
    assert (output / "shared_relative_evidence_summary.csv").is_file()
    assert (output / "session_story_shift_summary.csv").is_file()
    assert (output / "model_evidence_artifact_comparison_summary.csv").is_file()


def test_compare_artifacts_exact_only_filters_truncated_rows(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    _write_scores(
        left,
        "event_model_evidence.csv",
        [
            {"session": "Rat1/Open1", "event_index": 0, "model": "diffusion", "log_evidence": -3.0},
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "momentum",
                "log_evidence": -1.0,
                "diagnostic_state_space_momentum_evidence_support": "truncated_full_grid",
            },
        ],
    )
    _write_scores(
        right,
        "all_sessions_event_model_evidence.csv",
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "log_evidence": -2.0,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "log_evidence": -0.5,
                "diagnostic_state_space_momentum_evidence_support": "truncated_full_grid",
            },
        ],
    )

    tables = compare_artifacts(left, right, left_label="left", right_label="right", output=output, exact_only=True)

    assert tables["summary"].iloc[0]["matched_events"] == 1
    assert set(tables["best_comparison"]["left_canonical_best_model"]) == {"diffusion"}
    assert set(tables["best_comparison"]["right_canonical_best_model"]) == {"diffusion"}
    assert set(tables["support_counts"]["evidence_support"]) == {"exact_full_grid"}

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_model_evidence_runs import canonical_model_name, compare_runs  # noqa: E402


def _write_event_scores(path: Path, rows: list[dict[str, object]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "event_model_evidence.csv", index=False)


def test_compare_runs_handles_empty_successful_score_tables(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    failed_row = {
        "session": "Rat1/Open1",
        "event_index": 0,
        "model": "momentum",
        "log_evidence": 0.0,
        "status": "error",
    }
    _write_event_scores(left, [failed_row])
    _write_event_scores(right, [failed_row])

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 0
    assert summary["right_events"] == 0
    assert summary["matched_events"] == 0
    assert summary["canonical_best_agreements"] == 0
    assert pd.isna(summary["canonical_best_agreement_fraction"])
    assert tables["event_comparison"].empty
    assert tables["relative"].empty
    assert (output / "model_evidence_run_comparison_summary.csv").exists()


def test_compare_runs_ignores_nonfinite_successful_log_evidence(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    left_rows = [
        {"session": "Rat1/Open1", "event_index": 0, "model": "momentum", "log_evidence": float("nan"), "status": "success"},
        {"session": "Rat1/Open1", "event_index": 0, "model": "diffusion", "log_evidence": float("-inf"), "status": "success"},
        {"session": "Rat1/Open1", "event_index": 1, "model": "momentum", "log_evidence": 2.0, "status": "success"},
        {"session": "Rat1/Open1", "event_index": 1, "model": "diffusion", "log_evidence": 1.0, "status": "success"},
    ]
    right_rows = [
        {"session": "Rat1/Open1", "event_index": 0, "model": "momentum", "log_evidence": float("nan"), "status": "success"},
        {"session": "Rat1/Open1", "event_index": 0, "model": "diffusion", "log_evidence": float("inf"), "status": "success"},
        {"session": "Rat1/Open1", "event_index": 1, "model": "momentum", "log_evidence": 3.0, "status": "success"},
        {"session": "Rat1/Open1", "event_index": 1, "model": "diffusion", "log_evidence": 0.0, "status": "success"},
    ]
    _write_event_scores(left, left_rows)
    _write_event_scores(right, right_rows)

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 1
    assert summary["right_events"] == 1
    assert summary["matched_events"] == 1
    assert tables["event_comparison"]["event_index"].tolist() == [1]
    assert set(tables["relative"]["event_index"]) == {1}
    assert not tables["relative"]["left_relative_log_evidence"].isna().any()
    assert not tables["relative"]["right_relative_log_evidence"].isna().any()


def test_compare_runs_treats_blank_status_as_legacy_success(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    rows = [
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "momentum",
            "log_evidence": 2.0,
            "status": "",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": 1.0,
            "status": None,
        },
    ]
    _write_event_scores(left, rows)
    _write_event_scores(right, rows)

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output, exact_only=True)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 1
    assert summary["right_events"] == 1
    assert summary["matched_events"] == 1
    assert set(tables["event_comparison"]["left_canonical_best_model"]) == {"momentum"}
    assert set(tables["support_counts"]["evidence_comparable"]) == {True}


def test_compare_runs_treats_na_like_and_case_status_as_legacy_success(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    rows = [
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "momentum",
            "log_evidence": 2.0,
            "status": "<na>",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": 1.0,
            "status": "Success",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "stationary",
            "log_evidence": 99.0,
            "status": "failed",
        },
    ]
    _write_event_scores(left, rows)
    _write_event_scores(right, rows)

    tables = compare_runs(left, right, left_label="left", right_label="right", output=output, exact_only=True)

    summary = tables["summary"].iloc[0]
    assert summary["left_events"] == 1
    assert summary["right_events"] == 1
    assert summary["matched_events"] == 1
    assert set(tables["event_comparison"]["left_canonical_best_model"]) == {"momentum"}
    assert "stationary" not in set(tables["event_comparison"]["left_canonical_best_model"])
    assert set(tables["support_counts"]["evidence_comparable"]) == {True}


def test_canonical_model_name_maps_state_space_aliases():
    assert canonical_model_name("sorted-spike-state-space-momentum") == "momentum"
    assert canonical_model_name("sorted-spike-state-space-momentum-exact-sparse") == "momentum"
    assert canonical_model_name("sorted-spike-state-space-displacement-momentum") == "momentum"
    assert canonical_model_name("state-space-velocity-momentum") == "momentum"
    assert canonical_model_name("clusterless-state-space-momentum") == "momentum"
    assert canonical_model_name("state-space-first-order-imm") == "imm"
    assert canonical_model_name("sorted-spike-state-space-trajectory-imm-exact-sparse") == "imm"
    assert canonical_model_name("clusterless-state-space-displacement-imm") == "imm"
    assert canonical_model_name("state-space-diffusion") == "diffusion"
    assert canonical_model_name("jump") == "fragmented"
    assert canonical_model_name("stationary-gaussian") == "stationary-gaussian"


def test_compare_runs_matches_exact_imm_variants_by_canonical_label(tmp_path):
    left = tmp_path / "legacy"
    right = tmp_path / "exact"
    output = tmp_path / "comparison"
    _write_event_scores(
        left,
        [
            {"session": "Rat1/Open1", "event_index": 0, "model": "state-space-imm", "log_evidence": 5.0},
            {"session": "Rat1/Open1", "event_index": 0, "model": "state-space-diffusion", "log_evidence": 0.0},
        ],
    )
    _write_event_scores(
        right,
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "state-space-trajectory-imm-exact-sparse",
                "log_evidence": 6.0,
            },
            {"session": "Rat1/Open1", "event_index": 0, "model": "state-space-diffusion", "log_evidence": 0.0},
        ],
    )

    tables = compare_runs(left, right, left_label="legacy", right_label="exact", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["matched_events"] == 1
    assert summary["canonical_best_agreements"] == 1
    assert tables["event_comparison"].iloc[0]["legacy_canonical_best_model"] == "imm"
    assert tables["event_comparison"].iloc[0]["exact_canonical_best_model"] == "imm"
    assert tables["relative"].loc[tables["relative"]["canonical_model"] == "imm"].shape[0] == 1


def test_compare_runs_writes_best_model_and_relative_evidence_tables(tmp_path):
    left = tmp_path / "kd"
    right = tmp_path / "state"
    output = tmp_path / "comparison"
    _write_event_scores(
        left,
        [
            {"session": "Rat1/Open1", "event_index": 0, "model": "momentum", "log_evidence": -1.0},
            {"session": "Rat1/Open1", "event_index": 0, "model": "diffusion", "log_evidence": -3.0},
            {"session": "Rat1/Open1", "event_index": 1, "model": "momentum", "log_evidence": -4.0},
            {"session": "Rat1/Open1", "event_index": 1, "model": "diffusion", "log_evidence": -2.0},
        ],
    )
    _write_event_scores(
        right,
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

    tables = compare_runs(left, right, left_label="kd", right_label="state", output=output)

    summary = tables["summary"].iloc[0]
    assert summary["matched_events"] == 2
    assert summary["left_events"] == 2
    assert summary["right_events"] == 2
    assert summary["canonical_best_agreements"] == 1
    assert summary["canonical_best_agreement_fraction"] == 0.5

    counts = pd.read_csv(output / "best_model_counts_comparison.csv")
    assert counts.set_index(["run_label", "canonical_model"])["events"].to_dict() == {
        ("kd", "momentum"): 1,
        ("kd", "diffusion"): 1,
        ("state", "diffusion"): 2,
    }

    relative = pd.read_csv(output / "shared_model_relative_evidence_summary.csv")
    assert set(relative["canonical_model"]) == {"diffusion", "momentum"}
    assert set(relative["matched_events"]) == {2}

    session_comparison = pd.read_csv(output / "session_model_evidence_comparison.csv")
    assert session_comparison.to_dict(orient="records") == [
        {
            "session": "Rat1/Open1",
            "kd_diffusion_wins": 1,
            "state_diffusion_wins": 2,
            "kd_momentum_wins": 1,
            "state_momentum_wins": 0,
            "momentum_win_delta": -1,
            "canonical_best_agreement_fraction": 0.5,
            "mean_momentum_relative_evidence_delta": -1.0,
        }
    ]

    assert (output / "event_best_model_comparison.csv").exists()
    assert (output / "best_model_canonical_crosstab.csv").exists()
    assert (output / "shared_model_relative_evidence_comparison.csv").exists()
    assert (output / "session_model_evidence_comparison.csv").exists()
    assert (output / "model_evidence_run_comparison_summary.csv").exists()


def test_compare_runs_exact_only_filters_string_false_comparable_rows(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    output = tmp_path / "comparison"
    rows = [
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "diffusion",
            "log_evidence": -3.0,
            "status": "success",
            "evidence_support": "exact_full_grid",
            "evidence_comparable": "True",
        },
        {
            "session": "Rat1/Open1",
            "event_index": 0,
            "model": "momentum",
            "log_evidence": -1.0,
            "status": "success",
            "evidence_support": "truncated_full_grid",
            "evidence_comparable": "False",
        },
    ]
    _write_event_scores(left, rows)
    _write_event_scores(right, rows)

    tables = compare_runs(
        left,
        right,
        left_label="left",
        right_label="right",
        output=output,
        exact_only=True,
    )

    assert set(tables["event_comparison"]["left_canonical_best_model"]) == {"diffusion"}
    assert set(tables["support_counts"]["evidence_comparable"]) == {True}

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path("scripts").resolve()))
from audit_model_evidence_support import (  # noqa: E402
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    assert_no_mixed_support,
    ensure_evidence_support_columns,
    evidence_support_summary,
    mixed_support_violations,
    paired_delta_summary,
    pooled_paired_delta_summary,
    write_audit,
)
import model_evidence_support_audit as support_audit_tables  # noqa: E402


def _toy_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -10.0,
                "is_best_model": False,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -9.0,
                "is_best_model": False,
                "diagnostic_state_space_momentum_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-imm",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -8.5,
                "is_best_model": False,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -7.0,
                "is_best_model": True,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-momentum",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -8.0,
                "is_best_model": False,
                "diagnostic_state_space_momentum_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-imm",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -7.5,
                "is_best_model": False,
                "diagnostic_state_space_imm_evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
            },
        ]
    )


def test_ensure_evidence_support_columns_infers_exact_and_truncated():
    audited = ensure_evidence_support_columns(_toy_scores())
    by_model = audited.groupby("model")["evidence_support"].first().to_dict()

    assert by_model["sorted-spike-state-space-diffusion"] == EXACT_EVIDENCE_SUPPORT
    assert by_model["sorted-spike-state-space-momentum"] == TRUNCATED_EVIDENCE_SUPPORT
    assert by_model["sorted-spike-state-space-imm"] == TRUNCATED_EVIDENCE_SUPPORT
    assert audited.loc[audited["model"].str.endswith("diffusion"), "evidence_comparable"].all()
    assert not audited.loc[audited["model"].str.endswith("momentum"), "evidence_comparable"].any()


def test_support_summary_keeps_support_classes_separate():
    summary = evidence_support_summary(_toy_scores()).set_index("model")

    assert summary.loc["sorted-spike-state-space-diffusion", "evidence_support"] == EXACT_EVIDENCE_SUPPORT
    assert summary.loc["sorted-spike-state-space-momentum", "evidence_support"] == TRUNCATED_EVIDENCE_SUPPORT
    assert summary.loc["sorted-spike-state-space-diffusion", "events"] == 2
    assert summary.loc["sorted-spike-state-space-diffusion", "wins"] == 1


def test_support_summary_parses_string_bool_win_flags():
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -10.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": "True",
                "is_best_model": "False",
                "is_best_truncated_lower_bound": "False",
            },
            {
                "session": "Rat1/Open1",
                "event_index": 1,
                "model": "sorted-spike-state-space-diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -8.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": "True",
                "is_best_model": "True",
                "is_best_truncated_lower_bound": "False",
            },
        ]
    )

    summary = evidence_support_summary(scores).set_index("model")

    assert summary.loc["sorted-spike-state-space-diffusion", "wins"] == 1
    assert summary.loc[
        "sorted-spike-state-space-diffusion",
        "truncated_lower_bound_wins",
    ] == 0


def test_paired_delta_summary_labels_lower_bound_vs_exact():
    deltas = paired_delta_summary(_toy_scores())
    momentum_diffusion = deltas[
        deltas["comparison"].eq(
            "sorted-spike-state-space-momentum_minus_sorted-spike-state-space-diffusion"
        )
    ].iloc[0]

    assert momentum_diffusion["comparison_support"] == "truncated_lower_bound_vs_exact"
    assert momentum_diffusion["events"] == 2
    assert momentum_diffusion["positive_events"] == 1
    assert np.isclose(momentum_diffusion["mean_delta"], 0.0)


def test_paired_delta_summary_parses_string_false_comparable_flags():
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -10.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": "True",
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "sorted-spike-state-space-momentum",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": -9.0,
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": "False",
            },
        ]
    )

    deltas = paired_delta_summary(scores)
    momentum_diffusion = deltas[
        deltas["comparison"].eq(
            "sorted-spike-state-space-momentum_minus_sorted-spike-state-space-diffusion"
        )
    ].iloc[0]

    assert momentum_diffusion["comparison_support"] == "truncated_lower_bound_vs_exact"


def test_pooled_paired_delta_summary_aggregates_events():
    pooled = pooled_paired_delta_summary(paired_delta_summary(_toy_scores()))
    row = pooled[
        pooled["comparison"].eq(
            "sorted-spike-state-space-momentum_minus_sorted-spike-state-space-diffusion"
        )
    ].iloc[0]

    assert row["events"] == 2
    assert row["positive_events"] == 1
    assert np.isclose(row["positive_fraction"], 0.5)


def test_mixed_support_violations_are_detected():
    violations = mixed_support_violations(paired_delta_summary(_toy_scores()))

    assert not violations.empty
    assert set(violations["comparison_support"]) == {"truncated_lower_bound_vs_exact"}


def test_assert_no_mixed_support_accepts_like_with_like():
    like_with_like = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "comparison": ["a_minus_b", "c_minus_d"],
            "comparison_support": [EXACT_EVIDENCE_SUPPORT, TRUNCATED_EVIDENCE_SUPPORT],
            "left_model": ["a", "c"],
            "right_model": ["b", "d"],
            "events": [2, 2],
            "positive_events": [1, 1],
            "mean_delta": [0.0, 0.0],
        }
    )

    assert_no_mixed_support(like_with_like)


def test_write_audit_writes_expected_csvs(tmp_path):
    input_csv = tmp_path / "all_sessions_event_model_evidence.csv"
    output_dir = tmp_path / "audit"
    _toy_scores().to_csv(input_csv, index=False)

    write_audit(input_csv, output_dir)

    assert (output_dir / "evidence_support_summary.csv").is_file()
    assert (output_dir / "session_paired_delta_summary.csv").is_file()
    assert (output_dir / "pooled_paired_delta_summary.csv").is_file()
    assert (output_dir / "mixed_support_violations.csv").is_file()


def test_write_audit_can_fail_on_mixed_support(tmp_path):
    input_csv = tmp_path / "all_sessions_event_model_evidence.csv"
    output_dir = tmp_path / "audit"
    _toy_scores().to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="mixed evidence support"):
        write_audit(input_csv, output_dir, fail_on_mixed_support=True)

    assert (output_dir / "mixed_support_violations.csv").is_file()


def test_support_audit_tables_parse_string_false_comparable_flags():
    scores = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "diffusion",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": 1.0,
                "evidence_support": EXACT_EVIDENCE_SUPPORT,
                "evidence_comparable": "True",
            },
            {
                "session": "Rat1/Open1",
                "event_index": 0,
                "model": "momentum",
                "model_family": "trajectory",
                "status": "success",
                "log_evidence": 2.0,
                "evidence_support": TRUNCATED_EVIDENCE_SUPPORT,
                "evidence_comparable": "False",
            },
        ]
    )

    event_audit = support_audit_tables.event_support_audit(scores)
    pairwise = support_audit_tables.pairwise_support_audit(scores)

    assert event_audit.loc[0, "comparable_rows"] == 1
    assert bool(event_audit.loc[0, "has_uncomparable_rows"]) is True
    assert bool(pairwise.loc[0, "model_b_comparable"]) is False
    assert bool(pairwise.loc[0, "both_exact_comparable"]) is False

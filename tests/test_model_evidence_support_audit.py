from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("scripts").resolve()))
from audit_model_evidence_support import (  # noqa: E402
    EXACT_EVIDENCE_SUPPORT,
    TRUNCATED_EVIDENCE_SUPPORT,
    ensure_evidence_support_columns,
    evidence_support_summary,
    paired_delta_summary,
    pooled_paired_delta_summary,
    write_audit,
)


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


def test_write_audit_writes_expected_csvs(tmp_path):
    input_csv = tmp_path / "all_sessions_event_model_evidence.csv"
    output_dir = tmp_path / "audit"
    _toy_scores().to_csv(input_csv, index=False)

    write_audit(input_csv, output_dir)

    assert (output_dir / "evidence_support_summary.csv").is_file()
    assert (output_dir / "session_paired_delta_summary.csv").is_file()
    assert (output_dir / "pooled_paired_delta_summary.csv").is_file()

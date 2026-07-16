from __future__ import annotations

import pandas as pd

import hipporeplayimm.advanced_result_diagnostics as diagnostics


def test_empty_threshold_sequence_preserves_threshold_sweep_schema():
    scores = pd.DataFrame(
        columns=[
            "session",
            "event_index",
            "model",
            "log_evidence",
            "true_model",
        ]
    )

    sweep = diagnostics.paired_model_margin_threshold_sweep(
        scores,
        positive_model="momentum",
        reference_model="diffusion",
        thresholds=(),
        group_cols=("session", "event_index"),
        true_model_col="true_model",
    )

    expected_columns = {
        "events",
        "positive_model",
        "reference_model",
        "margin_threshold",
        "positive_model_claims",
        "reference_model_claims",
        "ambiguous_events",
        "positive_claim_fraction",
        "mean_positive_minus_reference_log_evidence",
        "median_positive_minus_reference_log_evidence",
        "thresholded_binary_accuracy",
        "positive_true_events",
        "reference_true_events",
        "positive_true_claimed_events",
        "reference_true_rejected_events",
        "positive_claim_recall",
        "reference_specificity",
        "false_positive_claims",
        "false_negative_claims",
        "group_cols",
    }
    assert sweep.empty
    assert expected_columns.issubset(sweep.columns)

    selected = diagnostics.select_paired_model_margin_threshold(sweep)
    assert selected.loc[0, "selection_status"] == "empty_threshold_sweep"

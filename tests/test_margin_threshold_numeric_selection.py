import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    select_paired_model_margin_threshold,
)


def test_threshold_selector_orders_csv_threshold_strings_numerically():
    sweep = pd.DataFrame(
        {
            "margin_threshold": ["10", "2"],
            "false_positive_claims": ["0", "0"],
            "positive_claim_recall": ["1.0", "1.0"],
        }
    )

    selected = select_paired_model_margin_threshold(
        sweep,
        max_false_positive_claims=0,
        min_positive_claim_recall=0.5,
    )

    assert selected.loc[0, "selection_status"] == "passed_specificity_gate"
    assert selected.loc[0, "selected_margin_threshold"] == 2.0


def test_threshold_selector_orders_csv_false_positive_counts_numerically_in_fallback():
    sweep = pd.DataFrame(
        {
            "margin_threshold": ["1", "2"],
            "false_positive_claims": ["10", "2"],
            "positive_claim_recall": ["0.9", "0.8"],
        }
    )

    selected = select_paired_model_margin_threshold(
        sweep,
        max_false_positive_claims=0,
        min_positive_claim_recall=0.0,
    )

    assert selected.loc[0, "selection_status"] == "fallback_no_gate_pass"
    assert selected.loc[0, "false_positive_claims"] == 2
    assert selected.loc[0, "selected_margin_threshold"] == 2.0

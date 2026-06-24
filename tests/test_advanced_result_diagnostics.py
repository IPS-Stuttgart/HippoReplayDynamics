from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    adversarial_synthetic_case_specs,
    classify_evidence_margin,
    common_support_from_emissions,
    evidence_margin_table,
    hierarchical_summary,
    mark_drift_diagnostics,
    model_disagreement_events,
    paired_model_margin_decisions,
    paired_model_margin_summary,
    paired_model_margin_threshold_sweep,
    place_field_quality_from_arrays,
    select_paired_model_margin_threshold,
    posterior_predictive_count_checks,
    rat_bootstrap_wrong_map_absolute_evidence_summary,
    stable_cell_ids,
    wrong_map_absolute_evidence_deltas,
    wrong_map_absolute_evidence_summary,
    wrong_map_delta_summary,
    wrong_map_family_margin_difference_in_differences,
    wrong_map_family_margin_difference_in_differences_summary,
)


def test_evidence_margin_table_classifies_decisive_and_ties():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 1, 1],
            "model": ["a", "b", "a", "b"],
            "log_evidence": [10.0, 0.0, 2.0, 1.5],
            "status": ["success"] * 4,
            "evidence_comparable": [True] * 4,
        }
    )
    margins = evidence_margin_table(scores)
    assert margins.loc[margins["event_index"] == 0, "evidence_margin_category"].iloc[0] == "strong"
    assert margins.loc[margins["event_index"] == 1, "evidence_margin_category"].iloc[0] == "tie"
    merged = add_evidence_margin_columns(scores)
    assert "evidence_margin_to_second_best" in merged


def test_evidence_margin_table_excludes_string_false_comparable_rows():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0, 0],
            "model": ["stationary", "diffusion", "lower-bound-audit"],
            "log_evidence": [0.0, 2.0, 100.0],
            "status": ["success"] * 3,
            "evidence_comparable": ["True", "True", "False"],
        }
    )

    margins = evidence_margin_table(scores)

    assert margins.iloc[0]["best_model_by_evidence"] == "diffusion"
    assert margins.iloc[0]["second_best_model_by_evidence"] == "stationary"
    assert margins.iloc[0]["models_compared"] == 2


def test_wrong_map_delta_summary():
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "event_index": [0],
            "model": ["m"],
            "log_evidence": [5.0],
            "status": ["success"],
        }
    )
    wrong = current.copy()
    wrong["log_evidence"] = 2.0
    out = wrong_map_delta_summary(current, wrong)
    assert float(out["delta_vs_wrong_environment_map"].iloc[0]) == 3.0


def test_wrong_map_absolute_evidence_is_primary_map_control():
    current = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0, 0, 1, 1],
            "model": [
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
            ],
            "log_evidence": [0.0, 10.0, 0.0, 8.0],
            "status": ["success"] * 4,
        }
    )
    wrong = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 4,
            "event_index": [0, 0, 1, 1],
            "map_session": ["Rat1/Open2"] * 4,
            "model": [
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
                "sorted-spike-state-space-stationary",
                "sorted-spike-state-space-first-order-imm",
            ],
            # The wrong map has lower absolute evidence, but a larger
            # trajectory-minus-stationary margin.
            "log_evidence": [-100.0, -50.0, -80.0, -40.0],
            "status": ["success"] * 4,
        }
    )

    deltas = wrong_map_absolute_evidence_deltas(current, wrong)
    summary = wrong_map_absolute_evidence_summary(deltas).set_index("statistic")
    bootstrap = rat_bootstrap_wrong_map_absolute_evidence_summary(
        deltas,
        n_bootstrap=20,
        random_seed=7,
    ).set_index("statistic")
    did = wrong_map_family_margin_difference_in_differences(current, wrong)
    did_summary = wrong_map_family_margin_difference_in_differences_summary(did).iloc[0]

    best = summary.loc["best_exact_trajectory_model_real_map"]
    assert int(best["events"]) == 2
    assert int(best["positive_delta_events"]) == 2
    assert float(best["mean_delta_map_log_evidence"]) == 54.0
    assert bootstrap.loc["best_exact_trajectory_model_real_map", "mean_delta_ci95_low"] > 0

    assert did["margin_difference_in_differences"].tolist() == [-40.0, -32.0]
    assert float(did_summary["mean_difference_in_differences"]) == -36.0


def test_paired_model_margin_decisions_reject_weak_momentum_claims():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1"] * 8,
            "event_index": [0, 0, 1, 1, 2, 2, 3, 3],
            "true_model": ["diffusion", "diffusion", "diffusion", "diffusion", "momentum", "momentum", "momentum", "momentum"],
            "model": [
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
            ],
            "log_evidence": [0.0, 4.0, 0.0, -6.0, 0.0, 8.0, 0.0, 2.0],
            "status": ["success"] * 8,
            "evidence_comparable": [True] * 8,
        }
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="sorted-spike-state-space-momentum-exact-sparse",
        reference_model="sorted-spike-state-space-diffusion",
        margin_threshold=5.0,
        true_model_col="true_model",
        positive_true_label="momentum",
    )
    summary = paired_model_margin_summary(decisions, true_model_col="true_model")

    assert decisions["margin_decision"].tolist() == [
        "ambiguous",
        "sorted-spike-state-space-diffusion",
        "sorted-spike-state-space-momentum-exact-sparse",
        "ambiguous",
    ]
    assert decisions["positive_model_claimed"].tolist() == [False, False, True, False]
    assert summary.loc[0, "false_positive_claims"] == 0
    assert summary.loc[0, "reference_specificity"] == 1.0
    assert summary.loc[0, "positive_claim_recall"] == 0.5


def test_paired_model_margin_decisions_reject_invalid_scalar_thresholds():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["momentum", "diffusion"],
            "log_evidence": [1.0, 0.0],
            "status": ["success", "success"],
            "evidence_comparable": [True, True],
        }
    )

    for threshold in (True, np.bool_(False), np.nan, np.inf, -1.0):
        with pytest.raises(ValueError, match="finite nonnegative"):
            paired_model_margin_decisions(
                scores,
                positive_model="momentum",
                reference_model="diffusion",
                margin_threshold=threshold,
            )


def test_paired_model_margin_summary_parses_string_bool_decisions():
    decisions = pd.DataFrame(
        {
            "positive_model": ["momentum", "momentum"],
            "reference_model": ["diffusion", "diffusion"],
            "margin_threshold": [5.0, 5.0],
            "positive_model_claimed": ["False", "True"],
            "margin_decision": ["ambiguous", "momentum"],
            "positive_minus_reference_log_evidence": [1.0, 6.0],
            "margin_binary_correct": ["True", "False"],
            "true_is_positive": ["False", "False"],
            "true_model": ["diffusion", "diffusion"],
        }
    )

    summary = paired_model_margin_summary(decisions, true_model_col="true_model")

    assert summary.loc[0, "positive_model_claims"] == 1
    assert summary.loc[0, "positive_claim_fraction"] == 0.5
    assert summary.loc[0, "false_positive_claims"] == 1
    assert summary.loc[0, "reference_specificity"] == 0.5


def test_model_disagreement_events_parses_string_false_best_flags():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open1"],
            "event_index": [0, 0],
            "model": ["lower-bound", "exact"],
            "log_evidence": [100.0, 1.0],
            "status": ["success", "success"],
            "is_best_model": ["False", "True"],
        }
    )

    out = model_disagreement_events(scores)

    assert out.loc[0, "best_model"] == "exact"


def test_paired_model_margin_threshold_selection_uses_synthetic_specificity():
    scores = pd.DataFrame(
        {
            "matrix_id": ["cell-a"] * 8,
            "random_seed": [11] * 8,
            "session": ["Rat1/Open1"] * 8,
            "simulation_event_index": [0, 0, 1, 1, 2, 2, 3, 3],
            "true_model": [
                "diffusion",
                "diffusion",
                "diffusion",
                "diffusion",
                "momentum",
                "momentum",
                "momentum",
                "momentum",
            ],
            "model": [
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
                "sorted-spike-state-space-diffusion",
                "sorted-spike-state-space-momentum-exact-sparse",
            ],
            "log_evidence": [0.0, 4.0, 0.0, -6.0, 0.0, 8.0, 0.0, 6.0],
            "status": ["success"] * 8,
            "evidence_comparable": [True] * 8,
        }
    )

    sweep = paired_model_margin_threshold_sweep(
        scores,
        positive_model="sorted-spike-state-space-momentum-exact-sparse",
        reference_model="sorted-spike-state-space-diffusion",
        thresholds=(0.0, 3.0, 5.0),
        true_model_col="true_model",
        positive_true_label="momentum",
    )
    selected = select_paired_model_margin_threshold(sweep, max_false_positive_claims=0)

    assert sweep["group_cols"].unique().tolist() == [
        "matrix_id,random_seed,session,simulation_event_index"
    ]
    assert sweep["false_positive_claims"].tolist() == [1, 1, 0]
    assert sweep["positive_claim_recall"].tolist() == [1.0, 1.0, 1.0]
    assert selected.loc[0, "selected_margin_threshold"] == 5.0
    assert selected.loc[0, "selection_status"] == "passed_specificity_gate"


def test_place_field_quality_and_stable_cells():
    rates = np.array([[1.0, 10.0, 1.0], [0.5, 0.5, 0.5]])
    occupancy = np.array([1.0, 1.0, 1.0])
    q = place_field_quality_from_arrays(rates, occupancy, cell_ids=[11, 12])
    assert set(q["cell_id"]) == {11, 12}
    stable = stable_cell_ids(q, min_spatial_information_bits=0.1, min_peak_rate_hz=1.0)
    assert 11 in set(stable)


def test_common_support_from_emissions_includes_extras():
    ll = np.array([[0.0, 1.0, 2.0], [3.0, 1.0, 0.0]])
    support = common_support_from_emissions(ll, top_k=1, extra_candidate_sets=[[0], [2]])
    assert set(support[0]) == {0, 2}
    assert set(support[1]) == {0, 2}


def test_mark_drift_diagnostics_returns_blocks():
    times = np.arange(8, dtype=float)
    marks = np.column_stack([times, times + 1.0])
    out = mark_drift_diagnostics(times, marks, n_blocks=4)
    assert len(out) == 4
    assert "mark_mean_distance_from_first_block" in out


def test_posterior_predictive_count_checks():
    obs = np.array([[0, 1], [2, 0]])
    exp = np.array([[0.2, 0.8], [1.5, 0.1]])
    out = posterior_predictive_count_checks(obs, exp)
    assert set(out["predictive_check"]).issuperset({"total_spike_count", "silent_bin_fraction"})


def test_hierarchical_summary_and_synthetic_specs():
    scores = pd.DataFrame(
        {
            "session": ["Rat1/Open1", "Rat1/Open2"],
            "model": ["m", "m"],
            "relative_log_evidence": [1.0, 3.0],
            "status": ["success", "success"],
        }
    )
    out = hierarchical_summary(scores)
    assert float(out.loc[out["model"] == "m", "event_mean"].iloc[0]) == 2.0
    assert "reverse_replay" in set(adversarial_synthetic_case_specs()["synthetic_case"])


def test_classify_evidence_margin_boundaries():
    assert classify_evidence_margin(0.5) == "tie"
    assert classify_evidence_margin(2.0) == "weak"
    assert classify_evidence_margin(5.0) == "strong"
    assert classify_evidence_margin(11.0) == "decisive"

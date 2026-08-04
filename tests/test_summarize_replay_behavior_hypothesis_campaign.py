from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.summarize_replay_behavior_hypothesis_campaign import (
    benjamini_hochberg,
    build_campaign_results,
    build_gate_summary,
    map_off_swr_route_context,
)


def test_bh_uses_complete_ten_hypothesis_family() -> None:
    values = pd.Series([0.01, 0.02, 0.5, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    adjusted = benjamini_hochberg(values)
    assert adjusted.iloc[0] == 0.1
    assert adjusted.iloc[1] == 0.1


def _context_rows() -> pd.DataFrame:
    rows = []
    for hypothesis, estimate, p_value in (
        ("H1", 0.2, 0.02),
        ("H2", -0.2, 0.03),
        ("H3", -0.1, 0.04),
        ("H4", 0.1, 0.5),
        ("H10", -0.1, 0.04),
    ):
        rows.append(
            {
                "hypothesis": hypothesis,
                "test": f"{hypothesis}_primary",
                "role": "primary",
                "outcome": "outcome",
                "events": 160,
                "rats": 4,
                "sessions": 8,
                "estimate": estimate,
                "rat_bootstrap_ci_low": estimate - 0.05,
                "rat_bootstrap_ci_high": estimate + 0.05,
                "permutation_p_value": p_value,
                "status_before_campaign_fdr": "directional_pass_unadjusted",
                "null_control": "shuffle",
            }
        )
    return pd.DataFrame(rows)


def test_campaign_keeps_insufficient_tests_in_fdr_as_one() -> None:
    h5 = pd.DataFrame(
        [
            {
                "hypothesis": "H5",
                "role": "primary",
                "test": "branching",
                "events": 160,
                "rats": 4,
                "sessions": 8,
                "estimate": 0.03,
                "rat_bootstrap_ci_low": 0.01,
                "rat_bootstrap_ci_high": 0.05,
                "permutation_p_value": 0.03,
                "positive_rats": 3,
                "leave_one_rat_out_positive": True,
                "null_control": "circular",
            }
        ]
    )
    transitions = pd.DataFrame(
        {"stationary_continuous_switch_probability_given_nonfragmented": [0.1, 0.2]}
    )
    h6 = pd.DataFrame(
        {
            "session": ["Rat1/Open1"],
            "rat": ["Rat1"],
            "event_index": [1],
            "assembly_boundary_candidate_count": [0],
        }
    )
    h7 = pd.DataFrame(
        [
            {
                "selection_rule": "strongest_exact_margin",
                "source_event_groups": 7,
                "candidate_rats": 3,
                "candidate_sessions": 5,
                "trajectory_confident_candidates": 7,
            }
        ]
    )
    h8 = pd.DataFrame(
        [
            {
                "events": 20,
                "rats": 4,
                "ordered_trajectory_fraction_excess": 0.2,
                "empirical_p_value_one_sided": 1 / 21,
                "original_ordered_trajectory_fraction": 0.5,
                "median_shuffle_ordered_trajectory_fraction": 0.3,
            }
        ]
    )
    h8_gates = pd.DataFrame(
        [{"gate": "ordered_grammar_exceeds_whole_bin_shuffle", "passed": True}]
    )
    h9 = pd.DataFrame(
        [
            {
                "population_contrast": "all",
                "metric": "post_minus_pre_time_order_advantage_imm_minus_fragmented",
                "animals": 4,
                "equal_animal_mean": 0.1,
                "rat_bootstrap_ci_low": 0.01,
                "rat_bootstrap_ci_high": 0.2,
                "one_sided_sign_test_p": 0.0625,
                "positive_robust": True,
            }
        ]
    )
    results = build_campaign_results(
        context_tests=_context_rows(),
        h5_tests=h5,
        h6_transitions=transitions,
        h6_splits=h6,
        h7_summary=h7,
        h7_context=pd.DataFrame(
            {"route_timing_relation": ["during_segmented_movement"] * 7}
        ),
        h8_summary=h8,
        h8_gates=h8_gates,
        h9_inference=h9,
    )
    assert results["hypothesis"].tolist() == [f"H{i}" for i in range(1, 11)]
    assert results.set_index("hypothesis").loc["H6", "fdr_input_p_value"] == 1.0
    assert results.set_index("hypothesis").loc["H7", "fdr_input_p_value"] == 1.0
    assert not results["campaign_significant"].any()
    gates = build_gate_summary(results)
    assert bool(gates.set_index("gate").loc["overall", "passed"])
    assert np.isfinite(results["bh_q_value_10_hypotheses"]).all()


def test_off_swr_route_context_distinguishes_pause_and_movement() -> None:
    decisions = pd.DataFrame(
        [
            {
                "selection_rule": "strongest_exact_margin",
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 1,
                "null_index": 0,
                "window_start_s": 12.0,
                "window_end_s": 12.2,
            },
            {
                "selection_rule": "strongest_exact_margin",
                "session": "Rat1/Open1",
                "rat": "Rat1",
                "event_index": 2,
                "null_index": 1,
                "window_start_s": 16.0,
                "window_end_s": 16.2,
            },
        ]
    )
    routes = pd.DataFrame(
        [
            {
                "session": "Rat1/Open1",
                "route_id": "route_1",
                "interval_start_time_s": 10.0,
                "interval_end_time_s": 20.0,
                "movement_start_time_s": 15.0,
                "movement_end_time_s": 19.0,
                "duration_s": 10.0,
            }
        ]
    )
    context = map_off_swr_route_context(decisions, routes)
    assert context["route_timing_relation"].tolist() == [
        "pre_departure_pause",
        "during_segmented_movement",
    ]

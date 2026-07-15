from __future__ import annotations

import pandas as pd

from scripts.report_virtual_movement_existence import build_evidence_table


def test_build_evidence_table_separates_pf_positive_from_tanni_stopgate() -> None:
    time_order = pd.DataFrame(
        {
            "event_group": ["clean_imm"],
            "events": [20],
            "original_above_shuffle_p95_count": [19],
            "median_time_order_advantage": [45.0],
        }
    )
    content = pd.DataFrame(
        {
            "group": ["overall"],
            "events": [108],
            "moderate_content_pass_count": [97],
            "moderate_content_pass_fraction": [97 / 108],
            "median_posterior_net_displacement_cm": [66.8],
        }
    )
    map_specificity = pd.DataFrame(
        {
            "analysis_scope": ["clean_imm_fixed_subset"],
            "metric": ["mean_nonstationary_mode_probability"],
            "events": [108],
            "empirical_p_le_0p05_count": [79],
            "median_real_minus_null_median": [0.20],
        }
    )
    heldout = pd.DataFrame(
        {
            "scope": ["all_events"],
            "events": [160],
            "event_heldout_delta_positive_count": [145],
            "event_heldout_delta_positive_fraction": [145 / 160],
            "median_event_heldout_delta": [7.73],
        }
    )
    decoder = pd.DataFrame({"median_posterior_mean_error_cm": [12.0, 14.0]})
    tanni_events = pd.DataFrame(
        {
            "ordered_model_confident": [True, True, False],
            "original_ordered_margin": [7.0, 6.0, -1.0],
            "p95_time_shuffle_margin": [5.0, 5.0, 2.0],
            "p95_map_shuffle_margin": [4.0, 7.0, 2.0],
            "displacing": [True, True, False],
        }
    )
    tanni_summary = pd.DataFrame(
        {
            "scope": ["all_events", "one_per_source_group"],
            "events": [3, 2],
            "ordered_model_confident": [2, 1],
            "strict_virtual_movement": [0, 0],
            "median_original_ordered_margin": [1.0, 1.0],
        }
    )

    evidence = build_evidence_table(
        time_order,
        content,
        map_specificity,
        heldout,
        decoder,
        tanni_events,
        tanni_summary,
    )

    indexed = evidence.set_index(["dataset", "test"])
    assert indexed.loc[("Pfeiffer/Foster", "posterior trajectory content"), "passed_count"] == 97
    assert indexed.loc[("Pfeiffer/Foster", "held-out-cell prediction"), "passed_count"] == 145
    assert indexed.loc[("Tanni large arenas", "event-level two-null movement candidates"), "passed_count"] == 1
    assert indexed.loc[("Tanni large arenas", "familywise strict virtual movement"), "passed_count"] == 0

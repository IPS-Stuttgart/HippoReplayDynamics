from __future__ import annotations

import numpy as np
import pandas as pd

from hipporeplayimm.advanced_result_diagnostics import (
    add_evidence_margin_columns,
    adversarial_synthetic_case_specs,
    classify_evidence_margin,
    common_support_from_emissions,
    evidence_margin_table,
    hierarchical_summary,
    mark_drift_diagnostics,
    place_field_quality_from_arrays,
    posterior_predictive_count_checks,
    stable_cell_ids,
    wrong_map_delta_summary,
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

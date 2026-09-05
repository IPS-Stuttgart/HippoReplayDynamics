import numpy as np
import pandas as pd
import pytest

from scripts.report_replay_coverage_counterfactual import METRICS, endpoint_summary, gradient_availability, spike_budget_summary


def common():
    return {"dataset": "d", "rate_scale": 3., "truth_kind": "continuous", "gradient": 0.,
            "regime": "native", "cell_fraction": 1., "likelihood": "poisson", "estimator": "map", "bin_filter": "unfiltered"}


def test_replicate_session_animal_weighting_and_missing_metrics():
    rows = [{**common(), "animal": animal, "session": session, "replicate": replicate,
             **{metric: value if metric != "median_speed_cm_s" else np.nan for metric in METRICS}}
            for animal, session, value in [("a", "a1", 0), ("a", "a2", 0), ("b", "b1", 1)]
            for replicate in range(3)]
    table = pd.DataFrame(rows)
    animals, summary = endpoint_summary(table, 3., 1, 5, 100)
    measured = summary[summary.metric.eq("geometric_pass_fraction")]
    assert measured.estimate.eq(.5).all()  # Not the pooled-session mean of 1/3.
    missing = summary[summary.metric.eq("median_speed_cm_s")]
    assert missing.estimate.isna().all() and missing.finite_animals.eq(0).all()
    assert animals.animal.nunique() == 2
    with pytest.raises(ValueError):
        endpoint_summary(pd.concat([table, table]), 3., 1, 5, 100)


def test_gradient_missingness_stays_in_denominator():
    rows = [{**common(), "animal": animal, "session": animal, "replicate": 0,
             "selection": "selected", "coordinate": "true_coordinate", "readout": "decoded",
             "gradient": gradient, "normalized_slope": gradient if animal == "a" or gradient < 0 else np.nan}
            for animal in ["a", "b"] for gradient in [-.5, .5]]
    table = pd.DataFrame(rows)
    pairs, summary = gradient_availability(table, 3., 1)
    assert len(pairs) == 4  # Primary and dose are explicitly different analysis sets.
    assert summary.session_replicate_pairs.eq(2).all()
    assert summary.available_session_replicate_pairs.eq(1).all()
    assert summary.animals.eq(2).all() and summary.animals_with_any_available_replicate.eq(1).all()
    with pytest.raises(ValueError):
        gradient_availability(table[table.gradient.lt(0)], 3., 1)


def test_source_budget_uses_duration_normalization_and_keeps_zero_unknown():
    identity = {"dataset": "d", "animal": "a", "session": "s", "source_event_index": 1,
                "replicate": 0, "truth_kind": "continuous", "gradient": 0.}
    trials = pd.DataFrame([{**identity, "source_spikes": 10, "source_duration_s": .1, "simulated_duration_s": .05}])
    obs = pd.DataFrame([{**identity, "rate_scale": 3., "regime": "native", "cell_fraction": 1., "spikes": 10}])
    summary = spike_budget_summary(trials, obs, 3., 1)
    assert summary.simulated_to_source_rate_ratio.eq(2.).all()
    zero = spike_budget_summary(trials.assign(source_spikes=0), obs, 3., 1)
    assert zero.simulated_to_source_rate_ratio.isna().all()
    with pytest.raises(ValueError):
        spike_budget_summary(trials, obs.assign(source_event_index=2), 3., 1)

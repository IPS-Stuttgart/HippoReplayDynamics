import numpy as np
import pandas as pd
import pytest

from scripts.report_tirole_first_pair_content import intervals, paired_arrays, validate_source_sets


def rows():
    return pd.DataFrame(
        [
            {"event_id": eid, "split": s, "repeat": r, "fraction": f, "sequence_accepted": f == 1, "evaluation_track2_probability": eid / 10}
            for eid in [1, 2]
            for s in range(5)
            for r in range(5)
            for f in [1.0, 0.5]
        ]
    )


def test_frozen_pairing_does_not_count_splits_as_events():
    full, half = paired_arrays(rows(), [2, 1])
    assert full.shape == half.shape == (2, 25)
    assert full.all() and not half.any()
    a, b = paired_arrays(rows(), [2, 1], "evaluation_track2_probability")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[:, 0], [0.2, 0.1])


@pytest.mark.parametrize("change", ["one_row", "whole_split", "wrong_events", "duplicate"])
def test_incomplete_pairs_fail(change):
    d = rows()
    ids = [1, 2]
    if change == "one_row":
        d = d.iloc[1:]
    elif change == "whole_split":
        d = d[d.split != 4]
    elif change == "wrong_events":
        ids = [1, 3]
    else:
        d = pd.concat([d, d.iloc[:1]])
    with pytest.raises(ValueError):
        paired_arrays(d, ids)


def test_reports_cannot_mix_score_campaigns():
    source = [{"manifest": "a", "sha256": "123"}]
    r, i, f = {"input_score_manifests": source}, {"scorer_manifests": source}, {"source_score_manifests": source}
    validate_source_sets(source, r, i, f)
    f["source_score_manifests"] = source + source
    with pytest.raises(ValueError, match="provenance"):
        validate_source_sets(source, r, i, f)
    f["source_score_manifests"] = [{"manifest": "a", "sha256": "changed"}]
    with pytest.raises(ValueError, match="provenance"):
        validate_source_sets(source, r, i, f)


def test_intervals_use_unique_events_and_keep_empty_retained_sets_missing():
    d = rows().assign(arm="real", epoch="POST", candidate_stratum="ripple", ripple_peak_z=6)
    d["conditional_evaluation_track2_probability"] = d.evaluation_track2_probability
    events = pd.DataFrame({"event_id": [1, 2], "start_s": [0, 300]}).set_index("event_id")
    context = pd.DataFrame(
        {"event_id": [1, 2], "epoch": "POST", "fraction": 0.5, "group": "lost", "signed_evaluation_z": [1.0, 3.0], "signed_conditional_evaluation_z": [0.0, 2.0]}
    )
    forecast = pd.DataFrame(
        {"event_id": [1, 2], "epoch": "POST", "fraction": 0.5, "group": "lost", "horizon_ms": 40, "contrast": "dynamic_minus_matched_own", "delta_per_spike": [0.1, 0.3]}
    )
    result = intervals(d, events, context, context, forecast)
    focal = result[(result.candidate_tier == "ripple_z3") & (result.block_s == 60)].set_index("metric")
    assert focal.loc["half_minus_full_acceptance", "point"] == -1
    assert focal.loc["rate_matched_identity_lost_context_z", "point"] == 2
    assert focal.loc["rate_matched_identity_lost_context_z", "n_informative_events"] == 2
    assert focal.loc["lost_future_matched_advantage", "point"] == pytest.approx(0.2)
    assert np.isnan(focal.loc["poisson_selective_loss_shift", "point"])
    assert focal.loc["poisson_selective_loss_shift", "baseline_estimable_realizations"] == 0
    assert focal.loc["poisson_selective_loss_shift", "interval_status"] == "insufficient_blocks_or_nonempty_draws"


def test_invariant_heldout_probabilities_are_enforced():
    d = rows().assign(arm="real", epoch="POST", candidate_stratum="ripple", ripple_peak_z=6)
    d["conditional_evaluation_track2_probability"] = d.evaluation_track2_probability
    d.loc[d.fraction == 0.5, "evaluation_track2_probability"] = 0.9
    events = pd.DataFrame({"event_id": [1, 2], "start_s": [0, 300]}).set_index("event_id")
    context = pd.DataFrame(columns=["event_id", "epoch", "fraction", "group", "signed_evaluation_z", "signed_conditional_evaluation_z"])
    forecast = pd.DataFrame(columns=["event_id", "epoch", "fraction", "group", "horizon_ms", "contrast", "delta_per_spike"])
    with pytest.raises(ValueError, match="readout changed"):
        intervals(d, events, context, context, forecast)

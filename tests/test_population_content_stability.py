import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts.measure_population_content_stability import (
    decode,
    endpoint_counts,
    local_run_errors,
    partition,
    simulate_counts,
    tile_ids,
)
from scripts.validate_population_content_stability import (
    aggregate_sessions,
    features,
    fit,
    predict,
    rank_selection,
    session_results,
    weights,
)


def fixture():
    rows = []
    for animal in range(4):
        for session in range(2):
            for event in range(20):
                spikes = event+1
                rows.append(dict(dataset="pfeiffer_foster", animal=f"R{animal}", session=f"R{animal}/S{session}",
                    event_index=event, source="real", split=0, draw=-1, a_spikes=spikes, a_active=min(spikes, 10),
                    a_entropy=1/(1+spikes/10), a_width_cm=50/(1+spikes/10), a_peak=spikes/30,
                    n_cells=20, a_global_run_error_cm=12, a_local_run_error_cm=8+event%3, a_coverage=1,
                    grid_diagonal_cm=250, endpoint_separation_cm=70/(1+spikes/10), regional_tv=.4,
                    b_entropy=.5, b_width_cm=20, a_truth_error_cm=15, b_truth_error_cm=20,
                    map_region_agreement=.5))
    return pd.DataFrame(rows)


def test_partition_is_equal_disjoint_deterministic():
    a, b = partition(31, 17)
    assert len(a) == len(b) == 15
    assert not set(a) & set(b)
    assert np.array_equal(a, partition(31, 17)[0])
    with pytest.raises(ValueError):
        partition(19, 17)


def test_endpoints_exclude_partial_not_full_last_bin():
    data = dict(candidate_event_indices=np.array([11, 12]), candidate_offsets=np.array([0, 6, 9]),
                candidate_base_durations_s=np.array([.005]*5+[.002]+[.005]*3),
                candidate_base_starts_s=np.arange(9)*.005,
                candidate_base_counts=np.arange(18).reshape(9, 2))
    anchor, counts, excluded = endpoint_counts(data, np.array([True, False]))
    assert anchor.event_index.tolist() == [11]
    assert counts.tolist() == [[2+4+6+8]]
    assert anchor.start_s.iloc[0] == .005
    assert anchor.end_s.iloc[0] == .025
    assert excluded[0]["event_index"] == 12


def test_decode_is_flat_prior_independent_and_normalized():
    grid = np.array([[0., 0], [8., 0], [16., 8]])
    rates = np.array([[8., 1, 2], [1, 8, 2]])
    counts = np.array([[4, 0], [0, 4], [0, 0]])
    result = decode(counts, rates, grid)
    assert np.allclose(result["p"].sum(axis=1), 1)
    assert np.allclose(result["regional"].sum(axis=1), 1)
    assert np.allclose(decode(counts[::-1], rates, grid)["p"], result["p"][::-1])
    assert result["mean"][0, 0] < result["mean"][1, 0]
    with pytest.raises(ValueError):
        decode(np.array([[.5, 1]]), rates, grid)


def test_tiles_stable_at_edges():
    grid = np.array([[0., 0], [30, 30], [15, 15]])
    assert tile_ids(grid).tolist() == [0, 8, 4]


def test_local_feature_missing_outside_coverage():
    truth = np.column_stack([np.arange(10), np.zeros(10)])
    result = local_run_errors(np.array([[4., 0], [200., 200]]), truth, np.arange(10))
    assert result[0] == 4.5
    assert np.isnan(result[1])


def test_conditional_simulation_preserves_zero_and_nonzero_totals():
    totals = np.array([0, 4, 10, 2])
    counts = simulate_counts(totals, np.array([[1., 9], [9., 1]]), np.array([0, 0, 1, 1]), np.random.default_rng(1))
    assert np.array_equal(counts.sum(axis=1), totals)
    assert not counts[0].any()


def test_training_isolation_and_roundtrip():
    data = fixture()
    state = fit(data, "full")
    restored = json.loads(json.dumps(state))
    assert np.allclose(predict(data, state), predict(data, restored))
    for column, value in [("dataset", "tanni2022"), ("source", "run_test"), ("split", 1)]:
        bad = data.copy()
        bad[column] = value
        with pytest.raises(ValueError):
            fit(bad, "full")


def test_b_and_truth_cannot_change_features_or_predictions():
    data = fixture()
    state = fit(data, "full")
    altered = data.copy()
    for column in ("b_entropy", "b_width_cm", "endpoint_separation_cm", "regional_tv", "a_truth_error_cm"):
        altered[column] = 10000
    assert features(data, "full").equals(features(altered, "full"))
    assert np.array_equal(predict(data, state), predict(altered, state))


def test_imputation_uses_frozen_training_values():
    data = fixture()
    state = fit(data, "full")
    before = copy.deepcopy(state)
    data["a_local_run_error_cm"] = np.nan
    assert np.isfinite(predict(data, state)).all()
    assert state == before


def test_weights_equalize_rat_and_session():
    data = fixture()
    data = data.loc[~((data.animal == "R0") & (data.event_index > 1))].copy()
    data["weight"] = weights(data)
    assert np.allclose(data.groupby("animal").weight.sum(), .25)
    assert np.allclose(data.groupby(["animal", "session"]).weight.sum(), .125)


def test_fixed_coverage_does_not_use_outcomes():
    frame = fixture().iloc[:11].copy()
    selected = rank_selection(frame, np.zeros(11), .5)
    assert selected.sum() == 6
    assert frame.loc[selected, "event_index"].tolist() == list(range(6))
    frame["endpoint_separation_cm"] = 9999
    assert np.array_equal(selected, rank_selection(frame, np.zeros(11), .5))
    with pytest.raises(ValueError):
        rank_selection(frame, np.full(11, np.nan), .5)


def test_draws_not_independent_animals():
    rows = pd.DataFrame(dict(source=["s"]*4, animal=["A", "A", "A", "B"],
                             session=["x", "x", "y", "z"], value=[2., 2., 6., 10.]))
    animal, total = aggregate_sessions(rows, ["source"], ["value"])
    assert animal.value.tolist() == [4., 10.]
    assert total.value.iloc[0] == 7


def test_session_summary_reports_matched_retention():
    data = fixture()
    states = {m: fit(data, m) for m in ("mean", "spikes", "entropy", "spikes_entropy", "full")}
    summary, forecasts, selections = session_results(data, states)
    primary = summary.loc[summary.retention.eq(.5)]
    assert primary.n_selected.eq(10).all()
    assert selections.groupby(["animal", "session"]).selected.sum().eq(10).all()
    assert forecasts.log_mse.ge(0).all()

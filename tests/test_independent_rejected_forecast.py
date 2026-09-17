"""Leakage, detection, causal forecasting and aggregation regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.independent_rejected_forecast import (
    BASELINES,
    aggregate,
    classify,
    detect_candidates,
    detection_grid,
    event_counts,
    forecast_distributions,
    full_counts,
    partitions,
    score_predictions,
)
from hipporeplayimm.lagged_neural_prediction import NeuralOperator
from hipporeplayimm.occupancy_matched_forecast import OccupancyMatchedNull
from scripts.audit_independent_rejected_forecasts import decisions, validate


def model(iid=False):
    transition = np.full((3, 3), 1 / 3) if iid else np.array([[0.1, 0.85, 0.05], [0.05, 0.1, 0.85], [0.85, 0.05, 0.1]])
    op = NeuralOperator(np.full(3, 1 / 3), transition, np.full(3, 1 / 3))
    return op, OccupancyMatchedNull.from_operator(op)


def test_cell_partitions_are_reproducible_and_disjoint():
    a = partitions(48, ("PF", "Rat1", "Open1"))
    b = partitions(48, ("PF", "Rat1", "Open1"))
    np.testing.assert_array_equal(a[0], b[0])
    det, pop, splits = a
    assert not set(det) & set(pop)
    assert sorted([*det, *pop]) == list(range(48))
    for (train, half, held), other in zip(splits, b[2], strict=True):
        assert not set(train) & set(held)
        assert set(half) < set(train)
        assert sorted([*train, *held]) == list(range(len(pop)))
        for x, y in zip((train, half, held), other, strict=True):
            np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        partitions(5, ("a",))


def detector_fixture():
    clock = (np.arange(10000) + 0.5) * 0.001
    times = np.linspace(4.92, 5.08, 65)
    spikes = np.column_stack((np.r_[times, times + 0.0001], np.repeat([1, 2], len(times))))
    return spikes, clock, np.zeros(len(clock))


def test_detection_ignores_evaluation_spikes_and_preserves_boundaries():
    spikes, clock, speed = detector_fixture()
    a, da = detect_candidates(spikes, [1, 2], clock, speed)
    extra = np.column_stack((np.linspace(0, 10, 5000), np.full(5000, 99)))
    b, db = detect_candidates(np.vstack((spikes, extra)), [1, 2], clock, speed)
    assert len(a) == 1 and a == b and da == db
    assert a[0]["detector_active_cells"] == 2


def test_movement_and_position_gaps_not_detected():
    spikes, clock, speed = detector_fixture()
    speed[(clock >= 4.7) & (clock <= 5.3)] = np.nan
    assert detect_candidates(spikes, [1, 2], clock, speed)[0] == []
    speed[:] = 20
    assert detect_candidates(spikes, [1, 2], clock, speed)[0] == []
    t = np.r_[np.arange(0, 2, 0.02), np.arange(3, 5, 0.02)]
    pos = np.column_stack((t, np.zeros(len(t)), np.zeros(len(t))))
    c, s = detection_grid(pos, [[0, 4.98]])
    assert np.isnan(s[(c > 2) & (c < 3)]).all()


def test_encoding_motion_is_excluded_even_after_smoothing():
    t = np.arange(0, 2, 0.01)
    x = np.zeros(len(t))
    x[t >= 1] = 2
    c, s = detection_grid(np.column_stack((t, x, np.zeros(len(t)))), [[0, 1.99]])
    raw = abs(np.gradient(x, t))
    dangerous = np.interp(c, t, raw) >= 10
    assert not np.any((s < 5) & dangerous)


def test_count_bins_half_open_partial_recorded():
    spikes = {1: np.array([0, 0.005, 0.02, 0.05]), 2: np.array([0.019, 0.047])}
    base, dt = event_counts(spikes, [1, 2], 0, 0.05)
    counts, discarded = full_counts(base, dt)
    assert counts.tolist() == [[2, 1], [1, 0]]
    assert discarded == 1
    base, dt = event_counts(spikes, [1, 2], 0, 0.053)
    counts, discarded = full_counts(base, dt)
    assert discarded == 2


def test_classification_never_reads_evaluation_columns():
    rng = np.random.default_rng(1)
    base = rng.poisson(1, (40, 6))
    rates = np.maximum(rng.random((6, 20)), 1e-4)
    centers = np.column_stack((np.arange(20) * 8, np.zeros(20)))
    a = classify(base, np.full(40, 0.005), rates, centers, np.array([0, 1, 2, 3]), np.array([0, 2]))
    base[:, 4:] = 1000
    rates[4:] *= 1e6
    b = classify(base, np.full(40, 0.005), rates, centers, np.array([0, 1, 2, 3]), np.array([0, 2]))
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


def test_no_future_or_heldout_spikes_enter_forecast():
    op, null = model()
    emissions = np.full((3, 3), 0.01) + 0.97 * np.eye(3)
    train = np.eye(3, dtype=int)[np.arange(12) % 3] * 3
    a = forecast_distributions(train, emissions, op, null)
    changed = train.copy()
    changed[1:] = changed[1:, ::-1] + 1
    b = forecast_distributions(changed, emissions, op, null)
    for h in a:
        for name in a[h]:
            np.testing.assert_array_equal(a[h][name][0], b[h][name][0])
    held = train.copy()
    saved = {h: {k: q.copy() for k, q in values.items()} for h, values in a.items()}
    score_predictions(a, held, emissions, np.ones(3))
    score_predictions(a, held[:, ::-1], emissions, np.ones(3))
    for h in a:
        for name in a[h]:
            np.testing.assert_array_equal(a[h][name], saved[h][name])


def test_synthetic_order_generalizes_beyond_matched_null():
    op, null = model()
    emissions = 0.01 + 0.97 * np.eye(3)
    train = np.eye(3, dtype=int)[np.arange(30) % 3] * 5
    held = np.eye(3, dtype=int)[np.arange(30) % 3] * 4
    predictions = forecast_distributions(train, emissions, op, null)
    scores = {r["horizon"]: r for r in score_predictions(predictions, held, emissions, np.ones(3))}
    assert scores[2]["score_dynamic"] > scores[2]["score_matched_own"] + 10
    assert scores[2]["score_dynamic"] > scores[2]["score_global"]
    assert null.equilibrium_error < 1e-10


def test_iid_has_no_spurious_temporal_advantage():
    op, null = model(iid=True)
    emissions = 0.01 + 0.97 * np.eye(3)
    rng = np.random.default_rng(99)
    train = rng.poisson(1, (30, 3))
    held = rng.poisson(1, (30, 3))
    scores = score_predictions(forecast_distributions(train, emissions, op, null), held, emissions, np.ones(3))
    for row in scores:
        assert row["score_dynamic"] == pytest.approx(row["score_matched_own"], abs=1e-10)
        assert row["score_dynamic"] == pytest.approx(row["score_no_history"], abs=1e-10)


def score_rows():
    rows = []
    for event, values in [(0, [1.0, 3.0, 100.0]), (1, [9.0] * 5)]:
        for split, delta in enumerate(values):
            r = dict(
                dataset="pfeiffer_foster",
                animal="Rat1",
                session="Open1",
                event_id=event,
                split=split,
                level="full",
                model="learned_hmm",
                horizon=2,
                n_target_bins=2,
                n_heldout_target_spikes=1,
                status="scored",
                rejected_with_opportunity=True,
                geometric_pass=False,
                lost_with_thinning=False,
                score_dynamic=-200.0,
            )
            r.update({"score_" + b: -200.0 - delta for b in BASELINES})
            rows.append(r)
    return pd.DataFrame(rows)


def test_event_medians_precede_session_averages():
    _, events, _, _, summary, _ = aggregate(score_rows())
    chosen = summary[(summary.group == "rejected_with_opportunity") & (summary.contrast == "dynamic_minus_matched_own") & (summary.metric == "delta_per_spike")]
    assert chosen.iloc[0]["mean"] == 6
    assert len(events[events.group == "rejected_with_opportunity"]) == 10


def test_missing_empty_and_zero_targets_not_vacuous_passes():
    with pytest.raises(ValueError):
        aggregate(pd.DataFrame())
    r = score_rows()
    with pytest.raises(ValueError):
        aggregate(pd.concat([r, r.iloc[:1]]))
    r["n_heldout_target_spikes"] = 0
    out = aggregate(r)
    assert out[1].delta_per_spike.isna().all()
    if not out[4].empty:
        result = decisions(out[4], [])
        assert not result.primary_and_adequacy_supported.any()
    with pytest.raises(ValueError):
        validate(score_rows(), [])

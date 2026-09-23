import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("sparse_calibration", Path(__file__).parents[1] / "scripts/calibrate_dandi000978_sparse_content.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def synthetic():
    rows, counts = [], []
    for epoch in range(3):
        for route in range(4):
            for rep in range(3):
                for window in range(2):
                    rows.append({"trial_id": epoch * 100 + route * 10 + rep, "epoch": epoch, "route": route})
                    v = np.ones(6, dtype=int)
                    v[route] = 30 + window
                    counts.append(v)
    return pd.DataFrame(rows), np.asarray(counts)


@pytest.fixture(autouse=True)
def small_parameters(monkeypatch):
    monkeypatch.setitem(m.PARAMETERS, "repeats", 2)
    monkeypatch.setitem(m.PARAMETERS, "null_draws", 19)


def test_thin_exact_preserves_support_counts_and_seed():
    c = np.array([0, 3, 10, 4])
    a = m.thin_exact(c, 8, np.random.default_rng(6))
    b = m.thin_exact(c, 8, np.random.default_rng(6))
    assert np.array_equal(a, b)
    assert a.sum() == 8 and (a <= c).all() and a[0] == 0
    assert m.thin_exact(c, 18, np.random.default_rng(6)) is None
    assert not m.thin_exact(c, 0, np.random.default_rng(6)).any()


@pytest.mark.parametrize("counts,target", [([-1, 2], 1), ([0.5, 1], 1), ([1, 2], -1), ([1, 2], 0.5), ([np.nan], 0)])
def test_invalid_counts_rejected(counts, target):
    with pytest.raises(ValueError):
        m.thin_exact(counts, target, np.random.default_rng(0))


def test_holdout_does_not_use_test_spikes_for_fit_or_inclusion():
    bins, counts = synthetic()
    rates, use, train, test = m.fit_holdout(bins, counts, 2)
    changed = counts.copy()
    changed[bins.epoch == 2] *= 100
    after, after_use, _, _ = m.fit_holdout(bins, changed, 2)
    assert np.array_equal(rates, after)
    assert np.array_equal(use, after_use)
    assert not set(train) & set(test)


def test_missing_test_route_fails():
    bins, counts = synthetic()
    mask = ~((bins.epoch == 2) & (bins.route == 3))
    with pytest.raises(ValueError, match="Missing test route"):
        m.fit_holdout(bins[mask], counts[mask], 2)


def test_anchor_matching_has_no_clipping_or_zero_exclusion():
    bins, counts = synthetic()
    test = bins.epoch == 2
    frame, native, sparse = m.sample_anchors(bins[test].reset_index(drop=True), counts[test], [0, 4, 1000], ("test",))
    assert len(frame) == 3 * 2 * 4 * 2
    assert not frame[frame.target_count == 1000].matched.any()
    assert frame[frame.target_count == 0].matched.all()
    assert (sparse[frame.matched].sum(axis=1) == frame.loc[frame.matched, "target_count"]).all()
    assert (sparse <= native).all()
    repeat = m.sample_anchors(bins[test].reset_index(drop=True), counts[test], [0, 4, 1000], ("test",))
    pd.testing.assert_frame_equal(frame, repeat[0])
    assert np.array_equal(sparse, repeat[2])


def test_null_keeps_repeated_trial_labels_together():
    ids = np.array([4, 1, 4, 3, 1, 2])
    routes = np.array([3, 0, 3, 2, 0, 1])
    unique, shuffled, inv = m.trial_null_labels(ids, routes, ("null",))
    assert len(unique) == 4
    assert (shuffled[:, inv[0]] == shuffled[:, inv[2]]).all()
    assert (np.sort(shuffled, axis=1) == np.arange(4)).all()


def test_uniform_and_zero_posteriors_get_fractional_credit():
    score, p = m.composition_scores(np.array([[0, 0], [3, 7]]), np.ones((4, 2)))
    assert np.allclose(m.tie_credit(p), 0.25)
    assert np.allclose(score, 0)


def test_positive_synthetic_readout_and_null():
    bins, counts = synthetic()
    rates, use, _, _ = m.fit_holdout(bins, counts, 2)
    test = bins.epoch == 2
    frame, native, sparse = m.sample_anchors(bins[test].reset_index(drop=True), counts[test][:, use], [5, 10], ("positive",))
    rows, nulls, audit = m.evaluate_anchors(frame, native, sparse, rates, ("positive",))
    assert len(rows) == 4 and len(nulls) == 4 * 19
    assert all(r["accuracy"] > 0.75 for r in rows)
    assert all(r["centered_score"] > r["null_score_p95"] for r in rows)
    assert audit["null_labels"].shape[0] == 19


def test_summary_missing_fold_fails_nonvacuously(tmp_path):
    bins, counts = synthetic()
    rates, use, _, _ = m.fit_holdout(bins, counts, 2)
    test = bins.epoch == 2
    frame, native, sparse = m.sample_anchors(bins[test].reset_index(drop=True), counts[test][:, use], [5], ("summary",))
    rows, nulls, _ = m.evaluate_anchors(frame, native, sparse, rates, ("summary",))
    base = {"animal": "JS14", "region": "PFC", "heldout_epoch": 2, "file": "f"}
    folds = pd.DataFrame([{**base, **r} for r in rows])
    ns = pd.DataFrame([{**base, **r} for r in nulls])
    assert not m.summarize(folds, ns, [("f", 2, "PFC"), ("missing", 2, "PFC")], tmp_path)
    gates = pd.read_csv(tmp_path / "gate_summary.csv").set_index("gate").passed
    assert not gates["all_expected_folds_complete"]
    assert not gates["sleep_content_validated"]
    assert not gates["paper_ready"]

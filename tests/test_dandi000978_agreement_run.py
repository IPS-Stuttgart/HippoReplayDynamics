"""Metadata, fold-isolation and numerical controls for the provisional RUN pilot."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("agreement", Path(__file__).parents[1] / "scripts/dandi000978_agreement_run.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def unit_table():
    return pd.DataFrame({"unit_id": np.arange(10), "nwb_row_region": ["CA1"] * 5 + ["PFC"] * 5,
                         "alternative_tetrode_id_region_not_author_confirmed": ["CA1"] * 3 + ["PFC", "unresolved"] + ["PFC"] * 3 + ["CA1", "unresolved"]})


def test_agreement_excludes_both_disputed_and_unresolved():
    groups, members = m.agreement_groups(unit_table(), np.arange(10)[::-1])
    assert set(members.unit_id.iloc[groups["A"]]) == {0, 1, 2}
    assert set(members.unit_id.iloc[groups["B"]]) == {5, 6, 7}
    assert not members.anatomy_confirmed.any()


def test_duplicate_ids_fail_closed():
    u = unit_table()
    u.loc[1, "unit_id"] = 0
    with pytest.raises(ValueError):
        m.agreement_groups(u, np.arange(10))


def test_too_few_units_fail_closed():
    u = unit_table()
    u.loc[0, "nwb_row_region"] = "unknown"
    with pytest.raises(ValueError):
        m.agreement_groups(u, np.arange(10))


@pytest.mark.parametrize("scheme", ["blocked_trial", "leave_one_epoch_out"])
def test_every_trial_tested_once_no_overlap(scheme):
    epoch = np.repeat(np.arange(4), 21)
    folds = m.make_folds(epoch, scheme)
    assert sorted(np.concatenate([t for _, _, t, _ in folds])) == list(range(len(epoch)))
    for _, tr, te, purge in folds:
        assert not set(tr) & set(te)
        assert not set(tr) & set(purge)
        if scheme == "leave_one_epoch_out":
            assert not set(epoch[tr]) & set(epoch[te])
        else:
            for i in te:
                for j in (i-1, i+1):
                    if 0 <= j < len(epoch) and epoch[i] == epoch[j]:
                        assert j not in tr


def test_boundary_spike_goes_to_one_trial_only():
    arrays = {"unit_ids": np.array([0]), "spike_ends": np.array([4]), "spikes": np.array([0., 1., 2., 3.])}
    trials = pd.DataFrame({"start_time": [0., 1.], "stop_time": [1., 3.]})
    np.testing.assert_array_equal(m.trial_counts(arrays, trials), [[1], [2]])


def fixture_data():
    rng = np.random.default_rng(15)
    y = np.tile([0, 1, 2], 40)
    duration = rng.uniform(1, 2, len(y))
    rates = np.array([[14, 2, 1, 4], [1, 14, 2, 4], [2, 1, 14, 4]])
    x = rng.poisson(rates[y] * duration[:, None])
    epoch = np.repeat(np.arange(6), 20)
    return x, duration, y, epoch


@pytest.mark.parametrize("model", m.MODELS)
def test_test_labels_and_other_test_values_cannot_change_prediction(model):
    x, d, y, _ = fixture_data()
    tr, te = np.arange(80), np.arange(80, 120)
    p = m.fit_predict(x, d, y, tr, te[:1], 3, model)
    yy, xx, dd = y.copy(), x.copy(), d.copy()
    yy[te] = (yy[te] + 1) % 3
    xx[te[1:]] += 10000
    dd[te[1:]] *= 100
    pp = m.fit_predict(xx, dd, yy, tr, te[:1], 3, model)
    np.testing.assert_allclose(p, pp)


def test_composition_cannot_learn_a_pure_total_count_difference():
    y = np.repeat([0, 1], 20)
    x = np.r_[np.full((20, 3), 10), np.full((20, 3), 100)]
    p = m.fit_predict(x, np.ones(40), y, np.arange(40), np.arange(40), 2, "conditional_identity")
    np.testing.assert_allclose(p, 0.5)


def test_silent_population_is_uniform():
    x, d, y, _ = fixture_data()
    x[80:] = 0
    p = m.fit_predict(x, d, y, np.arange(80), np.arange(80, 120), 3, "conditional_identity")
    np.testing.assert_allclose(p, 1/3)


@pytest.mark.parametrize("kind", m.NULLS)
def test_nulls_keep_per_epoch_class_counts_and_are_seeded(kind):
    _, _, y, ep = fixture_data()
    yp = m.null_labels(y, ep, np.random.default_rng(4), kind)
    np.testing.assert_array_equal(yp, m.null_labels(y, ep, np.random.default_rng(4), kind))
    for e in np.unique(ep):
        np.testing.assert_array_equal(np.sort(yp[ep == e]), np.sort(y[ep == e]))
    assert not np.array_equal(y, yp)


def test_synthetic_route_identity_is_recovered_out_of_epoch():
    x, d, y, ep = fixture_data()
    p = m.cross_predict(x, d, y, 3, m.make_folds(ep, "leave_one_epoch_out"), "conditional_identity")
    assert m.metrics(y, p)["balanced_accuracy"] > 0.85
    assert m.metrics(y, p)["macro_log_loss_nats"] < np.log(3)


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        m.fit_predict(np.array([[-1], [2.]]), np.ones(2), np.array([0, 1]), np.array([0]), np.array([1]), 2, "conditional_identity")


def test_unsupported_route_still_has_finite_predictions():
    x, d, y, _ = fixture_data()
    for model in m.MODELS:
        p = m.fit_predict(x, d, y, np.flatnonzero(y != 2), np.flatnonzero(y == 2), 3, model)
        assert np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(axis=1), 1)


def test_complete_synthetic_output_contract(tmp_path):
    x, d, y, ep = fixture_data()
    trials = pd.DataFrame({"id": np.arange(len(y)), "route": np.array(["1->2", "2->1", "1->3"])[y],
                           "epoch_index": ep, "duration_s": d})
    result = m.evaluate(trials, np.c_[x, x], {"A": np.arange(4), "B": np.arange(4, 8)}, tmp_path, n_null=2, seed=9)
    assert len(result) == 12
    assert not result[result.model == "conditional_identity"].nominal_pilot_screen_pass.any()
    for name in ("run_cv_summary.csv", "run_fold_metrics.csv", "run_predictions.csv", "null_scores.csv", "route_confusion.csv", "fold_plan.json"):
        assert (tmp_path / name).stat().st_size > 0
    predictions = pd.read_csv(tmp_path / "run_predictions.csv")
    assert not predictions.duplicated(["group", "scheme", "model", "trial_id"]).any()

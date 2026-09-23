import numpy as np
import pytest

from scripts.simulate_hc11_phase_order_identifiability import (
    equilibrium,
    generate,
    make_world,
    predictive_gain,
    stochastic,
    transport_transition,
)


def test_transition_transport_constraints():
    rng = np.random.default_rng(4)
    original = rng.dirichlet(np.ones(6), 6)
    target = rng.dirichlet(np.ones(6), 6)
    result = transport_transition(original, target)
    pi = equilibrium(target)
    np.testing.assert_allclose(result.sum(1), 1, atol=1e-11)
    np.testing.assert_allclose(pi @ result, pi, atol=1e-11)
    np.testing.assert_allclose(np.diag(result), np.diag(target), atol=1e-12)
    def odds(a):
        return a[0, 2] * a[1, 3] / (a[0, 3] * a[1, 2])
    assert odds(result) == pytest.approx(odds(original), rel=1e-10)


def test_nuisance_only_transport_recovers_generating_matrix():
    pre, post = make_world(42, "occupancy_dwell_only")
    np.testing.assert_allclose(transport_transition(pre[0], post[0]), post[0], atol=1e-10)
    np.testing.assert_allclose(transport_transition(post[0], pre[0]), pre[0], atol=1e-10)


def test_generative_controls_act_on_named_quantities_only():
    pre, post = make_world(42, "emission_only")
    np.testing.assert_array_equal(pre[0], post[0])
    assert not np.allclose(pre[1], post[1])
    assert pre[2] == post[2]
    pre, post = make_world(42, "order_changed")
    np.testing.assert_array_equal(pre[1], post[1])
    np.testing.assert_allclose(equilibrium(pre[0]), equilibrium(post[0]), atol=1e-12)
    np.testing.assert_array_equal(np.diag(pre[0]), np.diag(post[0]))
    assert not np.allclose(pre[0], post[0])


def test_heldout_spikes_cannot_change_predictive_distributions():
    pre, post = make_world(33, "order_changed")
    x = generate(np.random.default_rng(5), pre, 1, 20)[0]
    train, held = np.arange(18), np.arange(18, 24)
    _, _, before = predictive_gain(x, pre[1], pre[0], post[0], train, held)
    modified = x.copy()
    modified[:, held] += 100
    _, _, after = predictive_gain(modified, pre[1], pre[0], post[0], train, held)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a, b)


def test_unobserved_gap_and_future_cannot_change_origin_prediction():
    pre, post = make_world(33, "order_changed")
    x = generate(np.random.default_rng(5), pre, 1, 20)[0]
    train, held = np.arange(18), np.arange(18, 24)
    _, _, before = predictive_gain(x, pre[1], pre[0], post[0], train, held)
    modified = x.copy()
    modified[6:, train] += 100
    _, _, after = predictive_gain(modified, pre[1], pre[0], post[0], train, held)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_array_equal(a[:6], b[:6])


def test_true_order_control_predicts_independent_neurons():
    pre, post = make_world(7, "order_changed")
    sequences = generate(np.random.default_rng(8), pre, 100, 20)
    gains = [predictive_gain(x, pre[1], pre[0], post[0], np.arange(18), np.arange(18, 24))[0] for x in sequences]
    assert np.mean(gains) > 0


def test_identical_models_score_identically():
    pre, _ = make_world(3, "unchanged")
    x = generate(np.random.default_rng(3), pre, 1, 20)[0]
    score, spikes, _ = predictive_gain(x, pre[1], pre[0], pre[0], np.arange(18), np.arange(18, 24))
    assert score == 0 and spikes > 0


@pytest.mark.parametrize("bad", [np.ones((3, 3)), np.eye(3), np.full((3, 2), 0.5)])
def test_invalid_transition_fails(bad):
    with pytest.raises(ValueError):
        stochastic(bad)


@pytest.mark.parametrize("invalid", ["none", "duplicate_units", "unordered_epochs", "invalid_time"])
def test_preflight_native_schema(monkeypatch, tmp_path, invalid):
    from types import SimpleNamespace as NS

    from scripts.preflight_hc11_phase_order import inspect
    epoch = NS(PREEpoch=np.array([[0, 2]]), MazeEpoch=np.array([[3, 4]]), POSTEpoch=np.array([[5, 7]]))
    if invalid == "unordered_epochs":
        epoch.MazeEpoch = np.array([[1, 4]])
    times = np.empty(2, dtype=object)
    times[:] = [np.array([1., 6.]), np.array([1.5, 6.5])]
    if invalid == "invalid_time":
        times[0][0] = np.nan
    spikes = NS(UID=[1, 1] if invalid == "duplicate_units" else [1, 2], times=times,
                region=["CA1", "CA1"], _fieldnames=["rawWaveform"])
    fixtures = {"position": NS(Epochs=epoch), "spikes": spikes,
                "SleepState": NS(ints=NS(NREMstate=np.array([[0, 2], [5, 7]])))}
    monkeypatch.setattr("scripts.preflight_hc11_phase_order.mat_struct", lambda path, name: fixtures[name])
    folder = tmp_path / "Rat" / "session"
    if invalid != "none":
        with pytest.raises(ValueError):
            inspect(folder)
    else:
        result = inspect(folder)
        assert result["ca1_units_active_both"] == 2
        assert result["pre_nrem_spikes"] == result["post_nrem_spikes"] == 2
        assert result["pre_nrem_seconds"] == pytest.approx(1.9)
        assert not result["unit_drift_validated"]

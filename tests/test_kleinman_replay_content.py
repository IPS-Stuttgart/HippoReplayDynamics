import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from scripts.calibrate_kleinman_replay_content import (
    ARMS,
    early_late,
    measure,
    observations,
    posterior,
    readiness,
    select_sessions,
    sliding_windows,
    trajectory,
    weighted_correlation,
)


def model():
    centers = np.arange(1, 200, 2)
    peaks = np.linspace(1, 199, 24)
    fields = 0.01 + 30 * np.exp(-0.5 * ((centers[:, None] - peaks) / 8) ** 2)
    rates = np.block([[fields, fields * 0.01], [fields * 0.01, fields]])
    return {"rates": rates, "support": np.ones(200, bool), "edges": np.arange(0, 202, 2), "ends": np.array([10.0, 190.0])}


def test_first_pass_session_selected_without_outcome_or_accuracy_ranking():
    f = pd.DataFrame([{"animal": str(a), "session": str(s), "decoder_pass": s != 0, "mean_posterior_mean_error_cm": 20 - s} for a in range(6) for s in range(3)])
    assert set(select_sessions(f.sample(frac=1, random_state=2)).session) == {"1"}
    with pytest.raises(ValueError, match="duplicate"):
        select_sessions(pd.concat([f, f.iloc[:1]]))
    with pytest.raises(ValueError, match="six"):
        select_sessions(f.loc[f.animal != "5"])


def test_windows_fully_contained_and_counts_not_independent_replicates():
    values = np.arange(100)
    counts, centers = sliding_windows(values)
    assert len(counts) == 13
    assert counts[0] == sum(values[:40])
    assert counts[-1] == sum(values[60:100])
    np.testing.assert_allclose(centers[[0, -1]], [0.02, 0.08])
    with pytest.raises(ValueError):
        sliding_windows(np.ones(20))


def test_truth_functional_does_not_claim_unobserved_full_extent():
    x = trajectory(0.2, 0.5, 0, "linear", np.array([10.0, 190.0]))
    window_truth, centers = sliding_windows(x)
    delta = early_late(window_truth / 40, centers)
    assert 0 < delta < 90
    reverse = trajectory(0.2, 0.5, 1, "linear", np.array([10.0, 190.0]))
    np.testing.assert_allclose(reverse + x, 200)
    for profile in ["cosine", "pause_step"]:
        a = trajectory(0.2, 0.5, 0, profile, np.array([10.0, 190.0]))
        assert np.all(np.diff(a) >= 0)


def test_oracle_poisson_matches_direct_likelihood_and_support():
    m = model()
    m["support"][:2] = False
    counts = np.arange(48)[None, :] % 3
    gain = 3.2
    p = posterior(counts, m["rates"], m["support"], gain, "matched_gain_poisson")
    means = 0.04 * gain * m["rates"]
    direct = (counts * np.log(means) - means).sum(axis=1)
    direct[~m["support"]] = -np.inf
    np.testing.assert_allclose(p[0], np.exp(direct - logsumexp(direct)))
    assert p[0, :2].sum() == 0


def test_conditional_decoder_invariant_to_gain_and_silent_windows_flat():
    m = model()
    counts = np.ones((2, 48))
    counts[0] = 0
    a = posterior(counts, m["rates"], m["support"], 1, "count_conditioned")
    b = posterior(counts, m["rates"] * 17, m["support"], 17, "count_conditioned")
    np.testing.assert_allclose(a, b, atol=1e-13)
    np.testing.assert_allclose(a[0], 1 / 200)
    run = posterior(counts, m["rates"], m["support"], 17, "run_rate_poisson")
    original = posterior(counts, m["rates"], m["support"], 1, "run_rate_poisson")
    np.testing.assert_array_equal(run, original)


def test_same_seed_same_spikes_and_replay_recovers_on_informative_map():
    m = model()
    args = {"duration": 0.2, "extent": 0.75, "side": 0, "profile": "linear", "expected_spikes": 500, "event_index": 12}
    a, b = observations(m, **args), observations(m, **args)
    np.testing.assert_array_equal(a["counts"], b["counts"])
    assert a["n_spikes"] > 400 and a["true_support_fraction"] == 1
    p = posterior(a["counts"], m["rates"], m["support"], a["gain"], "matched_gain_poisson")
    scores = measure(p, m, a["centers"], 0)
    truth = early_late(a["truth_windows"], a["centers"])
    assert abs(scores["displacement_cm"] - truth) < 8
    assert scores["reverse_content_call"]
    assert scores["incoming_direction_mass"] > 0.9


def test_static_truth_and_unavailable_support_not_hidden():
    m = model()
    m["support"][:] = False
    a = observations(m, duration=0.1, extent=0, side=1, profile="linear", expected_spikes=48, event_index=0)
    assert a["true_support_fraction"] == 0
    assert early_late(a["truth_windows"], a["centers"]) == pytest.approx(0)
    with pytest.raises(ValueError):
        posterior(a["counts"], m["rates"], m["support"], 1, ARMS[0])


def test_weighted_correlation_and_static_map_do_not_assert_order():
    t = np.arange(8.0)
    x = np.arange(8.0) * 10
    assert weighted_correlation(np.eye(8), t, x) == pytest.approx(1)
    assert weighted_correlation(np.eye(8)[:, ::-1], t, x) == pytest.approx(-1)
    assert weighted_correlation(np.ones((8, 8)), t, x) == pytest.approx(0)


def perfect_frame():
    return pd.DataFrame(
        [
            {
                "arm": arm,
                "animal": str(animal),
                "side": side,
                "duration_s": 0.2,
                "expected_spikes": count,
                "extent": extent,
                "profile": "linear",
                "error_fraction": 0.0,
                "absolute_error_fraction": 0.0,
                "weighted_correlation": 0.0,
                "true_support_fraction": 1.0,
                "displacement_fraction": extent * 0.6,
                "reverse_content_call": extent > 0,
            }
            for arm in ARMS
            for animal in range(6)
            for side in range(2)
            for count in [48, 96]
            for extent in [0, 0.25, 0.5, 0.75]
            for _ in range(16)
        ]
    )


def test_readiness_failures_are_nonvacuous_and_not_pooled_away():
    f = perfect_frame()
    _, summary = readiness(f)
    assert summary.passed.all()
    f.loc[(f.animal == "5") & (f.extent == 0), "reverse_content_call"] = True
    gates, summary = readiness(f)
    assert not summary.passed.any()
    assert not gates.loc[gates.animal == "5", "static_specificity"].any()
    _, summary = readiness(f.loc[f.animal != "5"])
    assert not summary.passed.any()
    _, summary = readiness(f.iloc[:0])
    assert not summary.passed.any()


def test_count_dependent_displacement_bias_and_missing_repeats_fail():
    f = perfect_frame()
    f.loc[f.expected_spikes == 96, "error_fraction"] = 0.2
    gates, _ = readiness(f)
    assert not gates.count_stability.any()
    f = perfect_frame().iloc[1:]
    gates, summary = readiness(f)
    assert not gates.complete.all()
    assert not summary.passed.all()

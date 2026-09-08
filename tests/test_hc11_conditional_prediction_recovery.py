from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from scripts import simulate_hc11_conditional_prediction_recovery as recovery


def fixture():
    data = {
        "centers": np.arange(4) * 4.0 + 2.0,
        "counts_POST_1": np.array([[0, 0, 0, 0, 0], [2, 0, 1, 0, 1], [0, 1, 0, 2, 0], [1, 1, 0, 0, 0]]),
        "rates_direction_mixture_0": np.array([[5, 1, 1, 1], [1, 5, 1, 1], [1, 1, 5, 1], [1, 1, 1, 5], [2, 3, 4, 2]], dtype=float),
    }
    data["rates_direction_mixture_1"] = data["rates_direction_mixture_0"][:, ::-1].copy()
    edges = np.arange(5) * 0.02
    kernels = recovery.frozen.transitions(data["centers"], edges, "circular", 16.0)
    return data, edges, kernels


@pytest.mark.parametrize("generator", recovery.GENERATORS)
def test_reproducibility_and_exact_count_profile(generator):
    data, _edges, kernels = fixture()
    first = recovery.generated_event(data, "POST_1", generator, 0, "example", kernels)
    second = recovery.generated_event(data, "POST_1", generator, 0, "example", kernels)
    for a, b in zip(first, second, strict=True):
        assert_allclose(a, b, atol=0)
    counts, path, modes, direction, seed = first
    assert np.array_equal(counts.sum(axis=1), data["counts_POST_1"].sum(axis=1))
    assert np.all(counts[0] == 0)
    assert np.all((path >= 0) & (path < 4))
    assert direction in (0, 1)
    if generator == "static_location":
        assert len(set(path)) == 1
    if generator == "first_order_imm":
        assert np.all((modes >= 0) & (modes < 3))
    else:
        assert np.all(modes == -1)
    assert recovery.generated_event(data, "POST_1", generator, 1, "example", kernels)[-1] != seed


def test_invalid_generator_and_totals_fail():
    _, _, kernels = fixture()
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        recovery.draw_latent("unknown", 4, 4, kernels, rng)
    with pytest.raises(ValueError):
        recovery.draw_population([1.5, 2], np.ones((3, 4)), [0, 1], rng)
    with pytest.raises(ValueError):
        recovery.draw_population([-1, 2], np.ones((3, 4)), [0, 1], rng)
    with pytest.raises(ValueError):
        recovery.draw_population([1, 2], np.zeros((3, 4)), [0, 1], rng)


def test_heldout_changes_do_not_change_inference(monkeypatch):
    data, edges, kernels = fixture()
    rates = {"direction_mixture": [data[f"rates_direction_mixture_{d}"] for d in range(2)]}
    counts = data["counts_POST_1"].copy()
    train = np.array([0, 1, 2])
    held = np.array([3, 4])
    observed = []
    original = recovery.frozen.infer_training

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result.copy())
        return result

    monkeypatch.setattr(recovery.frozen, "infer_training", capture)
    a = recovery.predict_event(counts, edges, rates, train, held, kernels)
    counts[:, held] += 10
    b = recovery.predict_event(counts, edges, rates, train, held, kernels)
    for left, right in zip(observed[:4], observed[4:], strict=True):
        assert_allclose(left, right, atol=0)
    assert a[0]["score_first_order_imm"] != b[0]["score_first_order_imm"]


def test_split_event_is_same_population():
    data, _edges, kernels = fixture()
    counts, *_ = recovery.generated_event(data, "POST_1", "diffusion", 2, "example", kernels)
    for held in ([0, 1], [1, 2], [3, 4]):
        train = np.setdiff1d(np.arange(5), held)
        assert np.array_equal(counts[:, train].sum(axis=1) + counts[:, held].sum(axis=1), counts.sum(axis=1))


def test_bootstrap_weights_and_animal_balance():
    cohort = pd.DataFrame({"rat": ["a", "a", "b", "b", "b", "b"], "session": ["a1", "a1", "b1", "b1", "b2", "b2"]})
    w = recovery.bootstrap_weights(cohort, draws=3000, seed=19)
    assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert abs(w[:, :2].sum(axis=1).mean() - 0.5) < 0.025
    assert_allclose(w, recovery.bootstrap_weights(cohort, draws=3000, seed=19), atol=0)


def test_paired_medians_and_tie_abstention():
    scores = pd.DataFrame(
        [
            {
                "session": "s",
                "rat": "r",
                "phase": "POST",
                "event_index": 1,
                "generator": "diffusion",
                "replicate": 0,
                "encoding_variant": "pooled",
                "split": i,
                "score_iid_position": -20.0,
                "score_static_location": -20.0,
                "score_diffusion": -20.0 + gain,
                "score_first_order_imm": -20.0 + 2 * gain,
                "score_oracle": -2.0,
            }
            for i, gain in enumerate([0, 1, 2, 3, 100])
        ]
    )
    event = recovery.event_contrasts(scores.copy()).iloc[0]
    assert event.imm_minus_iid_position == 4
    assert event.imm_minus_diffusion == 2
    assert event.raw_predictive_winner == "first_order_imm"
    for col in scores.columns:
        if col.startswith("score_"):
            scores[col] = -20.0
    assert recovery.event_contrasts(scores).iloc[0].raw_predictive_winner == "ambiguous"
    with pytest.raises(ValueError):
        recovery.event_contrasts(pd.concat([scores, scores.iloc[:1]]))


def test_zero_observed_errors_not_zero_upper_bound():
    low, high = recovery.binomial_interval(0, 50)
    assert low == 0 and 0.05 < high < 0.08
    low, high = recovery.binomial_interval(50, 50)
    assert 0.92 < low < 0.95 and high == 1
    with pytest.raises(ValueError):
        recovery.binomial_interval(0, 0)


def test_pattern_requires_both_primary_axes(monkeypatch):
    monkeypatch.setattr(recovery, "REPLICATES", 2)
    panels = pd.DataFrame(
        [
            {
                "phase": "POST",
                "generator": "diffusion",
                "encoding_variant": "pooled",
                "replicate": rep,
                "contrast": contrast,
                "equal_animal_mean": 1.0,
                "ci_low": low,
                "ci_high": 2.0,
                "positive_animals": 4,
                "n_animals": 4,
            }
            for rep in range(2)
            for contrast, low in (("imm_minus_iid_position", 0.2), ("imm_minus_static_location", -0.1 if rep == 1 else 0.1))
        ]
    )
    real = pd.DataFrame(
        [{"phase": "POST", "encoding_variant": "pooled", "inference_temperature": 1.0, "contrast": "imm_minus_iid_position", "equal_animal_mean_event_median_delta": 0.0}]
    )
    summary, patterns = recovery.summarize(panels, real)
    assert patterns.iloc[0].positive_pattern_fraction == 0.5
    assert summary.iloc[0].replicates == 2

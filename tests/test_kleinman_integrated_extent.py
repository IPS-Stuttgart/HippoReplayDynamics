import numpy as np
import pandas as pd
import pytest

from scripts.calibrate_kleinman_integrated_extent import (
    condition_summary,
    fit_events,
    replay_counts,
    screens,
    template_library,
)
from scripts.calibrate_kleinman_replay_content import observations


def model():
    x = np.arange(1, 200, 2)
    fields = 0.01 + 50 * np.exp(-0.5 * ((x[:, None] - np.linspace(1, 199, 30)) / 7) ** 2)
    return {"rates": np.block([[fields, 0.01 * fields], [0.01 * fields, fields]]), "edges": np.arange(0, 202, 2), "ends": np.array([10.0, 190.0]), "support": np.ones(200, bool)}


def test_library_unknown_origin_direction_and_timing_exclusion():
    logq, templates = template_library(model(), 0.2)
    assert len(templates) == 306
    assert set(templates.direction) == {0, 1}
    assert templates.start.nunique() == templates.end.nunique() == 9
    assert set(templates.profile) == {"linear", "cosine", "static"}
    assert (templates.extent == 0).sum() == 18
    assert logq.shape == (306, 20, 60)
    np.testing.assert_allclose(np.exp(logq).sum(axis=2), 1)
    with pytest.raises(ValueError):
        template_library(model(), 0.205)


def test_integrated_probabilities_include_motion_within_window():
    m = model()
    logq, meta = template_library(m, 0.1)
    k = meta.index[(meta.start == 0) & (meta.end == 1) & (meta.direction == 0) & (meta.profile == "linear")][0]
    x = 10 + 180 * (np.arange(100) + 0.5) / 100
    b = np.digitize(x, m["edges"]) - 1
    first_rates = m["rates"][b[:10]].sum(axis=0)
    np.testing.assert_allclose(np.exp(logq[k, 0]), first_rates / first_rates.sum())
    midpoint_rates = m["rates"][b[5]]
    assert not np.allclose(np.exp(logq[k, 0]), midpoint_rates / midpoint_rates.sum())


def test_same_spikes_reconstructed_without_new_bank():
    m = model()
    a = observations(m, duration=0.2, extent=0.5, side=0, profile="cosine", expected_spikes=400, event_index=4)
    row = pd.Series(
        {
            "duration_s": 0.2,
            "extent": 0.5,
            "side": 0,
            "profile": "cosine",
            "expected_spikes": 400,
            "event_index": 4,
            "n_spikes": a["n_spikes"],
            "n_active_units": a["n_active_units"],
        }
    )
    counts = replay_counts(m, row)
    assert counts.shape == (20, 60) and counts.sum() == a["n_spikes"]
    logq, meta = template_library(m, 0.2)
    fit = fit_events(counts[None], logq, meta).iloc[0]
    assert fit.estimated_extent == pytest.approx(0.5)
    assert fit.estimated_start == pytest.approx(0)
    assert fit.crossfit_moving_minus_static > 0
    row.n_spikes += 1
    with pytest.raises(ValueError, match="reconstruction"):
        replay_counts(m, row)


def test_global_gain_cancels_and_support_exclusions_explicit():
    m = model()
    q, t = template_library(m, 0.2)
    m["rates"] *= 12
    q2, t2 = template_library(m, 0.2)
    np.testing.assert_allclose(q, q2, atol=1e-12)
    pd.testing.assert_frame_equal(t, t2)
    m["support"][:] = False
    with pytest.raises(ValueError):
        template_library(m, 0.2)


def test_crossfit_score_matches_independent_train_only_argmax():
    m = model()
    q, meta = template_library(m, 0.1)
    counts = np.random.default_rng(9).poisson(0.1, size=(3, 10, 60))
    fit = fit_events(counts, q, meta)
    for event in range(3):
        delta = 0.0
        for parity in [0, 1]:
            train = np.arange(10) % 2 == parity
            scores = []
            for static in [True, False]:
                ids = np.flatnonzero((meta.extent.to_numpy() == 0) == static)
                train_values = np.array([(counts[event, train] * q[k, train]).sum() for k in ids])
                best = ids[train_values >= train_values.max() - 1e-10]
                scores.append(np.mean([(counts[event, ~train] * q[k, ~train]).sum() for k in best]))
            delta += scores[1] - scores[0]
        assert fit.iloc[event].crossfit_moving_minus_static == pytest.approx(delta)


def perfect_frame():
    return pd.DataFrame(
        [
            {
                "animal": str(animal),
                "session": "s",
                "side": side,
                "profile": profile,
                "extent": extent,
                "duration_s": duration,
                "expected_spikes": count,
                "event_index": repeat,
                "estimated_extent": extent,
                "likelihood_weighted_extent": extent,
                "crossfit_moving_minus_static": 10 if extent else -10,
            }
            for animal in range(6)
            for side in [0, 1]
            for extent, profile in [(0, "linear")] + [(e, p) for e in [0.25, 0.5, 0.75] for p in ["linear", "cosine", "pause_step"]]
            for duration in [0.1, 0.2, 0.4]
            for count in [24, 48, 96]
            for repeat in range(16)
        ]
    )


def test_all_screens_and_nuisance_failures():
    f = perfect_frame()
    assert len(condition_summary(f)) == 1080
    gates = screens(f)
    assert len(gates) == 36 and gates.passed.all()
    f.loc[(f.profile == "pause_step") & (f.expected_spikes == 96), "estimated_extent"] += 0.2
    gates = screens(f)
    assert gates.loc[gates.screen == "matched_timing", "passed"].all()
    assert not gates.loc[gates.screen == "unseen_timing", "passed"].any()
    f = perfect_frame()
    f.loc[f.duration_s == 0.4, "estimated_extent"] += 0.2
    assert not screens(f).query("screen == 'duration_only'").passed.any()


def test_static_overfit_and_missing_repeats_cannot_pass():
    f = perfect_frame()
    f.loc[f.extent == 0, "estimated_extent"] = 0.25
    gates = screens(f)
    assert not gates.loc[gates.screen != "duration_only", "static_specificity"].any()
    f = perfect_frame()
    f = f.loc[~((f.animal == "0") & (f.duration_s == 0.2) & (f.event_index == 1))]
    gates = screens(f)
    assert not gates.loc[(gates.animal == "0"), "passed"].any()


def test_zero_spikes_exposes_ties_and_no_predictive_advantage():
    q, meta = template_library(model(), 0.1)
    fit = fit_events(np.zeros((1, 10, 60)), q, meta).iloc[0]
    assert fit.n_best_ties == 306
    assert fit.crossfit_moving_minus_static == 0
    with pytest.raises(ValueError):
        fit_events(np.zeros((1, 9, 60)), q, meta)

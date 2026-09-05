"""Interval semantics, exchangeability boundaries, no test-data fitting."""

import numpy as np
import pandas as pd
import pytest

from hipporeplayimm.replay_speed_identifiability import (
    bootstrap_slope,
    conformal_radius,
    event_slope,
    fit_inverse,
    interval_decision,
    inverse_intervals,
)
from scripts.calibrate_replay_speed_identifiability import draw_schedule, evaluate_panels, summarize_decisions


def test_equal_event_slope_and_unsupported_panels():
    moments = [[x, 1+.3*x, x*x+.1, x*(1+.3*x)+.03] for x in np.linspace(-1, 1, 12)]
    assert event_slope(moments) == pytest.approx(.3)
    point, lower, upper = bootstrap_slope(moments, 77)
    assert lower == pytest.approx(point) and upper == pytest.approx(point)
    assert np.isnan(event_slope(moments[:4]))
    assert bootstrap_slope(moments[:4], 77)[1:] == (-np.inf, np.inf)


def test_inverse_corrects_offset_and_attenuation_without_test_labels():
    g = np.linspace(-.75, .75, 40)
    model = fit_inverse(.3+.2*g, g)
    assert model["slope"] == pytest.approx(5)
    assert model["intercept"] == pytest.approx(-1.5)
    intervals = inverse_intervals(model, .1, .3)
    assert intervals["inverse_conformal"] == pytest.approx((0, -.1, .1))
    assert all(a[1:] == (-np.inf, np.inf) for a in inverse_intervals(model, .1, np.nan).values())
    assert fit_inverse(np.ones(40), g)["status"] == "uninformative_fit"


def test_finite_sample_quantile_missingness_not_dropped():
    model = fit_inverse(np.arange(40.), np.arange(40.))
    assert conformal_radius(model, np.zeros(19), np.arange(19.)) == 18
    assert np.isinf(conformal_radius(model, np.zeros(18), np.arange(18.)))
    x = np.zeros(99)
    x[-10:] = np.nan
    assert np.isinf(conformal_radius(model, x, np.zeros(99)))
    with pytest.raises(ValueError):
        conformal_radius(model, [1], [1, 2])


def test_missing_fit_and_unbounded_coverage_are_not_informative():
    model = fit_inverse(np.full(40, np.nan), np.zeros(40))
    assert model["status"] == "insufficient_fit"
    decision = interval_decision(-np.inf, np.inf, 0., .25)
    assert decision["covered"] and not decision["finite_interval"]
    assert not decision["equivalence_claim"] and decision["decision"] == "abstain"


def test_equivalence_boundary_and_nonzero_can_be_distinct():
    assert interval_decision(.1, .2, .15, .25)["nonzero_claim"]
    assert interval_decision(.1, .2, .15, .25)["equivalence_claim"]
    assert interval_decision(-.1, .1, .25, .25)["false_equivalence"]
    assert not interval_decision(-.25, .25, 0, .25)["equivalence_claim"]
    with pytest.raises(ValueError):
        interval_decision(1, 0, .1, .25)


def fixture_panels():
    rows = []
    for phase, count in [("fit", 40), ("calibration", 99), ("test", 8)]:
        for index, g in enumerate(np.linspace(-.7, .7, count)):
            rows.append({"dataset": "test", "animal": "R1", "session": "S1", "phase": phase,
                "draw_id": index, "generator": "A", "observation": "poisson", "stratum": "uniform",
                "estimator": "posterior_mean", "bin_filter": "unfiltered", "selection": "all",
                "gradient": g, "statistic": .3+.2*g, "naive_lower": .2*g, "naive_upper": .6+.2*g})
    return pd.DataFrame(rows)


def test_test_outcomes_cannot_change_fit_or_calibration():
    frame = fixture_panels()
    decisions, before = evaluate_panels(frame)
    changed = frame.copy()
    changed.loc[changed.phase.eq("test"), "gradient"] = -100
    changed.loc[changed.phase.eq("test"), "statistic"] = 100
    _, after = evaluate_panels(changed)
    pd.testing.assert_frame_equal(before, after)
    assert len(decisions) == 24
    broken = frame.copy()
    broken.loc[0, "generator"] = "B"
    with pytest.raises(ValueError, match="leaked"):
        evaluate_panels(broken)


def test_summary_reports_all_and_finite_denominators_separately():
    frame = fixture_panels()
    frame["statistic"] = np.nan
    frame["naive_lower"], frame["naive_upper"] = -np.inf, np.inf
    decisions, _ = evaluate_panels(frame)
    summary = summarize_decisions(decisions)
    assert summary.coverage.eq(1).all() and summary.finite_fraction.eq(0).all()
    assert summary.finite_coverage.isna().all() and summary.equivalence_fraction.eq(0).all()


def test_schedule_frozen_unique_and_parameter_distribution_separate():
    a = draw_schedule(1, "A", 20, 19, 3, 2)
    pd.testing.assert_frame_equal(a, draw_schedule(1, "A", 20, 19, 3, 2))
    assert not a.duplicated(["phase", "draw_id"]).any()
    assert a.draw_seed.nunique() == len(a)
    assert a[a.phase.eq("test")].stratum.value_counts().to_dict() == {
        "uniform": 3, **{f"fixed_{g:+.2f}": 2 for g in [-.5, -.25, 0., .25, .5]}}
    assert not a.equals(draw_schedule(1, "B", 20, 19, 3, 2))


def test_conformal_noise_coverage_is_empirical_not_guaranteed_at_fixed_g():
    rng = np.random.default_rng(73)
    fit_g = rng.uniform(-.75, .75, 100)
    model = fit_inverse(.2+.4*fit_g+rng.normal(0, .07, 100), fit_g)
    cal_g = rng.uniform(-.75, .75, 999)
    radius = conformal_radius(model, .2+.4*cal_g+rng.normal(0, .07, 999), cal_g)
    test_g = rng.uniform(-.75, .75, 5000)
    test_x = .2+.4*test_g+rng.normal(0, .07, 5000)
    coverage = np.mean(np.abs(test_g-(model["intercept"]+model["slope"]*test_x)) <= radius)
    assert .92 < coverage < .98
    shifted = np.mean(np.abs(test_g-(model["intercept"]+model["slope"]*(test_x+1))) <= radius)
    assert shifted < .2


def test_panel_readout_uses_decoded_coordinate_and_does_not_bridge_gaps(monkeypatch):
    import scripts.calibrate_replay_speed_identifiability as scorer

    fine = np.ones((200, 2), int)
    fine[65:110] = 0
    counts = np.array([fine[t:t+20].sum(axis=0) for t in range(0, 181, 5)])
    x = 5*np.arange(len(counts), dtype=float)
    x[15:19] += 100
    points = np.column_stack([x, np.zeros(len(x))])
    monkeypatch.setattr(scorer, "decode_independent", lambda *a, **k: {
        "map": np.tile(points, (5, 1)), "posterior_mean": np.tile(points, (5, 1)), "posterior": None})
    model = {"rates_hz": np.ones((2, 2)), "grid_cm": np.zeros((2, 2)), "domain_cm": np.array([[0., 0.], [300., 200.]])}
    result = scorer.decode_panel([fine]*5, model, 7, False)
    valid = (counts.sum(axis=1) >= 3) & ((counts > 0).sum(axis=1) >= 2)
    q, speed = [], []
    for start in range(0, len(points)-4, 4):
        if all(valid[t] for t in range(start, start+5)):
            q.append((points[start, 0]+points[start+4, 0])/300.-1)
            speed.append(np.linalg.norm(points[start+4]-points[start])/.02/1000.)
    q, speed = np.array(q), np.array(speed)
    expected = np.mean((q-q.mean())*(speed-speed.mean()))/np.var(q)
    filtered = result[result.selection.eq("all") & result.bin_filter.eq("at_least_2cells_3spikes")]
    np.testing.assert_allclose(filtered.statistic, expected, atol=1e-12)
    assert filtered.valid_windows.eq(int(valid.sum())*5).all()
    assert not np.isclose(result[result.selection.eq("all") & result.bin_filter.eq("unfiltered")].statistic.iloc[0], expected)


def test_panel_contract_catches_missing_condition_and_wrong_draw_metadata():
    from io import StringIO
    from itertools import product

    from scripts.calibrate_replay_speed_identifiability import TEST_CONDITIONS, validate_panel_contract

    identity = {"dataset": "test", "animal": "R1", "session": "S1"}
    schedule = draw_schedule(12, "test", 1, 1, 1, 1)
    rows = []
    for draw in schedule.to_dict("records"):
        conditions = TEST_CONDITIONS if draw["phase"] == "test" else [("A", "poisson")]
        for (generator, observation), estimator, support, selection in product(conditions, ["map", "posterior_mean"], ["unfiltered", "at_least_2cells_3spikes"], ["all", "selected"]):
            rows.append({**identity, **draw, "generator": generator, "observation": observation, "estimator": estimator, "bin_filter": support, "selection": selection})
    panels = pd.DataFrame(rows)
    validate_panel_contract(panels, schedule, identity)
    validate_panel_contract(pd.read_csv(StringIO(panels.to_csv(index=False)), float_precision="round_trip"), schedule, identity)
    with pytest.raises(ValueError, match="missing"):
        validate_panel_contract(panels.iloc[:-1], schedule, identity)
    changed = panels.copy()
    changed.loc[0, "gradient"] += .01
    with pytest.raises(ValueError, match="schedule mismatch"):
        validate_panel_contract(changed, schedule, identity)
    changed = panels.copy()
    changed.loc[0, "generator"] = "C"
    with pytest.raises(ValueError, match="unexpected"):
        validate_panel_contract(changed, schedule, identity)


def test_report_monte_carlo_intervals_keep_missing_and_conditional_denominators():
    from scripts.report_replay_speed_identifiability import monte_carlo_intervals

    panels = fixture_panels()
    panels["statistic"] = np.nan
    panels["naive_lower"], panels["naive_upper"] = -np.inf, np.inf
    decisions, _ = evaluate_panels(panels)
    table = summarize_decisions(decisions)
    result = monte_carlo_intervals(table)
    finite = result[result.metric.eq("finite_coverage")]
    assert finite.denominator.eq(0).all()
    assert finite[["estimate", "mc95_low", "mc95_high"]].isna().all().all()
    coverage = result[result.metric.eq("coverage")]
    assert coverage.numerator.eq(8).all() and coverage.denominator.eq(8).all()
    np.testing.assert_allclose(coverage.mc95_low, .6755924, atol=1e-7)
    np.testing.assert_allclose(coverage.mc95_high, 1)
    outside = result[result.metric.eq("false_equivalence_fraction_0.25")]
    assert outside.denominator.eq(6).all() and outside.numerator.eq(0).all()
    assert outside.mc95_high.gt(0).all()
    invalid = table.copy()
    invalid.loc[0, "coverage"] = .123
    with pytest.raises(ValueError, match="integer"):
        monte_carlo_intervals(invalid)

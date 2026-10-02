"""Bounded real-period numerical verification; not scientific calibration."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.measure_run_pair_coordination_endpoint import period_endpoint
from scripts.verify_run_pair_glm_endpoint import global_predictions, independently_measure_period


def fixture():
    p = json.loads((Path(__file__).parents[1] / "docs/run_pair_coordination_endpoint_v3_protocol.json").read_text())
    p["glm_workers"] = 1
    times = np.arange(.0005, 20, .001)
    phase = np.column_stack((np.mod(2 * np.pi * 8 * times, 2 * np.pi) - np.pi,
                             np.mod(2 * np.pi * 8 * times + .3, 2 * np.pi) - np.pi))
    mu = .02 * np.exp(np.cos(phase))
    bank = {"pre_time_s": times, "pre_counts": np.random.default_rng(22).poisson(mu),
        "pre_theta_phase_rad": phase, "pre_position_cm": np.column_stack((5 + np.sin(times), 5 + np.cos(times))),
        "pre_direction_rad": np.mod(times, 2 * np.pi), "pre_speed_cm_s": np.full(len(times), 15.),
        "pre_bin_duration_s": np.full(len(times), .001)}
    return p, bank


def test_independent_real_period_matches_producer_endpoint_quality_and_lag_support():
    p, bank = fixture()
    original, producer_quality = period_endpoint(bank, "pre", p)
    independent, quality = independently_measure_period(bank, "pre", p)
    np.testing.assert_allclose(independent, original, rtol=1e-3, atol=2e-7)
    assert quality["physical_lag_opportunities"] == producer_quality["physical_lag_opportunities"]
    assert quality["usable_bins"] == quality["source_bins"] == 20000
    assert quality["heldout_poisson_improvement_over_global"] > 0
    np.testing.assert_allclose(quality["heldout_poisson_improvement_over_global"],
                              producer_quality["heldout_poisson_improvement_over_global"], rtol=1e-3)


def test_independent_global_rate_never_uses_heldout_block_counts():
    p, bank = fixture()
    counts, times = bank["pre_counts"], bank["pre_time_s"]
    first = global_predictions(counts, times, p)
    target = np.floor(times / p["crossfit_time_block_s"]).astype(int) % 2 == 0
    changed = counts.copy()
    changed[target] += 1000
    second = global_predictions(changed, times, p)
    np.testing.assert_array_equal(first[target], second[target])


def test_independent_refit_keeps_original_physical_clock_when_theta_is_missing():
    p, bank = fixture()
    bank["pre_theta_phase_rad"][1000:1200] = np.nan
    _, quality = independently_measure_period(bank, "pre", p)
    assert quality["source_bins"] == 20000 and quality["usable_bins"] == 19800
    _, original = period_endpoint(bank, "pre", p)
    assert quality["physical_lag_opportunities"] == original["physical_lag_opportunities"]


def test_independent_refit_rejects_out_of_domain_speed_without_clipping_it_to_a_pass():
    p, bank = fixture()
    bank["pre_speed_cm_s"][0] = 201
    with pytest.raises(ValueError, match="speed outside"):
        independently_measure_period(bank, "pre", p)

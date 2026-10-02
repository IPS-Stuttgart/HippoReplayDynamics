"""Train-only nuisance fitting checks; not full biological null calibration."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import _run_pair_rate_glm as glm
from scripts import measure_run_pair_coordination_endpoint as endpoint
from scripts.measure_run_pair_coordination_endpoint import crossfit_rates


def protocol():
    p = json.loads((Path(__file__).parents[1] / "docs/run_pair_coordination_endpoint_v2_protocol.json").read_text())
    return {**p, "glm_workers": 1}


def covariates(times):
    return {"position": np.column_stack((np.sin(times) + 5, np.cos(times) + 5)),
            "direction": np.mod(times, 2 * np.pi), "speed": np.full(len(times), 15.),
            "theta": np.column_stack((np.mod(2 * np.pi * 8 * times, 2 * np.pi) - np.pi,
                                      np.mod(2 * np.pi * 8 * times + .4, 2 * np.pi) - np.pi))}


def test_validation_coordinates_cannot_change_training_knots_scaling_or_features():
    times = np.arange(.0005, 20, .001)
    train, heldout = covariates(times[:10000]), covariates(times[10000:])
    first, _, _ = glm.design(train, heldout, protocol())
    heldout["position"] += 10000
    heldout["speed"][:] = 100
    second, _, qc = glm.design(train, heldout, protocol())
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    assert qc["outside_training_spatial_range_fraction"] == 1


def test_crossfit_heldout_counts_do_not_enter_fitted_prediction():
    p = protocol()
    times = np.arange(.0005, 20, .001)
    c = covariates(times)
    mu = .02 * np.exp(np.cos(c["theta"][:, 0]))
    counts = np.random.default_rng(8).poisson(mu)[:, None]
    original, qc = crossfit_rates(counts, times, np.zeros(len(times), int), .001, p, covariates=c)
    altered = counts.copy()
    heldout = np.floor(times / 5).astype(int) % 2 == 0
    altered[heldout] += 1
    changed, _ = crossfit_rates(altered, times, np.zeros(len(times), int), .001, p, covariates=c)
    def means(y, residual):
        return ((np.sqrt(residual**2 + 4 * y) - residual) / 2)**2
    np.testing.assert_allclose(means(counts[heldout], original[heldout]),
                               means(altered[heldout], changed[heldout]), rtol=1e-8, atol=1e-12)
    assert qc["predicted_bins"] == len(times)
    assert qc["heldout_poisson_improvement_over_global"] > 0
    assert all(f["all_cells_converged"] for f in qc["folds"])


def test_zero_spike_cells_and_windows_are_retained_with_a_training_only_prior():
    times = np.arange(.0005, 20, .001)
    residual, qc = crossfit_rates(np.zeros((len(times), 1), int), times, np.zeros(len(times), int),
                                  .001, protocol(), covariates=covariates(times))
    assert np.isfinite(residual).all() and qc["predicted_bins"] == len(times)


def test_prepared_covariate_cache_is_hash_bound_and_never_contains_spikes():
    times = np.arange(.0005, 20, .001)
    c, p = covariates(times), protocol()
    prepared = glm.prepare(times, c, p)
    assert all("counts" not in item for item in prepared["folds"])
    c["theta"][0, 0] += .01
    with pytest.raises(ValueError, match="identity"):
        crossfit_rates(np.zeros((len(times), 1)), times, np.zeros(len(times), int), .001, p,
                        covariates=c, prepared=prepared)


def test_missing_lfp_phase_is_not_replaced_by_a_spike_derived_control():
    times = np.arange(.0005, 20, .001)
    c = covariates(times)
    c["theta"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        glm.prepare(times, c, protocol())


def test_missing_crossfit_training_returns_nan_not_a_pass():
    times = np.arange(.0005, .2, .001)
    residual, qc = crossfit_rates(np.zeros((len(times), 1), int), times, np.zeros(len(times), int),
                                  .001, protocol(), covariates=covariates(times))
    assert np.isnan(residual).all() and qc["predicted_bins"] == 0


def test_explicit_nonconvergence_cannot_be_used_to_drop_difficult_cells():
    p = {**protocol(), "glm_max_iter": 1}
    times = np.arange(.0005, 20, .001)
    counts = np.zeros((len(times), 2), int)
    counts[::50, 0] = 1
    with pytest.warns(Warning):
        residual, qc = crossfit_rates(counts, times, np.zeros(len(times), int), .001, p,
                                      covariates=covariates(times))
    assert np.isnan(residual).all() and qc["predicted_bins"] == 0
    assert all(f["status"] == "glm_nonconvergence" for f in qc["folds"])


def test_one_failed_fold_invalidates_period_not_just_unfavorable_windows(monkeypatch):
    times = np.arange(.0005, 20, .001)
    c = covariates(times)
    bank = {"pre_time_s": times, "pre_counts": np.zeros((len(times), 2), int),
            "pre_bin_duration_s": np.full(len(times), .001)}
    for key, name in (("position", "position_cm"), ("direction", "direction_rad"),
                      ("speed", "speed_cm_s"), ("theta", "theta_phase_rad")):
        bank[f"pre_{name}"] = c[key]
    def incomplete(*args, **kwargs):
        residual = np.zeros((len(times), 2))
        residual[:100] = np.nan
        return residual, {"predicted_bins": len(times) - 100}
    monkeypatch.setattr(endpoint, "crossfit_rates", incomplete)
    result, qc = endpoint.period_endpoint(bank, "pre", protocol())
    assert result is None and qc["usable_bins"] == 0
    assert qc["status"] == "incomplete_glm_crossfit_predictions"


def test_future_period_spikes_and_covariates_cannot_change_pre_glm_endpoint():
    times = np.arange(.0005, 20, .001)
    c = covariates(times)
    bank = {"pre_time_s": times, "pre_counts": np.zeros((len(times), 2), int),
            "pre_bin_duration_s": np.full(len(times), .001), "post_counts": np.ones((5, 2), int),
            "post_theta_phase_rad": np.zeros((5, 2))}
    for key, name in (("position", "position_cm"), ("direction", "direction_rad"),
                      ("speed", "speed_cm_s"), ("theta", "theta_phase_rad")):
        bank[f"pre_{name}"] = c[key]
    bank["pre_counts"][::50, 0] = 1
    bank["pre_counts"][7::50, 1] = 1
    first, first_qc = endpoint.period_endpoint(bank, "pre", protocol())
    bank["post_counts"][:] = 1000
    bank["post_theta_phase_rad"][:] = np.nan
    second, second_qc = endpoint.period_endpoint(bank, "pre", protocol())
    np.testing.assert_array_equal(first, second)
    assert first_qc == second_qc


@pytest.mark.parametrize("key,value", [("glm_position_knot_cm", 0), ("glm_l2_penalty", np.nan),
                                       ("glm_theta_harmonics", 0), ("glm_workers", .5)])
def test_invalid_glm_settings_fail_before_fitting(key, value):
    times = np.arange(.0005, 20, .001)
    with pytest.raises(ValueError, match="GLM"):
        glm.prepare(times, covariates(times), {**protocol(), key: value})


def test_fixed_speed_domain_cannot_explode_when_target_speed_exceeds_training_range():
    times = np.arange(.0005, 20, .001)
    train, target = covariates(times[:10000]), covariates(times[10000:])
    target["speed"][:] = 200
    p = {**protocol(), "glm_speed_scaling": "fixed_log_run_bounds"}
    x, z, qc = glm.design(train, target, p)
    assert np.max(np.abs(x.data)) <= 1 + 1e-12 and np.max(np.abs(z.data)) <= 1 + 1e-12
    assert qc["outside_training_speed_range_fraction"] == 1
    assert qc["maximum_target_feature_absolute_value"] <= 1 + 1e-12

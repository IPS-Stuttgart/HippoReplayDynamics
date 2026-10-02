"""Endpoint arithmetic and leakage checks, not biological null calibration."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import measure_run_pair_coordination_endpoint as endpoint


def protocol():
    return json.loads((Path(__file__).parents[1] / "docs/run_pair_coordination_endpoint_protocol.json").read_text())


def test_joint_strata_include_position_direction_speed_and_all_lfp_references():
    p = protocol()
    position = np.array([[1., 2.]] * 5)
    direction = np.zeros(5)
    speed = np.full(5, 15.)
    phase = np.zeros((5, 2))
    position[1, 0] = 20
    direction[2] = np.pi
    speed[3] = 70
    phase[4, 1] = np.pi / 2
    assert len(np.unique(endpoint.strata(position, direction, speed, phase, p))) == 5
    phase[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        endpoint.strata(position, direction, speed, phase, p)


def test_crossfit_never_uses_heldout_block_counts():
    p = protocol()
    times = np.arange(.0005, 20, .001)
    counts = np.zeros((len(times), 2), int)
    counts[::100, 0] = 1
    counts[::80, 1] = 1
    labels = np.zeros(len(times), int)
    first, qc = endpoint.crossfit_rates(counts, times, labels, .001, p)
    changed = counts.copy()
    heldout = (np.floor(times / 5).astype(int) % 2) == 0
    changed[heldout, 0] += 2
    second, _ = endpoint.crossfit_rates(changed, times, labels, .001, p)
    # Reconstruct predicted means from residuals for bins with original zero counts.
    zero = heldout & (counts[:, 0] == 0)
    original_mu = first[zero, 0] ** 2
    new_r = second[zero, 0]
    new_mu = ((np.sqrt(new_r**2 + 4 * changed[zero, 0]) - new_r) / 2) ** 2
    np.testing.assert_allclose(original_mu, new_mu, atol=1e-12)
    assert qc["predicted_bins"] == len(times)
    assert any(np.all(row == 0) for row in counts)


def test_physical_lag_matches_independent_double_loop_without_concatenating_gaps():
    rng = np.random.default_rng(42)
    times = np.r_[np.arange(0, .009, .001), np.arange(.2, .208, .001)]
    residual = rng.normal(size=(len(times), 3))
    actual, n = endpoint.residual_coordination(residual, times, .001, .002, .006)
    expected = np.zeros((3, 3))
    count = 0
    for i, t in enumerate(times):
        for j in range(i + 1, len(times)):
            if .002 - 1e-12 <= times[j] - t <= .006 + 1e-12:
                expected += np.outer(residual[i], residual[j]) - np.outer(residual[j], residual[i])
                count += 1
    assert n == count
    np.testing.assert_allclose(actual, expected / count, atol=1e-12)
    np.testing.assert_allclose(actual, -actual.T, atol=1e-12)


def test_directional_endpoint_detects_known_order_and_sign_reversal():
    times = np.arange(20) * .001
    residual = np.zeros((20, 2))
    residual[1, 0], residual[8, 1] = 1, 1
    matrix, _ = endpoint.residual_coordination(residual, times, .001, .005, .01)
    assert matrix[0, 1] > 0
    reversed_matrix, _ = endpoint.residual_coordination(residual[:, ::-1], times, .001, .005, .01)
    assert reversed_matrix[0, 1] == -matrix[0, 1]


def test_missing_training_fold_is_explicit_not_zero_coordination():
    p = protocol()
    times = np.arange(.0005, .2, .001)
    residual, qc = endpoint.crossfit_rates(np.zeros((len(times), 2), int), times, np.zeros(len(times), int), .001, p)
    assert np.isnan(residual).all() and qc["predicted_bins"] == 0
    assert qc["folds"][0]["status"] == "no_guarded_training_bins"


def test_missing_theta_period_is_preserved_without_measurement():
    p = protocol()
    n = 10
    bank = {"pre_counts": np.zeros((n, 2), int), "pre_time_s": np.arange(n) * .001,
            "pre_theta_phase_rad": np.full((n, 1), np.nan), "pre_bin_duration_s": np.full(n, .001),
            "pre_position_cm": np.zeros((n, 2)), "pre_direction_rad": np.zeros(n), "pre_speed_cm_s": np.full(n, 15.)}
    value, qc = endpoint.period_endpoint(bank, "pre", p)
    assert value is None and qc["status"] == "no_valid_theta_endpoints"


def make_period_bank(prefix, times):
    n = len(times)
    return {f"{prefix}_counts": np.zeros((n, 2), int), f"{prefix}_time_s": times,
            f"{prefix}_theta_phase_rad": np.zeros((n, 1)), f"{prefix}_bin_duration_s": np.full(n, .001),
            f"{prefix}_position_cm": np.zeros((n, 2)), f"{prefix}_direction_rad": np.zeros(n),
            f"{prefix}_speed_cm_s": np.full(n, 15.)}


def test_future_post_spikes_cannot_change_pre_coordination():
    p = protocol()
    times = np.arange(.0005, 20, .001)
    bank = {**make_period_bank("pre", times), **make_period_bank("post", times + 100)}
    bank["pre_counts"][::100, 0] = 1
    bank["pre_counts"][7::100, 1] = 1
    before, before_qc = endpoint.period_endpoint(bank, "pre", p)
    bank["post_counts"][:] = 20
    after, after_qc = endpoint.period_endpoint(bank, "pre", p)
    np.testing.assert_array_equal(before, after)
    assert before_qc == after_qc


def test_no_physical_lag_support_cannot_be_called_measured():
    p = protocol()
    bank = make_period_bank("pre", np.array([.2005, 5.2005, 10.2005]))
    value, qc = endpoint.period_endpoint(bank, "pre", p)
    assert value is None and qc["status"] == "no_physical_lag_pairs"


def test_empty_lfp_reference_axis_is_rejected():
    bank = make_period_bank("pre", np.arange(10) * .001)
    bank["pre_theta_phase_rad"] = np.empty((10, 0))
    with pytest.raises(ValueError, match="At least one"):
        endpoint.period_endpoint(bank, "pre", protocol())


def test_zero_spike_windows_remain_in_crossfit_denominator():
    times = np.arange(.0005, 20, .001)
    residual, qc = endpoint.crossfit_rates(np.zeros((len(times), 2), int), times,
                                         np.zeros(len(times), int), .001, protocol())
    assert qc["predicted_bins"] == qc["total_bins"] == len(times)
    assert np.isfinite(residual).all()


def test_unseen_covariate_strata_are_reported_not_dropped():
    times = np.arange(.0005, 20, .001)
    labels = np.floor(times / 5).astype(int) % 2
    residual, qc = endpoint.crossfit_rates(np.zeros((len(times), 2), int), times, labels, .001, protocol())
    assert qc["unseen_stratum_fraction"] == 1
    assert qc["predicted_bins"] == len(times) and np.isfinite(residual).all()

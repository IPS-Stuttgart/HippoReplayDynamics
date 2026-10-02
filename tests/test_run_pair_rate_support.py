"""Expanded nuisance support cannot change endpoint identity or leak held-out bins."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import _run_pair_rate_glm as glm
from scripts.measure_run_pair_coordination_endpoint import matched_global_predictions, period_endpoint, support_pool
from scripts.measure_tanni_replay_run_coordination import assert_previous_bank, support_run_masks
from scripts.verify_run_pair_glm_endpoint import independently_measure_period


def data():
    root = Path(__file__).parents[1]
    p = json.loads((root / "docs/run_pair_coordination_endpoint_v5_support_protocol.json").read_text())
    p["glm_workers"] = 1
    t = np.arange(.0005, 20, .001)
    c = {"position": np.column_stack((5 + np.sin(t), 5 + np.cos(t))),
         "direction": np.mod(t, 2 * np.pi), "speed": np.full(len(t), 15.),
         "theta": (np.mod(16 * np.pi * t, 2 * np.pi) - np.pi)[:, None]}
    y = np.random.default_rng(8).poisson(.03 * np.exp(np.cos(c["theta"])), size=(len(t), 2))
    selected = (np.arange(len(t)) % 7) < 3
    bank = {"unit_ids": np.array([1, 2])}
    fields = {"position": "position_cm", "direction": "direction_rad", "speed": "speed_cm_s", "theta": "theta_phase_rad"}
    for prefix, mask in (("pre", selected), ("pre_rate_support", np.ones(len(t), bool))):
        bank.update({f"{prefix}_time_s": t[mask], f"{prefix}_counts": y[mask],
                     f"{prefix}_bin_duration_s": np.full(mask.sum(), .001)})
        bank.update({f"{prefix}_{fields[k]}": v[mask] for k, v in c.items()})
    return bank, p, t, c, y, selected


@pytest.mark.parametrize("newton", [False, True])
def test_support_fit_matches_independent_refit_without_changing_target_lag_opportunities(newton):
    bank, p, *_ = data()
    if newton:
        p["glm_group_solver"] = "scaled_scipy_lbfgs_newton"
    result, quality = period_endpoint(bank, "pre", p)
    other, qc = independently_measure_period(bank, "pre", p, newton_refit=newton)
    np.testing.assert_allclose(result, other, rtol=1e-3, atol=2e-7)
    np.testing.assert_allclose(quality["heldout_poisson_improvement_over_global"], qc["heldout_poisson_improvement_over_global"], rtol=1e-3, atol=2e-5)
    assert quality["physical_lag_opportunities"] == qc["physical_lag_opportunities"]
    assert quality["total_bins"] == len(bank["pre_time_s"]) < quality["training_pool_bins"]


@pytest.mark.parametrize("newton", [False, True])
def test_added_heldout_block_spikes_do_not_enter_prediction(newton):
    _, p, t, c, y, _ = data()
    if newton:
        p["glm_group_solver"] = "scaled_scipy_lbfgs_newton"
    prepared = glm.prepare(t, c, p)
    first = glm.crossfit(y, t, np.zeros(len(t), int), .001, p, c, prepared, return_predictions=True)[2]
    validation = prepared["folds"][0]["validation"]
    altered = y.copy()
    altered[validation] += 10
    second = glm.crossfit(altered, t, np.zeros(len(t), int), .001, p, c, prepared, return_predictions=True)[2]
    np.testing.assert_array_equal(first[validation], second[validation])
    for item in prepared["folds"]:
        assert not np.any(item["training"] & item["validation"])
        for block in np.unique(np.floor(t[item["validation"]] / 5)):
            assert not np.any(item["training"] & (t >= block * 5 - .06) & (t < (block + 1) * 5 + .06))


def test_matched_global_comparator_equals_original_target_only_glm_baseline():
    _, p, t, c, y, selected = data()
    target_times, target_counts = t[selected], y[selected]
    target_covariates = {k: v[selected] for k, v in c.items()}
    original = glm.crossfit(target_counts, target_times, np.zeros(selected.sum(), int), .001,
                           p, target_covariates, return_predictions=True)[3]
    np.testing.assert_array_equal(matched_global_predictions(target_counts, target_times, p), original)


@pytest.mark.parametrize("field", ["counts", "time_s", "speed_cm_s", "theta_phase_rad"])
def test_changed_target_identity_is_rejected(field):
    bank, p, *_ = data()
    bank[f"pre_{field}"] = bank[f"pre_{field}"].copy()
    bank[f"pre_{field}"][0] += 1
    phase = bank["pre_theta_phase_rad"]
    valid = np.isfinite(phase).all(axis=1)
    c = {k: bank[f"pre_{v}"][valid] for k, v in
         (("position", "position_cm"), ("direction", "direction_rad"), ("speed", "speed_cm_s"), ("theta", "theta_phase_rad"))}
    with pytest.raises(ValueError, match="changed|missing"):
        support_pool(bank, "pre", p, bank["pre_counts"][valid], bank["pre_time_s"][valid], c)


def test_support_masks_keep_epoch_gaps_boundaries_and_zero_spike_opportunities():
    matching = {"run_min_speed_cm_s": 10, "run_max_speed_cm_s": 200, "run_search_window_s": 2}
    links = {"t": np.arange(10.), "good": np.array([True, True, False, True, True, True, True, True, True]),
             "epoch": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]), "speed": np.full(9, 15.)}
    before, after = support_run_masks(links, {"start_s": 4., "end_s": 5., "epoch_index": 0}, matching)
    np.testing.assert_array_equal(np.flatnonzero(before), [3])
    np.testing.assert_array_equal(np.flatnonzero(after), [5, 6])
    # The mask uses no spikes or outcomes, so zero-spike opportunities remain.
    assert set(links) == {"t", "good", "epoch", "speed"}


def test_previous_bank_preserves_nan_fields_and_rejects_missing_or_changed_fields(tmp_path):
    path = tmp_path / "bank.npz"
    np.savez(path, theta=np.array([np.nan, 1.]), counts=np.zeros((2, 1), int))
    with np.load(path, allow_pickle=False) as previous:
        saved = {k: previous[k] for k in previous.files}
        assert_previous_bank({**saved, "support": np.ones(4)}, previous)
        with pytest.raises(ValueError, match="disappeared"):
            assert_previous_bank({"theta": saved["theta"]}, previous)
        with pytest.raises(ValueError, match="changed"):
            assert_previous_bank({**saved, "counts": np.ones((2, 1), int)}, previous)


def test_protocol_preserves_existing_model_and_measurement_settings():
    root = Path(__file__).parents[1] / "docs"
    old = json.loads((root / "run_pair_coordination_endpoint_v4_group_precision_protocol.json").read_text())
    new = json.loads((root / "run_pair_coordination_endpoint_v5_support_protocol.json").read_text())
    text = {"protocol_id", "frozen_before", "scope", "rate_model", "source_and_support", "claim_boundary"}
    assert set(new) - set(old) == {"rate_training_support", "rate_score_comparator"}
    assert all(new[k] == old[k] for k in set(old) - text)
    old = json.loads((root / "tanni_replay_run_measurement_v2_protocol.json").read_text())
    new = json.loads((root / "tanni_replay_run_rate_support_protocol.json").read_text())
    assert all(new[k] == v for k, v in old.items() if isinstance(v, (float, int, list, bool)))

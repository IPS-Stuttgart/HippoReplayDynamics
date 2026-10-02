"""RUN-only development grid checks, not biological association validation."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import xlogy

from scripts import _run_pair_rate_glm as glm
from scripts import audit_run_pair_rate_prediction as audit


def protocols():
    root = Path(__file__).parents[1] / "docs"
    base = json.loads((root / "run_pair_coordination_endpoint_v3_protocol.json").read_text())
    p = json.loads((root / "run_pair_rate_prediction_diagnostic_protocol.json").read_text())
    return {**base, "glm_workers": 1}, p


def fixture():
    times = np.arange(.0005, 20, .001)
    cov = {"position": np.column_stack((5 + np.sin(times), 5 + np.cos(times))),
        "direction": np.mod(times, 2 * np.pi), "speed": np.full(len(times), 15.),
        "theta": np.column_stack((np.mod(times * 16 * np.pi, 2 * np.pi) - np.pi,
                                  np.mod(times * 16 * np.pi + .4, 2 * np.pi) - np.pi))}
    rng = np.random.default_rng(11)
    counts = rng.poisson(.02 * np.exp(np.cos(cov["theta"][:, :1])), size=(len(times), 2))
    return counts, times, cov


def test_fixed_unique_grid_keeps_original_model_and_all_nuisance_terms():
    base, p = protocols()
    grid = audit.candidates(base, p)
    assert len(grid) == 12 and len({name for name, _ in grid}) == 12
    original = dict(grid)["knot16_l21_full"]
    assert original == base
    for _, candidate in grid:
        for key in set(base) - {"glm_position_knot_cm", "glm_l2_penalty", "glm_spatial_direction_interaction", "glm_spatial_theta_interaction"}:
            assert candidate[key] == base[key]
    assert grid == audit.candidates(base, p)


@pytest.mark.parametrize("key,value", [("l2_penalties", [1., 1.]), ("position_knot_cm", [0]),
                                       ("interaction_modes", ["no_theta"]), ("cells_per_pause", 0)])
def test_invalid_grid_cannot_relax_controls(key, value):
    base, p = protocols()
    with pytest.raises(ValueError):
        audit.candidates(base, {**p, key: value})


def test_cell_scores_reconcile_and_zero_spike_bins_remain_included():
    counts, times, cov = fixture()
    p, _ = protocols()
    rows, qc = audit.evaluate(counts, times, cov, np.array([31, 10]), p)
    _, independent_qc, means, global_means = glm.crossfit(counts, times, np.zeros(len(times), int), .001,
        p, cov, return_predictions=True)
    expected = np.sum(xlogy(counts, means / global_means) - means + global_means, axis=0)
    np.testing.assert_allclose([r["heldout_log_score_gain"] for r in rows], expected)
    assert sum(r["heldout_log_score_gain"] for r in rows) == pytest.approx(qc["heldout_poisson_improvement_over_global"])
    assert qc == independent_qc and qc["total_bins"] == len(times)
    assert (counts.sum(axis=1) == 0).any()
    assert [r["unit_id"] for r in rows] == [31, 10]
    assert all(r["association_fit"] is False and r["independent_biological_subject"] is False for r in rows)


def test_optional_predictions_do_not_change_default_or_leak_heldout_spikes():
    counts, times, cov = fixture()
    p, _ = protocols()
    args = (counts, times, np.zeros(len(times), int), .001, p, cov)
    residual, qc = glm.crossfit(*args)
    expanded = glm.crossfit(*args, return_predictions=True)
    np.testing.assert_array_equal(residual, expanded[0])
    assert qc == expanded[1]
    heldout = np.floor(times / 5).astype(int) % 2 == 0
    counts[heldout] += 1
    changed = glm.crossfit(*args, return_predictions=True)
    np.testing.assert_array_equal(expanded[2][heldout], changed[2][heldout])
    np.testing.assert_array_equal(expanded[3][heldout], changed[3][heldout])


def test_bank_order_and_physical_clock_are_preserved_without_reading_replay():
    counts, times, cov = fixture()
    bank = {"unit_ids": np.array([31, 10]), "pre_counts": counts, "pre_time_s": times,
            "pre_bin_duration_s": np.full(len(times), .001)}
    for key, name in (("position", "position_cm"), ("direction", "direction_rad"),
                      ("speed", "speed_cm_s"), ("theta", "theta_phase_rad")):
        bank[f"pre_{name}"] = cov[key].copy()
    bank["pre_theta_phase_rad"][100, 0] = np.nan
    original = audit.period_data(bank, "pre", 2, .001)
    bank["replay_counts"] = np.full((2, 2), np.nan)
    bank["post_counts"] = np.full((3, 2), 10000)
    changed = audit.period_data(bank, "pre", 2, .001)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[1], changed[1])
    assert original[4] == len(times) and len(original[2]) == len(times) - 1
    assert np.diff(original[2]).max() == pytest.approx(.002)


def test_missing_and_nonconverged_fits_do_not_drop_unfavorable_cells(monkeypatch):
    counts, times, cov = fixture()
    p, _ = protocols()
    def failed(*args, **kwargs):
        prediction = np.zeros(counts.shape)
        prediction[0] = np.nan
        return prediction, {"predicted_bins": len(times) - 1}, prediction, np.ones(counts.shape)
    monkeypatch.setattr(glm, "crossfit", failed)
    rows, _ = audit.evaluate(counts, times, cov, np.array([31, 10]), p)
    assert len(rows) == 2 and all(r["status"] == "incomplete_fit" and r["heldout_log_score_gain"] is None for r in rows)


def test_backwards_clock_and_changed_bin_width_are_rejected():
    counts, times, cov = fixture()
    bank = {"unit_ids": np.array([31, 10]), "pre_counts": counts, "pre_time_s": times.copy(),
            "pre_bin_duration_s": np.full(len(times), .002)}
    with pytest.raises(ValueError, match="grid"):
        audit.period_data(bank, "pre", 2, .001)
    bank["pre_time_s"][1] = times[0]
    with pytest.raises(ValueError, match="clock"):
        audit.period_data(bank, "pre", 2, .001)


def test_atomic_checkpoint_json_does_not_accept_nonfinite_values(tmp_path):
    path = tmp_path / "checkpoint.json"
    audit.atomic_json(path, {"identity": [1, 2]})
    assert json.loads(path.read_text()) == {"identity": [1, 2]}
    with pytest.raises(ValueError):
        audit.atomic_json(path, {"identity": np.nan})
    assert json.loads(path.read_text()) == {"identity": [1, 2]}

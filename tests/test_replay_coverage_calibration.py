"""Tests for inference, selection, and legacy reproduction contracts."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from hipporeplayimm.replay_coverage import continuity_metrics, decode_independent, path_metrics
from scripts.audit_replay_coverage_likelihood import compare_reproduction, likelihood_interactions, population_tables


def test_conditional_matches_closed_form_multinomial():
    rates = np.array([[1.0, 3.0], [3.0, 2.0]])
    counts = np.array([[2, 1]])
    result = decode_independent(counts, rates, np.array([[0.0], [10.0]]), 0.02, likelihood="conditional_multinomial")
    expected = np.array([(1 / 4) ** 2 * (3 / 4), (3 / 5) ** 2 * (2 / 5)])
    np.testing.assert_allclose(result["posterior"][0], expected / expected.sum())


def test_poisson_matches_legacy_formula():
    rng = np.random.default_rng(70)
    counts = rng.poisson(3, (6, 4))
    rates = rng.uniform(0.1, 10, (4, 9))
    grid = rng.uniform(0, 100, (9, 2))
    result = decode_independent(counts, rates, grid, 0.02)
    expected_counts = rates * 0.02
    logp = counts @ np.log(expected_counts) - expected_counts.sum(axis=0)[None, :]
    np.testing.assert_array_equal(result["map"], grid[np.argmax(logp, axis=1)])
    np.testing.assert_allclose(result["posterior"], softmax(logp, axis=1))


def test_fixed_total_decoder_invariant_to_location_population_gain():
    rates = np.array([[2.0, 5.0, 4.0], [4.0, 1.0, 6.0]])
    counts = np.array([[3, 1], [1, 4]])
    grid = np.arange(3.0)[:, None]
    gain = np.array([0.1, 100.0, 2.0])
    first = decode_independent(counts, rates, grid, 0.02, likelihood="conditional_multinomial")
    second = decode_independent(counts, rates * gain, grid, 0.02, likelihood="conditional_multinomial")
    np.testing.assert_allclose(first["posterior"], second["posterior"])
    assert not np.allclose(decode_independent(counts, rates, grid, 0.02)["posterior"], decode_independent(counts, rates * gain, grid, 0.02)["posterior"])


def test_models_agree_when_population_rate_constant():
    rates = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    counts = np.array([[3, 2], [0, 0]])
    grid = np.arange(3.0)[:, None]
    a = decode_independent(counts, rates, grid, 0.02)
    b = decode_independent(counts, rates, grid, 0.02, likelihood="conditional_multinomial")
    np.testing.assert_allclose(a["posterior"], b["posterior"])


def test_zero_counts_flat_only_for_conditioned_model():
    rates = np.array([[1.0, 50.0]])
    grid = np.array([[0.0], [10.0]])
    conditional = decode_independent([[0]], rates, grid, 0.02, likelihood="conditional_multinomial")
    np.testing.assert_allclose(conditional["posterior"], [[0.5, 0.5]])
    np.testing.assert_allclose(conditional["posterior_mean"], [[5.0]])
    np.testing.assert_allclose(conditional["posterior_rms_cm"], [5.0])
    assert decode_independent([[0]], rates, grid, 0.02)["posterior"][0, 0] > 0.5


@pytest.mark.parametrize("counts,rates", [([[0.5]], [[1.0]]), ([[-1]], [[1.0]]), ([[np.nan]], [[1.0]]), ([[1]], [[0.0]]), ([[1]], [[np.inf]])])
def test_invalid_observation_inputs_rejected(counts, rates):
    with pytest.raises(ValueError):
        decode_independent(counts, rates, [[0.0]], 0.02)


def test_decoder_is_permutation_equivariant_not_temporally_smoothed():
    rng = np.random.default_rng(2)
    counts = rng.poisson(2, (10, 3))
    rates = rng.uniform(0.1, 5, (3, 6))
    grid = rng.uniform(0, 10, (6, 2))
    order = rng.permutation(10)
    for likelihood in ["poisson", "conditional_multinomial"]:
        before = decode_independent(counts, rates, grid, 0.02, likelihood=likelihood)
        after = decode_independent(counts[order], rates, grid, 0.02, likelihood=likelihood)
        np.testing.assert_allclose(before["posterior"][order], after["posterior"])


def test_continuity_does_not_bridge_unsupported_bins():
    path = np.arange(12.0)[:, None] * 5
    assert continuity_metrics(path)["continuity_pass"]
    mask = np.ones(12, dtype=bool)
    mask[5] = False
    result = continuity_metrics(path, valid_bins=mask)
    assert not result["continuity_pass"]
    assert result["continuous_frames"] == 6
    path[5] = np.nan
    assert continuity_metrics(path)["continuous_frames"] == 6


def test_jump_exactly_at_threshold_breaks_run():
    result = continuity_metrics(np.array([[0.0], [5.0], [25.0], [30.0]]), min_frames=4, min_displacement_cm=10)
    assert result["continuous_frames"] == 2
    assert result["continuous_start"] == 0
    assert not result["continuity_pass"]


def test_known_speed_and_retained_run_speed():
    path = np.arange(13.0)[:, None] * 5
    result = path_metrics(path, path, step_s=0.005)
    assert result["continuity_pass"]
    assert result["median_speed_cm_s"] == 1000.0
    assert result["selected_median_speed_cm_s"] == 1000.0
    assert result["median_position_error_cm"] == 0
    assert result["large_jump_fraction"] == 0


def test_empty_and_unsupported_paths_do_not_pass():
    path = np.zeros((12, 2))
    metrics = path_metrics(path, path, step_s=0.005, valid_bins=np.zeros(12, dtype=bool))
    assert not metrics["continuity_pass"]
    assert metrics["valid_adjacent_steps"] == 0
    assert np.isnan(metrics["median_speed_cm_s"])
    assert not continuity_metrics(np.zeros((0, 2)))["continuity_pass"]


def reference_fixture():
    return pd.DataFrame([{"variant": "gaussian_exact_count", "condition": "pf_area_same_cells", "field_sigma_cm": 30, "population_replicate": 0, "event_index": 0, "event_spikes": 50, "decoded_large_jump_fraction": 0.0, "median_map_error_cm": 4.0, "foster_continuous_trajectory": True}])


def test_reproduction_missing_changed_and_duplicate_fail():
    reference = reference_fixture()
    assert compare_reproduction(reference, reference).passed.all()
    changed = reference.copy()
    changed["median_map_error_cm"] = 5
    assert not compare_reproduction(changed, reference).passed.all()
    changed["event_index"] = 1
    assert not compare_reproduction(changed, reference).passed.all()
    with pytest.raises(ValueError, match="duplicate"):
        compare_reproduction(pd.concat([reference, reference]), reference)
    assert not compare_reproduction(reference.iloc[:0], reference).passed.all()


def test_summary_uses_population_pairs_not_step_pseudoreplication():
    rows = []
    for condition, passed in [("pf_area_same_cells", True), ("tanni_area_same_cells", False), ("tanni_area_density_matched", True)]:
        for population in range(3):
            for event in range(2):
                rows.append({"variant": "gaussian_exact_count", "field_sigma_cm": 30, "likelihood": "poisson", "estimator": "map", "bin_filter": "unfiltered", "condition": condition, "population_replicate": population, "event_index": event, "continuity_pass": passed, "large_jump_fraction": 0.1, "median_speed_cm_s": 900, "median_position_error_cm": 5, "selected_median_speed_cm_s": 900 if passed else np.nan, "valid_adjacent_steps": 12})
    populations, contrasts = population_tables(pd.DataFrame(rows), 100, 1)
    assert len(populations) == 9
    subset = contrasts[(contrasts.contrast == "large_minus_small_arena") & (contrasts.metric == "pass_fraction")].iloc[0]
    assert subset.paired_populations == 3
    assert subset.comparison_minus_reference == -1
    assert subset.ci95_high == -1
    doubled = pd.concat([pd.DataFrame(rows), pd.DataFrame(rows)], ignore_index=True)
    _, more = population_tables(doubled, 100, 1)
    np.testing.assert_allclose(contrasts.comparison_minus_reference, more.comparison_minus_reference)


def test_likelihood_interaction_pairs_four_cells_within_population():
    rows = []
    for rep in range(3):
        for likelihood in ["poisson", "conditional_multinomial"]:
            for condition in ["pf_area_same_cells", "tanni_area_same_cells"]:
                value = 0.5 + rep / 10
                if condition == "tanni_area_same_cells":
                    value -= 0.2 if likelihood == "poisson" else 0.05
                rows.append({"variant": "fixed", "field_sigma_cm": 30, "estimator": "map", "bin_filter": "unfiltered", "population_replicate": rep, "likelihood": likelihood, "condition": condition, "pass_fraction": value, "large_jump_fraction": value, "median_event_speed_cm_s": value, "median_event_position_error_cm": value})
    result = likelihood_interactions(pd.DataFrame(rows), 100, 2)
    np.testing.assert_allclose(result.mean_interaction, 0.15)
    np.testing.assert_allclose(result.ci95_low, 0.15)
    assert result.paired_populations.eq(3).all()

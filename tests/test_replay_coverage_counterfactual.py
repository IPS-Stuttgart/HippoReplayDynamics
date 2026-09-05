import numpy as np
import pytest

from hipporeplayimm.replay_coverage_counterfactual import (
    aggregate_fine_counts,
    paired_counts,
    pooled_rates,
    restored_rates,
    shuffle_base_path,
)
from hipporeplayimm.replay_coverage_recovery import simulate_path


def subsets():
    return {1.: np.arange(4), .5: np.array([0, 2]), .25: np.array([2])}


def test_conserved_totals_kept_spikes_nested_exposures_and_determinism():
    rates = np.tile([20., 40., 60., 80.], (100, 1))
    first = paired_counts(rates, subsets(), [1, 3, 10, 30], .001, 4)
    second = paired_counts(rates, subsets(), [1, 3, 10, 30], .001, 4)
    assert len(first) == 28
    for key, values in first.items():
        np.testing.assert_array_equal(values, second[key])
        scale, family, fraction = key
        if family == "native":
            np.testing.assert_array_equal(values, first[scale, "native", 1.][:, subsets()[fraction]])
        else:
            np.testing.assert_array_equal(values.sum(axis=1), first[scale, "native", 1.].sum(axis=1))
            if family == "count_restored":
                assert np.all(values >= first[scale, "native", fraction])
            else:
                np.testing.assert_array_equal(values[:, :-1], first[scale, "native", fraction])
                removed = np.setdiff1d(np.arange(4), subsets()[fraction])
                np.testing.assert_array_equal(values[:, -1], first[scale, "native", 1.][:, removed].sum(axis=1))
        if scale != 30:
            assert np.all(values <= first[30, family, fraction])


def test_restored_observation_matches_adjusted_poisson_and_conditional_model():
    rates = np.tile([20., 40., 60., 80.], (50000, 1))
    obs = paired_counts(rates, subsets(), [1, 3], .01, 54)
    for scale in [1, 3]:
        y = obs[scale, "count_restored", .5]
        expected = np.array([.5, 1.5]) * scale
        np.testing.assert_allclose(y.mean(axis=0), expected, atol=.035)
        np.testing.assert_allclose(y.var(axis=0), expected, atol=.07)
        assert abs(np.cov(y.T)[0, 1]) < .05
        conditional = y[y.sum(axis=1) == 2]
        np.testing.assert_allclose(conditional.mean(axis=0), [.5, 1.5], atol=.04)
    maps = np.array([[20., 40.], [40., 20.], [60., 60.], [80., 80.]])
    restored = restored_rates(maps, subsets()[.5])
    pooled = pooled_rates(maps, subsets()[.5])
    np.testing.assert_array_equal(pooled, np.vstack([maps[[0, 2]], maps[[1, 3]].sum(axis=0)]))
    np.testing.assert_allclose(pooled.sum(axis=0), maps.sum(axis=0))
    np.testing.assert_allclose(restored.sum(axis=0), maps.sum(axis=0))
    np.testing.assert_allclose(restored / restored.sum(axis=0), maps[[0, 2]] / maps[[0, 2]].sum(axis=0))


@pytest.mark.parametrize("scales", [[], [1, 1], [0, 1], [np.nan], [-1]])
def test_reject_invalid_exposures(scales):
    with pytest.raises(ValueError):
        paired_counts(np.ones((10, 4)), subsets(), scales, .001, 1)


def test_missing_full_reference_rejected():
    with pytest.raises(ValueError):
        paired_counts(np.ones((10, 4)), {.5: np.array([0, 1])}, [1], .001, 1)


def test_pooling_does_not_increase_pairwise_position_information():
    from scipy.special import rel_entr

    rates = np.random.default_rng(15).lognormal(size=(12, 20))
    pooled = pooled_rates(rates, np.array([0, 3, 7, 9]))
    for a in range(20):
        for b in range(20):
            full_kl = np.sum(rel_entr(rates[:, a], rates[:, b]) - rates[:, a] + rates[:, b])
            pooled_kl = np.sum(rel_entr(pooled[:, a], pooled[:, b]) - pooled[:, a] + pooled[:, b])
            assert pooled_kl <= full_kl + 1e-12
            p, q = rates[:, a] / rates[:, a].sum(), rates[:, b] / rates[:, b].sum()
            pooled_p, pooled_q = pooled[:, a] / pooled[:, a].sum(), pooled[:, b] / pooled[:, b].sum()
            assert np.sum(rel_entr(pooled_p, pooled_q)) <= np.sum(rel_entr(p, q)) + 1e-12


def test_base_aggregation_and_zero_spike_restoration():
    obs = paired_counts(np.full((100, 4), 1e-20), subsets(), [1, 3], .001, 3)
    assert all(values.sum() == 0 for values in obs.values())
    assert aggregate_fine_counts(obs[1, "native", 1.], 5).shape == (20, 4)
    with pytest.raises(ValueError):
        aggregate_fine_counts(np.zeros((9, 4)), 5)


def test_whole_base_shuffle_preserves_blocks_and_population_snapshots():
    path = simulate_path(20, [[0, 0], [200, 200]], "continuous", 0, 1000, 5)
    permutation = np.random.default_rng(6).permutation(20)
    shuffled = shuffle_base_path(path, permutation)
    expected = path["midpoints_cm"].reshape(20, 5, 2)[permutation]
    np.testing.assert_array_equal(shuffled["midpoints_cm"].reshape(20, 5, 2), expected)
    np.testing.assert_array_equal(shuffled["edges_cm"][:-1].reshape(20, 5, 2), path["edges_cm"][:-1].reshape(20, 5, 2)[permutation])
    assert np.isnan(shuffled["speed_cm_s"]).all()
    with pytest.raises(ValueError):
        shuffle_base_path(path, np.zeros(20, int))

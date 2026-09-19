import numpy as np

from scripts.verify_tirole_evaluation_neuron_influence import odds, permutations_without, reference_interval


def test_independent_cycle_deletion():
    p = np.array([[1, 2, 0, 3], [0, 1, 3, 2]])
    np.testing.assert_array_equal(permutations_without(p, 1), [[1, 0, 2], [0, 2, 1]])


def test_equal_context_maps_have_zero_odds():
    counts = np.array([[1, 2], [0, 0], [2, 0]])
    maps = np.ones((2, 2, 3))
    valid = np.ones((2, 3), bool)
    for conditional in [False, True]:
        assert odds(counts, maps, valid, conditional) == 0
        assert np.isnan(odds(counts * 0, maps, valid, conditional))


def test_independent_interval_missingness():
    result = reference_interval(np.array([1.0, 3.0, np.nan]), np.array([0, 1, 2]), np.array([[1, 1, 1], [2, 1, 0], [1, 2, 0]]))
    assert result[0] == 2 and result[3] == 2 and result[4] == 2
    result = reference_interval(np.array([np.nan, np.nan]), np.array([0, 1]), np.array([[1, 1], [2, 0]]))
    assert np.isnan(result[:3]).all() and result[3] == 0

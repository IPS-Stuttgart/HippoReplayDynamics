import numpy as np

from scripts.report_regional_blind_bank_population_gaps import endpoint_counts


def test_endpoint_readout_keeps_native_half_open_boundaries():
    t = {"endpoints": np.array([.1]), "offsets": np.array([0, 4]), "times": np.array([.079, .08, .099, .1])}
    counts = endpoint_counts(np.array([0, 1, 0, 1]), t, 2, np.array([.08]))
    np.testing.assert_array_equal(counts, [[1, 1]])


def test_endpoint_readout_does_not_reconstruct_frozen_start():
    t = {"endpoints": np.array([.1]), "offsets": np.array([0, 2]), "times": np.array([.08, .09])}
    counts = endpoint_counts(np.array([0, 1]), t, 2, np.array([.080000000001]))
    np.testing.assert_array_equal(counts, [[0, 1]])

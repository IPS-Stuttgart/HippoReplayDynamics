import numpy as np

from scripts.audit_tirole_experience_coverage import coverage_subsets, run_information
from scripts.verify_tirole_experience_coverage import reference_cube, reference_information, reference_statistics, reference_subsets


def test_independent_information_and_subset_construction():
    rng = np.random.default_rng(1909)
    rates = rng.uniform(0.01, 15, (2, 13, 30))
    occupancy = rng.uniform(0, 100, (2, 30))
    valid = rng.random((2, 30)) > 0.1
    actual, mean = reference_information(rates, occupancy, valid)
    expected, expected_mean = run_information(rates, occupancy, valid)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(mean, expected_mean)
    for n in [12, 13]:
        independent, pairs, shared = reference_subsets(list(range(n)), actual, mean, "test", 0)
        original, original_pairs, original_shared = coverage_subsets(np.arange(n), expected, expected_mean, "test", 0)
        assert independent == {k: v.tolist() for k, v in original.items()}
        assert pairs == original_pairs.tolist() and shared == original_shared.tolist()


def test_tensor_reference_keeps_unsupported_and_empty_sets_undefined():
    shape = (4, 5, 13)
    selected = np.zeros(shape, bool)
    selected[0, :, 1] = True
    selected[3, :, 2] = True
    values = np.broadcast_to(np.array([0.2, 0.4, 0.6, 0.8])[:, None, None], shape)
    cube = reference_cube(selected, np.ones(shape, bool), np.full(shape, 2), values, values, np.full(shape, 0.2), np.full(shape, 10), np.full(shape, 12))
    result = reference_statistics(cube, np.ones(4))
    np.testing.assert_allclose(result["conditional_track2_score_mean"][0, 1:3], [0.2, 0.8])
    assert np.isnan(result["conditional_track2_score_mean"][0, 0])
    assert result["selected_event_equivalents"][0, 1] == 1
    missing = reference_cube(selected, np.ones(shape, bool), np.full(shape, 2), np.full(shape, np.nan), values, np.full(shape, 0.2), np.full(shape, 10), np.full(shape, 12))
    assert np.isnan(reference_statistics(missing, np.ones(4))["conditional_track2_score_mean"]).all()

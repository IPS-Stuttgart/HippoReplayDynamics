import numpy as np

from scripts.audit_regional_prevalence_calibration import independently_decode, raw_counts_by_assignment, reconstruct_response
from scripts.measure_regional_prevalence_calibration import fit_response, regional_scores


def test_dense_audit_reconstructs_overlapping_cell_populations():
    rng = np.random.default_rng(823)
    counts = rng.poisson(2, (80, 8))
    rates = rng.gamma(2, 3, (8, 21)) + .0001
    near = np.arange(21) < 5
    groups = [np.array([0, 2, 4, 7]), np.array([2, 4, 5, 6])]
    np.testing.assert_allclose(independently_decode(counts, rates, near, groups),
                               regional_scores(counts, rates, near, groups), atol=1e-12)


def test_assignment_audit_ignores_gap_spikes_and_unknown_cells():
    spikes = np.array([[0, 8], [.02, 8], [.5, 8], [1, 8], [1.005, 900], [1.01, 8], [2, 8]])
    result = raw_counts_by_assignment(spikes, np.array([1., 0.]), np.array([1.02, .02]), np.array([8]))
    np.testing.assert_array_equal(result, [[2], [1]])


def test_response_audit_matches_all_statuses():
    labels = np.repeat([False, True], 120)
    for values in ((.1, .8), (.1, .11), (.9, .2)):
        scores = np.repeat(values, 120)
        a, b = fit_response(scores, labels), reconstruct_response(scores, labels)
        assert a == b

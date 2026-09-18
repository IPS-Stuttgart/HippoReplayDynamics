import numpy as np

from hipporeplayimm.two_track_content import field_shift_posteriors
from scripts.verify_tirole_likelihood_selection import conditional_posterior, extended_ratios


def test_scalar_conditional_reference_matches_vectorized_and_omits_empty_bins():
    rng = np.random.default_rng(193)
    counts = rng.poisson(0.5, (8, 7))
    counts[2] = 0
    rates = rng.uniform(1e-6, 20, (2, 7, 12))
    valid = rng.random((2, 12)) > 0.2
    actual = conditional_posterior(counts, rates, valid)
    expected = field_shift_posteriors(counts, rates, valid, np.zeros((1, 2, 7), int), conditional_count=True)[0]
    expected[counts.sum(axis=1) == 0] = 0
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.all(actual[2] == 0)


def test_reference_distortion_uses_known_truth_not_reported_labels():
    cube = np.zeros((3, 2, 5, 7))
    cube[:, :, 0, 0] = 1
    cube[:, :, 0, 1] = 0.9
    cube[:, 1, 1, 0] = 1
    cube[:, 1, 1, 1] = 0
    result = extended_ratios(cube, np.ones((1, 3)))
    assert result["full_true_track2_fraction"][0] == 0.5
    assert result["half_true_track2_fraction"][0] == 1
    assert result["full_absolute_distortion"][0] == 0
    assert result["half_absolute_distortion"][0] == 0.5
    assert result["half_reported_track2_fraction"][0] == 0
    assert np.isnan(result["lost_true_track2_fraction"][0])

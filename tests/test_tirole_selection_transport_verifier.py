import numpy as np

from scripts.verify_tirole_selection_transport import reference_ratios, reference_statistics


def test_tensor_reference_separates_true_selection_and_labels():
    flags = np.zeros((2, 1, 2, 2, 1, 3), bool)
    flags[..., 0] = True
    flags[:, :, :, 1, :, 1:] = True
    labels = np.broadcast_to(np.array([1, 2])[None, None, None, :, None, None], flags.shape).copy()
    z = np.ones((2, 1, 2, 2, 2))
    cube = reference_statistics(flags, labels, z, np.array([0, 1]))
    values = reference_ratios(cube[:, :, 0], np.array([[1, 1], [2, 0]]))
    np.testing.assert_allclose(values["full_true_track2_fraction"], 0.5)
    np.testing.assert_allclose(values["half_true_track2_fraction"], 1.0)
    np.testing.assert_allclose(values["retained_minus_full_true_track2_fraction"], 0.5)
    np.testing.assert_allclose(values["lost_conditional_count_true_signed_z"], 1.0)
    assert np.isnan(values["gained_true_track2_fraction"]).all()


def test_tensor_reference_keeps_missing_evaluation_distinct_from_zero_signal():
    flags = np.ones((2, 1, 2, 2, 1, 3), bool)
    labels = np.ones(flags.shape, int)
    z = np.full((2, 1, 2, 2, 2), np.nan)
    cube = reference_statistics(flags, labels, z, np.array([0, 1]))
    values = reference_ratios(cube[:, :, 0], np.ones(2))
    assert np.isnan(values["full_poisson_true_signed_z"]).all()
    np.testing.assert_allclose(values["full_true_track2_fraction"], 0.5)
    np.testing.assert_allclose(values["full_reported_track2_fraction"], 0.0)
    np.testing.assert_allclose(values["full_correct_label_fraction"], 0.5)

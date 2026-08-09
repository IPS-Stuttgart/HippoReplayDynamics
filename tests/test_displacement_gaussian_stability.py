from __future__ import annotations

import numpy as np

import hipporeplayimm
from hipporeplayimm import state_space_displacement_imm as displacement_imm
from hipporeplayimm import state_space_displacement_momentum as displacement_momentum


def _two_extreme_points() -> np.ndarray:
    return np.array([[-1.0e308], [1.0e308]], dtype=float)


def _expected_two_point_gaussian() -> np.ndarray:
    far_weight = np.exp(-2.0) / (1.0 + np.exp(-2.0))
    return np.array(
        [
            [1.0 - far_weight, far_weight],
            [far_weight, 1.0 - far_weight],
        ],
        dtype=float,
    )


def test_shifted_gaussian_transition_preserves_extreme_scale_weights() -> None:
    hipporeplayimm.apply_runtime_patches()
    centers = _two_extreme_points()

    transition = displacement_momentum._shifted_gaussian_transition_matrix(
        centers,
        displacement=np.array([0.0], dtype=float),
        sigma_cm=1.0e308,
        max_step_sigma=3.0,
    )

    np.testing.assert_allclose(
        transition.toarray(),
        _expected_two_point_gaussian(),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert (
        displacement_imm._shifted_gaussian_transition_matrix
        is displacement_momentum._shifted_gaussian_transition_matrix
    )


def test_displacement_transition_preserves_extreme_scale_weights() -> None:
    hipporeplayimm.apply_runtime_patches()
    vectors = _two_extreme_points()

    transition = displacement_momentum._displacement_transition_matrix(
        vectors,
        sigma_cm=1.0e308,
        decay=1.0,
    )

    np.testing.assert_allclose(
        transition,
        _expected_two_point_gaussian(),
        rtol=1.0e-12,
        atol=0.0,
    )
    assert (
        displacement_imm._displacement_transition_matrix
        is displacement_momentum._displacement_transition_matrix
    )


def test_displacement_prior_preserves_extreme_scale_weights() -> None:
    hipporeplayimm.apply_runtime_patches()
    vectors = np.array([[-1.0e308], [0.0], [1.0e308]], dtype=float)
    side_weight = np.exp(-0.5)
    expected = np.array([side_weight, 1.0, side_weight], dtype=float)
    expected /= expected.sum()

    prior = displacement_momentum._zero_centered_displacement_prior(
        vectors,
        sigma_cm=1.0e308,
    )

    np.testing.assert_allclose(prior, expected, rtol=1.0e-12, atol=0.0)
    assert (
        displacement_imm._zero_centered_displacement_prior
        is displacement_momentum._zero_centered_displacement_prior
    )

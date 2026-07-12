from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from hipporeplayimm import state_space_sparse_momentum, state_space_utils


def test_dense_gaussian_transition_preserves_nearest_mass_after_underflow() -> None:
    centers = np.array([[0.0], [0.9], [1.0]], dtype=float)
    transition = state_space_utils._gaussian_transition_matrix(
        centers,
        sigma_cm=1.0e-3,
        max_step_sigma=1000.0,
        valid_bin_mask=np.array([False, True, True]),
    )

    source_zero = transition[:, 0].toarray().reshape(-1)
    np.testing.assert_allclose(
        source_zero,
        np.array([0.0, 1.0, 0.0]),
        atol=1.0e-12,
    )


def test_sparse_gaussian_row_preserves_nearest_mass_after_underflow() -> None:
    centers = np.array([[0.9], [1.0]], dtype=float)
    valid_indices = np.arange(centers.shape[0], dtype=int)
    destinations, weights = state_space_sparse_momentum._finite_gaussian_row(
        centers,
        valid_indices,
        cKDTree(centers),
        np.array([0.0]),
        sigma_cm=1.0e-3,
        max_step_sigma=1000.0,
    )

    by_destination = dict(zip(destinations.tolist(), weights.tolist(), strict=True))
    assert by_destination[0] > 1.0 - 1.0e-12
    assert by_destination[1] < 1.0e-12
    assert sum(by_destination.values()) == 1.0

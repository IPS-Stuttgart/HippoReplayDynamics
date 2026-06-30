from __future__ import annotations

import numpy as np
import pytest

from hipporeplayimm import state_space as state_space_public
from hipporeplayimm import state_space_sparse_momentum as sparse_momentum


@pytest.mark.parametrize(
    ("centers", "message"),
    [
        (np.array([[0.0], [np.nan]], dtype=float), "finite"),
        (np.array([[0.0], [np.inf]], dtype=float), "finite"),
        (np.empty((0, 1), dtype=float), "shape"),
        (np.empty((2, 0), dtype=float), "shape"),
    ],
)
def test_sparse_momentum_bin_center_validation_rejects_invalid_centers(
    centers: np.ndarray,
    message: str,
) -> None:
    assert state_space_public.StateSpaceReplayModel is not None

    with pytest.raises(ValueError, match=message):
        sparse_momentum._as_2d_centers(centers)


def test_sparse_momentum_bin_center_validation_preserves_one_dimensional_grid() -> None:
    assert state_space_public.StateSpaceReplayModel is not None

    centers = sparse_momentum._as_2d_centers(np.array([0.0, 1.0, 2.0], dtype=float))

    assert centers.shape == (3, 1)
    np.testing.assert_allclose(centers[:, 0], [0.0, 1.0, 2.0])

from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm.state_space import _displacement_lattice


def test_displacement_lattice_normalizes_arbitrary_precision_radius_overflow() -> None:
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(
        ValueError,
        match=r"displacement_radius_bins.*fit into integer range",
    ):
        _displacement_lattice(
            np.array([[0.0], [1.0]], dtype=float),
            radius_bins=10**10000,
        )

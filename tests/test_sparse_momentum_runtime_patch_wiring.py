from __future__ import annotations

import importlib

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import state_space_sparse_momentum as sparse_momentum


def test_runtime_patches_wire_sparse_momentum_validators_without_state_space_import() -> None:
    """Package-level runtime patching must harden direct sparse-momentum imports."""

    module = importlib.reload(sparse_momentum)
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="bin_centers must be finite"):
        module._as_2d_centers(np.array([[0.0], [np.nan]]))

    with pytest.raises(ValueError, match="boolean or 0/1"):
        module._coerce_valid_bin_mask(np.array(["1", "0"], dtype=str), 2)

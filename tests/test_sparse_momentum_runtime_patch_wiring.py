from __future__ import annotations

import importlib

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import state_space_sparse_momentum as sparse_momentum
from hipporeplayimm import state_space_trajectory_imm as trajectory_imm


def test_runtime_patches_refresh_trajectory_imm_sparse_helper_aliases() -> None:
    """Consumers imported by value must not retain pre-patch sparse helpers."""

    hipporeplayimm.apply_runtime_patches()

    assert trajectory_imm._as_2d_centers is sparse_momentum._as_2d_centers
    assert trajectory_imm._coerce_valid_bin_mask is sparse_momentum._coerce_valid_bin_mask

    with pytest.raises(ValueError, match="bin_centers must be finite"):
        trajectory_imm._as_2d_centers(np.array([[0.0], [np.nan]]))

    with pytest.raises(ValueError, match="boolean or 0/1"):
        trajectory_imm._coerce_valid_bin_mask(np.array([2, 0], dtype=int), 2)

    with pytest.raises(ValueError, match="finite"):
        trajectory_imm._coerce_valid_bin_mask(np.array([1.0, np.nan]), 2)


def test_runtime_patches_wire_sparse_momentum_validators_without_state_space_import() -> None:
    """Package-level runtime patching must harden direct sparse-momentum imports."""

    module = importlib.reload(sparse_momentum)
    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(ValueError, match="bin_centers must be finite"):
        module._as_2d_centers(np.array([[0.0], [np.nan]]))

    with pytest.raises(ValueError, match="boolean or 0/1"):
        module._coerce_valid_bin_mask(np.array(["1", "0"], dtype=str), 2)

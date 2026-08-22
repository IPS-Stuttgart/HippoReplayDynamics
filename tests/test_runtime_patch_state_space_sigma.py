from __future__ import annotations

import pytest

import hipporeplayimm
from hipporeplayimm import state_space_utils


def test_apply_runtime_patches_restores_state_space_sigma_validation(monkeypatch):
    patched = state_space_utils._per_bin_sigma
    original = getattr(patched, "__hipporeplayimm_original__", patched)
    monkeypatch.setattr(state_space_utils, "_per_bin_sigma", original)

    # The raw helper accepts booleans through float(True) == 1.0.
    state_space_utils._per_bin_sigma(True, 0.003)

    hipporeplayimm.apply_runtime_patches()

    with pytest.raises(TypeError, match="sigma_cm_sqrt_s.*boolean"):
        state_space_utils._per_bin_sigma(True, 0.003)

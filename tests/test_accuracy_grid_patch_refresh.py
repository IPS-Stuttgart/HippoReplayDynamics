from __future__ import annotations

import numpy as np
import pytest

import hipporeplayimm
from hipporeplayimm import accuracy_upgrades
from hipporeplayimm.accuracy_grid_parameter_validation import _PATCHED_FLAG


def _unwrap_helper(helper):
    seen: set[int] = set()
    current = helper
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is not None:
            current = wrapped
            continue
        original = getattr(current, "__hipporeplayimm_original__", None)
        if original is not None:
            current = original
            continue
        return current
    return current


def test_accuracy_grid_parameter_patch_refreshes_replaced_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    hipporeplayimm.apply_runtime_patches()
    assert getattr(accuracy_upgrades, _PATCHED_FLAG, False)

    raw_transition = _unwrap_helper(accuracy_upgrades.valid_grid_graph_transition)
    raw_init = _unwrap_helper(accuracy_upgrades.ValidStateGridReplayModel.__init__)

    monkeypatch.setattr(accuracy_upgrades, "valid_grid_graph_transition", raw_transition)
    monkeypatch.setattr(accuracy_upgrades.ValidStateGridReplayModel, "__init__", raw_init)

    # Reproduce the stale-sentinel state seen after reloads or direct helper restoration:
    # the module is marked patched, but the callables no longer carry the validation wrappers.
    assert getattr(accuracy_upgrades, _PATCHED_FLAG, False)

    hipporeplayimm.apply_runtime_patches()

    valid_mask = np.ones(4, dtype=bool)
    with pytest.raises(TypeError, match="diagonal_neighbors must be boolean"):
        accuracy_upgrades.valid_grid_graph_transition((2, 2), valid_mask, diagonal_neighbors="False")

    with pytest.raises(TypeError, match="diagonal_neighbors must be boolean"):
        accuracy_upgrades.ValidStateGridReplayModel(valid_mask, grid_shape=(2, 2), diagonal_neighbors="False")

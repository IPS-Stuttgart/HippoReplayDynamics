from __future__ import annotations

import sys
from types import ModuleType

import hipporeplayimm.first_order_imm_diagnostics_validation as diagnostics_validation
import hipporeplayimm.state_space_utils as state_space_utils


def test_mode_transition_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    def previous(n_modes, stickiness):
        return n_modes, stickiness

    external = ModuleType("hipporeplayimm_extension")
    external._mode_transition_matrix = previous
    package_child = ModuleType("hipporeplayimm._mode_transition_alias_probe")
    package_child._mode_transition_matrix = previous

    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)
    monkeypatch.setattr(state_space_utils, "_mode_transition_matrix", previous)

    diagnostics_validation._patch_mode_transition_count_validation()

    replacement = state_space_utils._mode_transition_matrix
    assert replacement is not previous
    assert external._mode_transition_matrix is previous
    assert package_child._mode_transition_matrix is replacement

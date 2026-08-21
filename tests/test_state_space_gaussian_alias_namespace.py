from __future__ import annotations

import sys
from types import ModuleType

import hipporeplayimm.state_space_gaussian_scalar_validation as gaussian_validation


def test_gaussian_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    def previous(*args, **kwargs):
        return args, kwargs

    def replacement(*args, **kwargs):
        return args, kwargs

    external = ModuleType("hipporeplayimm_extension")
    external._gaussian_transition_matrix = previous
    package_child = ModuleType("hipporeplayimm._gaussian_alias_probe")
    package_child._gaussian_transition_matrix = previous

    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    gaussian_validation._synchronize_aliases(
        "_gaussian_transition_matrix",
        previous,
        replacement,
    )

    assert external._gaussian_transition_matrix is previous
    assert package_child._gaussian_transition_matrix is replacement

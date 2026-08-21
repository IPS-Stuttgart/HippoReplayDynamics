from __future__ import annotations

import sys
from types import ModuleType

from hipporeplayimm.state_space_occupancy_threshold_validation import _sync_aliases


def test_occupancy_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    def previous(*args, **kwargs):
        return args, kwargs

    def replacement(*args, **kwargs):
        return args, kwargs

    external = ModuleType("hipporeplayimm_extension")
    external._valid_bin_mask_from_occupancy = previous
    package_child = ModuleType("hipporeplayimm._occupancy_alias_probe")
    package_child._valid_bin_mask_from_occupancy = previous

    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    _sync_aliases(previous, replacement)

    assert external._valid_bin_mask_from_occupancy is previous
    assert package_child._valid_bin_mask_from_occupancy is replacement

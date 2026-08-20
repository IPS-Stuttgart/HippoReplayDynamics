from __future__ import annotations

import sys
from types import ModuleType

from hipporeplayimm.continuous_time_imm_transition_patch import _synchronize_aliases


def test_continuous_time_alias_sync_respects_package_namespace(monkeypatch) -> None:
    def previous_helper() -> None:
        return None

    def active_helper() -> None:
        return None

    external = ModuleType("hipporeplayimm_extension")
    external.imported_helper = previous_helper
    internal = ModuleType("hipporeplayimm._continuous_time_namespace_test")
    internal.imported_helper = previous_helper

    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    _synchronize_aliases(previous_helper, active_helper)

    assert external.imported_helper is previous_helper
    assert internal.imported_helper is active_helper

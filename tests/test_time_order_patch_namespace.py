from __future__ import annotations

import sys
from types import ModuleType

from hipporeplayimm.time_order_patch import _synchronize_duration_resolver_aliases


def test_duration_resolver_alias_sync_respects_package_namespace(monkeypatch) -> None:
    original = object()
    replacement = object()

    external = ModuleType("hipporeplayimm_extension")
    external.transition_durations_s = original
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = ModuleType("hipporeplayimm._time_order_patch_test")
    internal.transition_durations_s = original
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    _synchronize_duration_resolver_aliases(original, replacement)

    assert external.transition_durations_s is original
    assert internal.transition_durations_s is replacement

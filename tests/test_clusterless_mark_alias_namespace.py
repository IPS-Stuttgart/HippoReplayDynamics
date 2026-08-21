from __future__ import annotations

import sys
import types

from hipporeplayimm.clusterless_mark_group_validation import (
    _synchronize_build_emission_aliases,
)


def test_clusterless_mark_alias_sync_respects_package_namespace(monkeypatch) -> None:
    previous = object()
    patched = object()

    external = types.ModuleType("hipporeplayimm_extension")
    external.build_clusterless_mark_emissions = previous
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = types.ModuleType("hipporeplayimm._clusterless_mark_alias_probe")
    internal.build_clusterless_mark_emissions = previous
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    _synchronize_build_emission_aliases(previous, patched)

    assert external.build_clusterless_mark_emissions is previous
    assert internal.build_clusterless_mark_emissions is patched

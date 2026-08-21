from __future__ import annotations

import sys
import types

from hipporeplayimm.clusterless_config_validation import _synchronize_aliases


def test_clusterless_alias_sync_stays_within_package_namespace(monkeypatch) -> None:
    previous = object()
    patched = object()

    foreign = types.ModuleType("hipporeplayimm_extension")
    foreign.fit_clusterless_mark_encoding = previous
    package_local = types.ModuleType("hipporeplayimm._clusterless_alias_probe")
    package_local.fit_clusterless_mark_encoding = previous

    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    monkeypatch.setitem(sys.modules, package_local.__name__, package_local)

    _synchronize_aliases(previous, patched)

    assert foreign.fit_clusterless_mark_encoding is previous
    assert package_local.fit_clusterless_mark_encoding is patched

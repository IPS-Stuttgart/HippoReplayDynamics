from __future__ import annotations

import sys
import types

import pytest

from hipporeplayimm.data_cell_id_validation import _synchronize_coerce_ripple_event_aliases


def test_data_cell_alias_sync_ignores_similarly_named_top_level_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = object()
    replacement = object()
    unrelated = types.ModuleType("hipporeplayimm_extension")
    package_child = types.ModuleType("hipporeplayimm._data_cell_alias_probe")
    unrelated._coerce_ripple_event = original
    package_child._coerce_ripple_event = original
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    _synchronize_coerce_ripple_event_aliases(original, replacement)

    assert unrelated._coerce_ripple_event is original
    assert package_child._coerce_ripple_event is replacement

"""Regression tests for run-local kinematics alias synchronization."""

from __future__ import annotations

import sys
from types import ModuleType

from hipporeplayimm.place_field_run_local_kinematics import _synchronize_aliases


def _original() -> None:
    pass


def _replacement() -> None:
    pass


def test_run_local_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    foreign = ModuleType("hipporeplayimm_extension")
    package_alias = ModuleType("hipporeplayimm._run_local_alias_probe")
    foreign.fit_place_field_encoding = _original
    package_alias.fit_place_field_encoding = _original

    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    monkeypatch.setitem(sys.modules, package_alias.__name__, package_alias)

    _synchronize_aliases(
        "fit_place_field_encoding",
        _original,
        _replacement,
    )

    assert foreign.fit_place_field_encoding is _original
    assert package_alias.fit_place_field_encoding is _replacement

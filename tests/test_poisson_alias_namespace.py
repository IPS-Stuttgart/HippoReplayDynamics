from __future__ import annotations

import sys
import types

import pytest

from hipporeplayimm.poisson_input_boolean_validation import (
    _synchronize_poisson_log_emission_aliases,
)


def test_poisson_alias_sync_ignores_similarly_named_top_level_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def original() -> None:
        return None

    def active() -> None:
        return None

    unrelated = types.ModuleType("hipporeplayimm_extension")
    package_child = types.ModuleType("hipporeplayimm._poisson_alias_probe")
    unrelated._poisson_log_emissions = original
    package_child._poisson_log_emissions = original
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    _synchronize_poisson_log_emission_aliases(original, active)

    assert unrelated._poisson_log_emissions is original
    assert package_child._poisson_log_emissions is active

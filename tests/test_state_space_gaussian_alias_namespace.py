from __future__ import annotations

import sys
import types

import pytest

from hipporeplayimm import state_space_gaussian_scalar_validation as gaussian_validation


def test_gaussian_alias_sync_stays_inside_package_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "_pairwise_gaussian_log_prob"
    original = object()
    replacement = object()

    package_module = types.ModuleType("hipporeplayimm._gaussian_alias_test")
    external_module = types.ModuleType("hipporeplayimm_extension")
    setattr(package_module, name, original)
    setattr(external_module, name, original)
    monkeypatch.setitem(sys.modules, package_module.__name__, package_module)
    monkeypatch.setitem(sys.modules, external_module.__name__, external_module)

    gaussian_validation._synchronize_aliases(name, original, replacement)

    assert getattr(package_module, name) is replacement
    assert getattr(external_module, name) is original

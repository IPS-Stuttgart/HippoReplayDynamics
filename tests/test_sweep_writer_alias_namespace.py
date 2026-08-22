from __future__ import annotations

import sys
from types import ModuleType

import hipporeplayimm.sweep_seed_validation as sweep_validation


def test_sweep_writer_alias_sync_stays_inside_package_namespace(monkeypatch) -> None:
    def previous(*args, **kwargs):
        return args, kwargs

    def active(*args, **kwargs):
        return args, kwargs

    setattr(active, sweep_validation._ORIGINAL_ATTR, previous)

    external = ModuleType("hipporeplayimm_extension")
    external.write_pyrecest_sweep_outputs = previous
    package_child = ModuleType("hipporeplayimm._sweep_writer_alias_probe")
    package_child.write_pyrecest_sweep_outputs = previous

    monkeypatch.setitem(sys.modules, external.__name__, external)
    monkeypatch.setitem(sys.modules, package_child.__name__, package_child)

    sweep_validation._synchronize_output_writer_aliases(active)

    assert external.write_pyrecest_sweep_outputs is previous
    assert package_child.write_pyrecest_sweep_outputs is active

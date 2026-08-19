import sys
import types

import hipporeplayimm
from hipporeplayimm import encoding, kd_reference


def test_duration_builder_sync_respects_package_namespace(monkeypatch):
    external = types.ModuleType("hipporeplayimm_extension")
    external_build_emissions = object()
    external_build_kd_emissions = object()
    external.build_emissions = external_build_emissions
    external.build_kd_emissions = external_build_kd_emissions
    monkeypatch.setitem(sys.modules, external.__name__, external)

    internal = types.ModuleType("hipporeplayimm._runtime_patch_test")
    internal.build_emissions = object()
    internal.build_kd_emissions = object()
    monkeypatch.setitem(sys.modules, internal.__name__, internal)

    hipporeplayimm._synchronize_duration_patched_emission_builders()

    assert external.build_emissions is external_build_emissions
    assert external.build_kd_emissions is external_build_kd_emissions
    assert internal.build_emissions is encoding.build_emissions
    assert internal.build_kd_emissions is kd_reference.build_kd_emissions

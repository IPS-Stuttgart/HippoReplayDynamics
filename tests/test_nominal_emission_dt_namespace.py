from __future__ import annotations

import sys
from types import ModuleType

import hipporeplayimm.clusterless as clusterless
import hipporeplayimm.encoding as encoding
import hipporeplayimm.kd_reference as kd_reference
from hipporeplayimm.nominal_emission_dt import _synchronize_builder_aliases


def test_nominal_emission_alias_sync_does_not_patch_similarly_prefixed_external_module(
    monkeypatch,
) -> None:
    external = ModuleType("hipporeplayimm_extension")

    def external_build_emissions():
        return None

    def external_build_kd_emissions():
        return None

    def external_build_clusterless_mark_emissions():
        return None

    external.build_emissions = external_build_emissions
    external.build_kd_emissions = external_build_kd_emissions
    external.build_clusterless_mark_emissions = external_build_clusterless_mark_emissions
    monkeypatch.setitem(sys.modules, external.__name__, external)

    _synchronize_builder_aliases(
        build_emissions=encoding.build_emissions,
        build_kd_emissions=kd_reference.build_kd_emissions,
        build_clusterless_mark_emissions=clusterless.build_clusterless_mark_emissions,
    )

    assert external.build_emissions is external_build_emissions
    assert external.build_kd_emissions is external_build_kd_emissions
    assert (
        external.build_clusterless_mark_emissions
        is external_build_clusterless_mark_emissions
    )
